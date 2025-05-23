"""
PDF file processor with memory optimization using PyPDF2/pdfplumber.
"""
import os
import io
import tempfile
from typing import Dict, Generator, List, Optional, Any, Union, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm
import PyPDF2
import pdfplumber
from PIL import Image

from app.utils.logging_config import get_logger
from app.processors.base_processor import BaseProcessor

logger = get_logger(__name__)

class PDFProcessor(BaseProcessor):
    """
    Process PDF files with memory optimization.
    
    This class handles PDF files efficiently by:
    1. Processing pages individually to minimize memory usage
    2. Using different extraction strategies based on PDF content
    3. Cleaning up resources after each page
    """
    
    def __init__(self, **kwargs):
        """Initialize the PDF processor with shared parameters."""
        super().__init__(**kwargs)
        # PDF-specific settings
        self.extract_tables = kwargs.get('extract_tables', True)
        self.extract_text = kwargs.get('extract_text', True)
        self.table_extraction_method = kwargs.get('table_extraction_method', 'pdfplumber')
        self.dpi = kwargs.get('dpi', 200)
        self.page_range = kwargs.get('page_range', None)  # (start, end) or None for all
        self.password = kwargs.get('password', None)
        self.temp_dir = kwargs.get('temp_dir', tempfile.gettempdir())
        
    def read_file(self, file_path: str) -> Generator[pd.DataFrame, None, None]:
        """
        Read a PDF file and extract structured data.
        
        Args:
            file_path: Path to the PDF file
            
        Yields:
            DataFrame chunks from the PDF file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
            
        # Get file info
        file_info = self.get_file_info(file_path)
        total_pages = file_info.get('page_count', 0)
        
        # Determine page range
        start_page, end_page = 0, total_pages - 1
        if self.page_range:
            start_page = max(0, self.page_range[0])
            end_page = min(total_pages - 1, self.page_range[1] if self.page_range[1] is not None else total_pages - 1)
        
        # Process pages one at a time to minimize memory usage
        with pdfplumber.open(file_path, password=self.password) as pdf:
            # Create progress bar
            page_range = range(start_page, end_page + 1)
            pbar = tqdm(page_range, desc=f"Processing {os.path.basename(file_path)}", 
                       disable=not self.show_progress)
            
            for page_num in pbar:
                try:
                    # Extract content from this page
                    page = pdf.pages[page_num]
                    
                    # Update progress description
                    pbar.set_description(f"Processing page {page_num+1}/{total_pages}")
                    
                    # Process tables if requested
                    if self.extract_tables:
                        tables = page.extract_tables()
                        for i, table in enumerate(tables):
                            # Convert table to DataFrame
                            if table and any(table):
                                # Clean up the table (remove empty cells, etc)
                                clean_table = [[cell if cell else '' for cell in row] for row in table]
                                
                                # Determine if first row is header
                                likely_header = self._is_likely_header(clean_table)
                                
                                if likely_header and len(clean_table) > 1:
                                    # Use first row as header
                                    df = pd.DataFrame(clean_table[1:], columns=clean_table[0])
                                else:
                                    # No header, use default column names
                                    df = pd.DataFrame(clean_table)
                                
                                # Add metadata
                                df['pdf_page'] = page_num + 1
                                df['table_num'] = i + 1
                                
                                # Optimize data types
                                df = self.optimize_dtypes(df)
                                
                                yield df
                    
                    # Process text if requested
                    if self.extract_text:
                        text = page.extract_text(x_tolerance=2.0)
                        if text:
                            # Create a simple DataFrame with the text content
                            text_df = pd.DataFrame({
                                'page_num': [page_num + 1],
                                'text_content': [text]
                            })
                            yield text_df
                            
                    # Explicitly clean up to reduce memory usage
                    del page
                    self.memory_monitor.cleanup_memory()
                    
                except Exception as e:
                    logger.error(f"Error processing PDF page {page_num}: {str(e)}")
                    # Create error dataframe
                    error_df = pd.DataFrame({
                        'page_num': [page_num + 1],
                        'error': [str(e)]
                    })
                    yield error_df
                    
                # Periodically force garbage collection
                if page_num % 5 == 0:
                    self.memory_monitor.cleanup_memory()
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get metadata about a PDF file without reading its full contents.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary with file metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
            
        file_info = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_size_bytes": os.path.getsize(file_path),
            "file_size_mb": os.path.getsize(file_path) / (1024 * 1024),
            "file_type": "pdf",
        }
        
        try:
            # Use PyPDF2 for metadata extraction (uses less memory than pdfplumber for this task)
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                if pdf_reader.is_encrypted and not pdf_reader.decrypt(self.password or ''):
                    file_info["encrypted"] = True
                    file_info["accessible"] = False
                    return file_info
                
                file_info["page_count"] = len(pdf_reader.pages)
                
                # Extract metadata if available
                metadata = pdf_reader.metadata
                if metadata:
                    file_info["metadata"] = {
                        "title": metadata.get('/Title', ''),
                        "author": metadata.get('/Author', ''),
                        "creator": metadata.get('/Creator', ''),
                        "producer": metadata.get('/Producer', ''),
                        "creation_date": metadata.get('/CreationDate', '')
                    }
                
                # Sample first page to detect tables
                if file_info["page_count"] > 0:
                    # Briefly open with pdfplumber to check for tables
                    with pdfplumber.open(file_path, password=self.password) as pdf:
                        if len(pdf.pages) > 0:
                            first_page = pdf.pages[0]
                            tables = first_page.extract_tables()
                            file_info["has_tables"] = len(tables) > 0
                            file_info["table_count_page1"] = len(tables)
                            
                            # Get text sample from first page
                            text_sample = first_page.extract_text(x_tolerance=2.0)
                            if text_sample:
                                file_info["has_text"] = True
                                # Include first few characters as sample
                                file_info["text_sample"] = text_sample[:200] + '...' if len(text_sample) > 200 else text_sample
                            else:
                                file_info["has_text"] = False
            
            # Estimate memory requirements
            file_info["estimated_memory_mb"] = self.file_calculator.estimate_file_memory_usage(file_path, "pdf")
            
            return file_info
            
        except Exception as e:
            logger.error(f"Error getting PDF file info for {file_path}: {str(e)}")
            file_info["error"] = str(e)
            # Return basic file info even if detailed analysis fails
            return file_info
            
    def extract_images(self, file_path: str, output_dir: Optional[str] = None) -> List[str]:
        """
        Extract images from PDF with memory optimization.
        
        Args:
            file_path: Path to the PDF file
            output_dir: Directory to save extracted images (None for temp dir)
            
        Returns:
            List of paths to extracted images
        """
        if output_dir is None:
            output_dir = os.path.join(self.temp_dir, "pdf_images")
            
        os.makedirs(output_dir, exist_ok=True)
        
        image_paths = []
        
        try:
            # Get file info
            file_info = self.get_file_info(file_path)
            total_pages = file_info.get('page_count', 0)
            
            # Determine page range
            start_page, end_page = 0, total_pages - 1
            if self.page_range:
                start_page = max(0, self.page_range[0])
                end_page = min(total_pages - 1, self.page_range[1] if self.page_range[1] is not None else total_pages - 1)
            
            # Use PyPDF2 to extract images
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                
                # Create progress bar
                pbar = tqdm(range(start_page, end_page + 1), 
                           desc=f"Extracting images from {os.path.basename(file_path)}", 
                           disable=not self.show_progress)
                
                for page_num in pbar:
                    try:
                        page = pdf_reader.pages[page_num]
                        
                        # Try to extract images using PyMuPDF if available
                        try:
                            import fitz  # PyMuPDF
                            with fitz.open(file_path) as doc:
                                page_obj = doc.load_page(page_num)
                                image_list = page_obj.get_images(full=True)
                                
                                # Extract each image
                                for img_idx, img in enumerate(image_list):
                                    xref = img[0]
                                    base_image = doc.extract_image(xref)
                                    image_bytes = base_image["image"]
                                    
                                    # Determine image format
                                    image_format = base_image["ext"]
                                    image_path = os.path.join(output_dir, 
                                                            f"page_{page_num+1}_img_{img_idx+1}.{image_format}")
                                    
                                    # Save the image
                                    with open(image_path, "wb") as img_file:
                                        img_file.write(image_bytes)
                                    image_paths.append(image_path)
                                    
                        except ImportError:
                            # Fall back to pdfplumber
                            with pdfplumber.open(file_path, password=self.password) as pdf:
                                page_obj = pdf.pages[page_num]
                                
                                # Extract images
                                for img_idx, img in enumerate(page_obj.images):
                                    try:
                                        # Get image as bytes
                                        img_bytes = img["stream"].get_data()
                                        
                                        # Save image
                                        image_path = os.path.join(output_dir, 
                                                                f"page_{page_num+1}_img_{img_idx+1}.png")
                                        
                                        # Convert bytes to PIL Image and save
                                        image = Image.open(io.BytesIO(img_bytes))
                                        image.save(image_path)
                                        image_paths.append(image_path)
                                    except Exception as e:
                                        logger.warning(f"Error extracting image {img_idx} from page {page_num+1}: {str(e)}")
                        
                        # Clean up memory
                        self.memory_monitor.cleanup_memory()
                    
                    except Exception as e:
                        logger.error(f"Error extracting images from page {page_num+1}: {str(e)}")
            
            return image_paths
            
        except Exception as e:
            logger.error(f"Error extracting images from PDF: {str(e)}")
            return image_paths
            
    def _is_likely_header(self, table: List[List[str]]) -> bool:
        """
        Check if the first row is likely a header.
        
        Args:
            table: Table data as list of lists
            
        Returns:
            True if first row appears to be a header
        """
        if not table or len(table) < 2:
            return False
            
        first_row = table[0]
        rest_rows = table[1:]
        
        # Check if first row has all strings
        first_row_all_strings = all(isinstance(cell, str) for cell in first_row)
        
        # Check if first row has shorter content on average
        first_row_avg_len = sum(len(str(cell)) for cell in first_row) / len(first_row) if len(first_row) > 0 else 0
        
        rest_rows_avg_lens = []
        for row in rest_rows:
            row_avg_len = sum(len(str(cell)) for cell in row) / len(row) if len(row) > 0 else 0
            rest_rows_avg_lens.append(row_avg_len)
        
        overall_avg_len = sum(rest_rows_avg_lens) / len(rest_rows_avg_lens) if rest_rows_avg_lens else 0
        
        # Header rows are usually shorter than data rows
        shorter_content = first_row_avg_len < overall_avg_len
        
        # Check if first row values are present in most columns below
        column_headers = []
        for col_idx in range(min(len(first_row), len(rest_rows[0]) if rest_rows else 0)):
            header_value = first_row[col_idx]
            # Check how many data rows contain the header value
            header_in_column = sum(1 for row in rest_rows if header_value in str(row[col_idx]))
            column_headers.append(header_in_column < len(rest_rows) * 0.1)  # Less than 10% contain header
            
        most_cols_have_headers = sum(column_headers) > len(column_headers) * 0.6  # At least 60% of columns
        
        return (first_row_all_strings and (shorter_content or most_cols_have_headers))