"""
Image processor with OCR capabilities and memory optimization.
"""
import os
import io
import tempfile
from typing import Dict, Generator, List, Optional, Any, Union, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFile
import pytesseract

from app.utils.logging_config import get_logger
from app.processors.base_processor import BaseProcessor

# Configure PIL to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = get_logger(__name__)

class ImageProcessor(BaseProcessor):
    """
    Process images with OCR and memory optimization.
    
    This class handles image processing efficiently by:
    1. Processing images in batches to minimize memory usage
    2. Optimizing image loading and OCR operations
    3. Converting OCR results to structured data
    """
    
    def __init__(self, **kwargs):
        """Initialize the image processor with shared parameters."""
        super().__init__(**kwargs)
        # Image-specific settings
        self.temp_dir = kwargs.get('temp_dir', tempfile.gettempdir())
        self.dpi = kwargs.get('dpi', 300)
        self.language = kwargs.get('language', 'eng')  # OCR language
        self.ocr_config = kwargs.get('ocr_config', '')  # Tesseract config
        self.preprocess_images = kwargs.get('preprocess_images', True)
        self.batch_size = kwargs.get('batch_size', 10)  # Process images in batches
        self.max_dimension = kwargs.get('max_dimension', 3000)  # Resize large images
        self.min_confidence = kwargs.get('min_confidence', 70)  # Min confidence for OCR
        
        # Set tessdata path if provided
        tessdata_path = kwargs.get('tessdata_path', None)
        if tessdata_path and os.path.exists(tessdata_path):
            pytesseract.pytesseract.tesseract_cmd = tessdata_path
        
    def read_file(self, file_path: str) -> Generator[pd.DataFrame, None, None]:
        """
        Process a single image file with OCR.
        
        Args:
            file_path: Path to the image file
            
        Yields:
            DataFrame with OCR results
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")
            
        # Process a single image
        result = self.process_image(file_path)
        if result:
            yield result
            
    def process_image(self, image_path: str) -> pd.DataFrame:
        """
        Process a single image with OCR and memory management.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            DataFrame with OCR results
        """
        try:
            # Load image with memory optimization
            img = self._load_image_optimized(image_path)
            
            # Preprocess image if requested
            if self.preprocess_images:
                img = self._preprocess_image(img)
            
            # Extract text using OCR
            ocr_result = pytesseract.image_to_data(
                img, 
                lang=self.language,
                config=self.ocr_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Convert to DataFrame
            ocr_df = pd.DataFrame(ocr_result)
            
            # Filter out low confidence results and empty text
            ocr_df = ocr_df[(ocr_df['conf'] > self.min_confidence) & (ocr_df['text'].str.strip() != '')]
            
            # Add source file info
            ocr_df['source_file'] = os.path.basename(image_path)
            
            # Clean up memory
            del img
            self.memory_monitor.cleanup_memory()
            
            # Optimize data types for memory efficiency
            ocr_df = self.optimize_dtypes(ocr_df)
            
            return ocr_df
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            # Return error DataFrame
            error_df = pd.DataFrame({
                'error': [str(e)],
                'source_file': [os.path.basename(image_path)]
            })
            return error_df
            
    def process_images_batch(self, image_paths: List[str]) -> Generator[pd.DataFrame, None, None]:
        """
        Process a batch of images efficiently.
        
        Args:
            image_paths: List of image file paths
            
        Yields:
            DataFrames with OCR results from each batch
        """
        if not image_paths:
            return
            
        # Process images in batches to minimize memory usage
        batches = [image_paths[i:i+self.batch_size] 
                  for i in range(0, len(image_paths), self.batch_size)]
                  
        # Process each batch
        for batch_idx, batch in enumerate(tqdm(batches, desc="Processing image batches", disable=not self.show_progress)):
            try:
                batch_results = []
                
                # Process each image in the batch
                for img_path in tqdm(batch, desc=f"Batch {batch_idx+1}", disable=not self.show_progress):
                    try:
                        result = self.process_image(img_path)
                        if result is not None:
                            batch_results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing image {img_path} in batch {batch_idx+1}: {str(e)}")
                
                # Combine batch results
                if batch_results:
                    combined_df = pd.concat(batch_results, ignore_index=True)
                    combined_df['batch_num'] = batch_idx + 1
                    
                    # Optimize data types
                    optimized_df = self.optimize_dtypes(combined_df)
                    
                    yield optimized_df
                    
                # Force memory cleanup after each batch
                self.memory_monitor.cleanup_memory(aggressive=True)
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx+1}: {str(e)}")
                # Return error DataFrame
                error_df = pd.DataFrame({
                    'error': [str(e)],
                    'batch_num': [batch_idx + 1]
                })
                yield error_df
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get metadata about an image file.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Dictionary with file metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")
            
        file_info = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_size_bytes": os.path.getsize(file_path),
            "file_size_mb": os.path.getsize(file_path) / (1024 * 1024),
            "file_type": "image",
        }
        
        try:
            # Open image to get metadata
            with Image.open(file_path) as img:
                file_info["image_format"] = img.format
                file_info["image_mode"] = img.mode
                file_info["width"] = img.width
                file_info["height"] = img.height
                file_info["aspect_ratio"] = img.width / img.height if img.height > 0 else 0
                
                # Get image metadata (EXIF, etc) if available
                metadata = {}
                if hasattr(img, '_getexif') and img._getexif():
                    exif = img._getexif()
                    if exif:
                        # Convert EXIF tags to readable format
                        from PIL.ExifTags import TAGS
                        for tag_id, value in exif.items():
                            tag = TAGS.get(tag_id, tag_id)
                            # Only include string-serializable values
                            try:
                                # Test if value can be serialized to string
                                str(value)
                                metadata[tag] = value
                            except:
                                metadata[tag] = "Complex value (not serialized)"
                                
                file_info["metadata"] = metadata
                
                # Check if image is likely to contain text (heuristic)
                # This is a simple heuristic: color depth and image size
                has_sufficient_detail = (img.mode in ['L', 'RGB', 'RGBA']) and (img.width > 300 and img.height > 300)
                file_info["likely_has_text"] = has_sufficient_detail
                
            # Estimate memory requirements
            file_info["estimated_memory_mb"] = self.file_calculator.estimate_file_memory_usage(file_path, "image")
            
            return file_info
            
        except Exception as e:
            logger.error(f"Error getting image file info for {file_path}: {str(e)}")
            file_info["error"] = str(e)
            # Return basic file info even if detailed analysis fails
            return file_info
    
    def _load_image_optimized(self, image_path: str) -> Image.Image:
        """
        Load an image with memory optimization.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL Image object
        """
        # Get image info to determine if resizing is needed
        with Image.open(image_path) as img_info:
            width, height = img_info.size
            format = img_info.format
            
        # Determine if the image should be resized for memory efficiency
        if width > self.max_dimension or height > self.max_dimension:
            # Calculate new dimensions while preserving aspect ratio
            if width > height:
                new_width = self.max_dimension
                new_height = int(height * (self.max_dimension / width))
            else:
                new_height = self.max_dimension
                new_width = int(width * (self.max_dimension / height))
            
            # Open and resize the image
            img = Image.open(image_path)
            img = img.resize((new_width, new_height), Image.LANCZOS)
            logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
            return img
        else:
            # Image is already small enough, load normally
            return Image.open(image_path)
            
    def _preprocess_image(self, img: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR results.
        
        Args:
            img: PIL Image object
            
        Returns:
            Preprocessed PIL Image
        """
        # Convert to grayscale for better OCR
        if img.mode != 'L':
            img = img.convert('L')
        
        # Apply additional preprocessing depending on the image
        # For example: thresholding, noise reduction, contrast enhancement
        
        # Import necessary libraries only when needed
        try:
            import cv2
            import numpy as np
            
            # Convert PIL image to OpenCV format for advanced processing
            img_np = np.array(img)
            
            # Apply adaptive thresholding
            img_np = cv2.adaptiveThreshold(
                img_np, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                11, 
                2
            )
            
            # Denoise the image
            img_np = cv2.fastNlMeansDenoising(img_np, None, 10, 7, 21)
            
            # Convert back to PIL
            img = Image.fromarray(img_np)
            
        except ImportError:
            logger.warning("OpenCV not available for advanced image preprocessing")
        
        return img
        
    def extract_text_blocks(self, ocr_df: pd.DataFrame) -> pd.DataFrame:
        """
        Group OCR results into logical text blocks.
        
        Args:
            ocr_df: DataFrame with OCR results
            
        Returns:
            DataFrame with text blocks
        """
        if ocr_df.empty:
            return pd.DataFrame(columns=['block_num', 'text', 'confidence', 'x', 'y', 'width', 'height'])
            
        # Group by block number
        blocks = []
        for block_num, group in ocr_df.groupby('block_num'):
            # Skip empty blocks
            if group.empty:
                continue
                
            # Combine text from the same block
            block_text = ' '.join(group['text'].dropna())
            
            # Calculate average confidence
            avg_confidence = group['conf'].mean()
            
            # Calculate bounding box
            x = group['left'].min()
            y = group['top'].min()
            width = group['width'].sum()
            height = group['height'].max()
            
            blocks.append({
                'block_num': block_num,
                'text': block_text,
                'confidence': avg_confidence,
                'x': x,
                'y': y, 
                'width': width,
                'height': height,
                'source_file': group['source_file'].iloc[0] if 'source_file' in group.columns else None
            })
        
        # Create DataFrame of text blocks
        blocks_df = pd.DataFrame(blocks)
        return blocks_df