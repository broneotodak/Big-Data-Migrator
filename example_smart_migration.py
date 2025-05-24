#!/usr/bin/env python3
"""
Example: Smart Data Migration with LLM-Assisted Schema Unification

This example demonstrates how to use the SmartSchemaUnifier to:
1. Analyze multiple file formats containing similar data
2. Use LLM to infer unified schema and relationships  
3. Perform programmatic data migration to Supabase
4. Handle large datasets efficiently without LLM bottlenecks
"""
import os
import asyncio
from typing import List
from pathlib import Path

from app.processors.smart_schema_unifier import SmartSchemaUnifier, UnifiedSchema
from app.llm.conversation_system import LLMConversationSystem
from app.memory.memory_monitor import MemoryMonitor

class SmartDataMigrator:
    """Complete smart data migration workflow."""
    
    def __init__(self):
        """Initialize the smart data migrator."""
        self.memory_monitor = MemoryMonitor()
        
        # Initialize LLM system for schema inference
        self.llm_system = LLMConversationSystem(
            memory_monitor=self.memory_monitor,
            enable_anthropic=False,  # Set to True if you have Anthropic API key
            enable_online_fallback=False  # Set to True if you have OpenAI API key
        )
        
        # Initialize schema unifier
        self.schema_unifier = SmartSchemaUnifier(
            llm_system=self.llm_system,
            memory_monitor=self.memory_monitor,
            confidence_threshold=0.8,
            max_sample_rows=100
        )
        
    async def migrate_files_to_supabase(self, 
                                      file_paths: List[str],
                                      table_name: str = "unified_data") -> dict:
        """
        Complete migration workflow from multiple files to Supabase.
        
        Args:
            file_paths: List of file paths to migrate
            table_name: Target table name in Supabase
            
        Returns:
            Migration results
        """
        print(f"🚀 Starting smart migration for {len(file_paths)} files")
        
        # Step 1: Analyze files and create unified schema
        print("\n📊 Step 1: Analyzing files and inferring unified schema...")
        analysis_result = self.schema_unifier.analyze_files_for_unification(file_paths)
        
        if not analysis_result["unified_schema"]:
            print("❌ Failed to create unified schema")
            return {"error": "Schema unification failed"}
        
        unified_schema = analysis_result["unified_schema"]
        unified_schema.table_name = table_name
        
        # Display schema information
        self._display_schema_analysis(analysis_result)
        
        # Step 2: Validate and confirm schema
        print("\n🔍 Step 2: Validating schema mappings...")
        validation = analysis_result["validation_results"]
        
        if validation["data_type_conflicts"]:
            print(f"⚠️  Found {len(validation['data_type_conflicts'])} data type conflicts")
            for conflict in validation["data_type_conflicts"]:
                print(f"   - Column '{conflict['target_column']}': {conflict['conflicting_types']}")
        
        if validation["low_confidence_mappings"] > 0:
            print(f"⚠️  {validation['low_confidence_mappings']} low-confidence mappings need review")
        
        # Step 3: Execute migration
        print(f"\n📤 Step 3: Migrating data to Supabase table '{table_name}'...")
        migration_result = self.schema_unifier.create_supabase_migration(
            unified_schema=unified_schema,
            file_paths=file_paths,
            batch_size=1000
        )
        
        # Display migration results
        self._display_migration_results(migration_result)
        
        return {
            "schema_analysis": analysis_result,
            "migration_result": migration_result,
            "success": migration_result.get("total_rows_migrated", 0) > 0
        }
    
    def _display_schema_analysis(self, analysis_result: dict):
        """Display schema analysis results."""
        unified_schema = analysis_result["unified_schema"]
        
        print(f"\n✅ Unified Schema Created: '{unified_schema.table_name}'")
        print("-" * 50)
        print(f"📋 Columns ({len(unified_schema.columns)}):")
        for col_name, data_type in unified_schema.columns.items():
            print(f"   - {col_name}: {data_type}")
        
        print(f"\n🔑 Primary Keys: {unified_schema.primary_keys}")
        
        if unified_schema.foreign_keys:
            print(f"🔗 Foreign Keys: {unified_schema.foreign_keys}")
        
        print(f"\n📁 File Mappings:")
        for file_path, mappings in unified_schema.file_mappings.items():
            file_name = os.path.basename(file_path)
            print(f"   {file_name}: {len(mappings)} column mappings")
        
        # Show recommendations
        recommendations = analysis_result["recommendations"]
        if recommendations:
            print(f"\n💡 Recommendations:")
            for rec in recommendations:
                print(f"   - {rec}")
    
    def _display_migration_results(self, migration_result: dict):
        """Display migration results."""
        print(f"\n📈 Migration Results:")
        print("-" * 40)
        print(f"✅ Table Created: {migration_result['table_created']}")
        print(f"📊 Total Rows Migrated: {migration_result['total_rows_migrated']:,}")
        print(f"📁 Files Processed: {migration_result['files_processed']}")
        print(f"⏱️  Processing Time: {migration_result['processing_time']:.2f}s")
        
        if migration_result["failed_files"]:
            print(f"❌ Failed Files: {len(migration_result['failed_files'])}")
            for failed in migration_result["failed_files"]:
                print(f"   - {os.path.basename(failed['file_path'])}: {failed['error']}")
        
        if migration_result["data_quality_issues"]:
            print(f"⚠️  Data Quality Issues: {len(migration_result['data_quality_issues'])}")
            # Group by issue type
            issue_summary = {}
            for issue in migration_result["data_quality_issues"]:
                issue_type = issue["type"]
                if issue_type not in issue_summary:
                    issue_summary[issue_type] = 0
                issue_summary[issue_type] += 1
            
            for issue_type, count in issue_summary.items():
                print(f"   - {issue_type}: {count} occurrences")
        
        # Show SQL schema
        if migration_result.get("schema_sql"):
            print(f"\n🗃️  Generated SQL Schema:")
            print(migration_result["schema_sql"])

def create_sample_data_files():
    """Create sample data files for testing."""
    print("📝 Creating sample data files for testing...")
    
    # Create sample directories
    os.makedirs("sample_data", exist_ok=True)
    
    # Sample customers CSV
    customers_csv = """id,name,email,phone,registration_date,country
1,John Doe,john.doe@email.com,+1-555-0101,2024-01-15,USA
2,Jane Smith,jane.smith@email.com,+1-555-0102,2024-01-16,USA
3,Bob Johnson,bob.johnson@email.com,+44-20-7946-0958,2024-01-17,UK
4,Alice Brown,alice.brown@email.com,+33-1-42-86-83-26,2024-01-18,France"""
    
    with open("sample_data/customers.csv", "w") as f:
        f.write(customers_csv)
    
    # Sample orders Excel data (saved as CSV for simplicity)
    orders_csv = """order_id,customer_id,product_name,quantity,price,order_date
1001,1,Laptop,1,999.99,2024-01-20
1002,2,Mouse,2,25.50,2024-01-21
1003,1,Keyboard,1,75.00,2024-01-22
1004,3,Monitor,1,299.99,2024-01-23"""
    
    with open("sample_data/orders.csv", "w") as f:
        f.write(orders_csv)
    
    # Sample products JSON (saved as CSV for simplicity)
    products_csv = """product_id,name,category,price,stock_quantity,supplier_id
1,Laptop,Electronics,999.99,50,101
2,Mouse,Electronics,25.50,200,102
3,Keyboard,Electronics,75.00,150,102
4,Monitor,Electronics,299.99,75,103"""
    
    with open("sample_data/products.csv", "w") as f:
        f.write(products_csv)
    
    return [
        "sample_data/customers.csv",
        "sample_data/orders.csv", 
        "sample_data/products.csv"
    ]

async def main():
    """Main example function."""
    print("🎯 Smart Data Migration Example")
    print("=" * 50)
    
    # Create sample data files
    sample_files = create_sample_data_files()
    print(f"✅ Created {len(sample_files)} sample data files")
    
    # Initialize migrator
    migrator = SmartDataMigrator()
    
    try:
        # Run migration
        result = await migrator.migrate_files_to_supabase(
            file_paths=sample_files,
            table_name="ecommerce_data"
        )
        
        if result["success"]:
            print("\n🎉 Migration completed successfully!")
        else:
            print("\n❌ Migration failed")
            
    except Exception as e:
        print(f"\n❌ Error during migration: {str(e)}")
    
    # Cleanup sample files
    import shutil
    try:
        shutil.rmtree("sample_data")
        print("\n🧹 Cleaned up sample data files")
    except:
        pass

def run_manual_test():
    """Run a manual test with your own files."""
    print("\n🔧 Manual Test Mode")
    print("-" * 30)
    
    # Get file paths from user
    file_paths = []
    print("Enter file paths (one per line, empty line to finish):")
    
    while True:
        file_path = input("File path: ").strip()
        if not file_path:
            break
        if os.path.exists(file_path):
            file_paths.append(file_path)
            print(f"✅ Added: {os.path.basename(file_path)}")
        else:
            print(f"❌ File not found: {file_path}")
    
    if not file_paths:
        print("No valid files provided")
        return
    
    # Get table name
    table_name = input("Enter target table name (default: unified_data): ").strip()
    if not table_name:
        table_name = "unified_data"
    
    # Run migration
    print(f"\n🚀 Starting migration with {len(file_paths)} files...")
    
    async def run_migration():
        migrator = SmartDataMigrator()
        return await migrator.migrate_files_to_supabase(file_paths, table_name)
    
    result = asyncio.run(run_migration())
    
    if result["success"]:
        print("\n🎉 Migration completed successfully!")
    else:
        print("\n❌ Migration failed")

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Run example with sample data")
    print("2. Run manual test with your own files")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(main())
    elif choice == "2":
        run_manual_test()
    else:
        print("Invalid choice") 