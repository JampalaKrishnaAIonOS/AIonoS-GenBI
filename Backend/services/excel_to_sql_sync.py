"""
Excel to SQL Database Sync Service

Handles:
- Loading Excel files from folder
- Creating SQL tables with proper schemas
- Inserting/updating data
- Handling multiple sheets per file
"""

import os
import pandas as pd
from sqlalchemy import create_engine, inspect, MetaData, Table, Column, String, Float, Integer, DateTime, Text, Boolean
from sqlalchemy.sql import text
from pathlib import Path
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class ExcelToSQLSync:
    def __init__(self, excel_folder: str, db_url: str):
        """
        Initialize the Excel to SQL sync service
        
        Args:
            excel_folder: Path to folder containing Excel files
            db_url: SQLAlchemy database URL (e.g., 'sqlite:///genbi.db')
        """
        self.excel_folder = Path(excel_folder)
        self.engine = create_engine(db_url, echo=False)
        self.metadata = MetaData()
        
    def get_excel_files(self) -> List[Path]:
        """Get all non-temporary Excel files"""
        if not self.excel_folder.exists():
            logger.warning(f"Excel folder not found: {self.excel_folder}")
            return []
            
        return [
            f for f in self.excel_folder.glob("*.xlsx")
            if not f.name.startswith('~$')
        ]
    
    def sanitize_table_name(self, filename: str, sheetname: str) -> str:
        """
        Create valid SQL table name from file and sheet names
        
        Example:
            CoalEnergCost_PlantMasterData.xlsx::Sheet1 
            → coal_energ_cost_plant_master_data__sheet1
        """
        import re
        # Remove .xlsx extension
        base = filename.replace('.xlsx', '')
        # Combine file and sheet
        combined = f"{base}__{sheetname}"
        # Convert to lowercase and replace special chars with underscore
        sanitized = re.sub(r'[^a-z0-9]', '_', combined.lower())
        # Remove consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        # Ensure it starts with a letter (SQL requirement)
        if sanitized and sanitized[0].isdigit():
            sanitized = 't_' + sanitized
        return sanitized
    
    def infer_sql_type(self, pandas_dtype):
        """Map pandas dtype to SQLAlchemy type"""
        dtype_str = str(pandas_dtype)
        
        if 'int' in dtype_str:
            return Integer
        elif 'float' in dtype_str:
            return Float
        elif 'datetime' in dtype_str:
            return DateTime
        elif 'bool' in dtype_str:
            return Boolean
        elif 'object' in dtype_str:
            return Text
        else:
            return Text  # Default fallback
    
    def create_table_from_dataframe(self, df: pd.DataFrame, table_name: str) -> bool:
        """
        Create SQL table from pandas DataFrame schema
        
        Returns:
            bool: True if table created successfully
        """
        try:
            # Drop table if exists
            with self.engine.begin() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
            
            # Infer types and create columns
            columns = []
            for col_name in df.columns:
                col_type = self.infer_sql_type(df[col_name].dtype)
                # Sanitize column name
                safe_col = col_name.replace(' ', '_').replace('-', '_')
                columns.append(Column(safe_col, col_type))
            
            # Create table
            table = Table(table_name, self.metadata, *columns, extend_existing=True)
            table.create(self.engine, checkfirst=True)
            
            logger.info(f"✅ Created table: {table_name} ({len(columns)} columns)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create table {table_name}: {e}")
            return False
    
    def load_excel_to_table(self, excel_path: Path, sheet_name: str, table_name: str) -> bool:
        """
        Load Excel sheet data into SQL table
        
        Returns:
            bool: True if data loaded successfully
        """
        try:
            # Read Excel sheet
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            
            # Clean column names (remove spaces, special chars)
            df.columns = [str(col).replace(' ', '_').replace('-', '_') for col in df.columns]
            
            # Handle NaN values
            # Using object/Text for 'None' safety if types are mixed used to be tricky, 
            # but fillna('') generally works for text columns. 
            # For numeric, pandas handles to_sql normally with NaN as NULL usually. 
            # The prompt suggested: df = df.fillna('') # Or use None for NULL values
            # Let's stick to what was requested but be careful with mixed types.
            # Actually, filling everything with '' turns numeric columns into objects if not careful.
            # But the Prompt explicitly asked for this code:
            # df = df.fillna('')
            # I will follow the prompt's provided code block exactly where possible.
            df = df.fillna('') 
            
            # Create table
            if not self.create_table_from_dataframe(df, table_name):
                return False
            
            # Insert data
            df.to_sql(table_name, self.engine, if_exists='replace', index=False)
            
            logger.info(f"✅ Loaded {len(df)} rows into {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load {excel_path}::{sheet_name}: {e}")
            return False
    
    def sync_all_excel_files(self) -> Dict[str, List[str]]:
        """
        Sync all Excel files to database
        
        Returns:
            Dict with 'success' and 'failed' table lists
        """
        results = {'success': [], 'failed': []}
        
        excel_files = self.get_excel_files()
        logger.info(f"📊 Found {len(excel_files)} Excel files to sync")
        
        # Ensure metadata is cleared to avoid clutter from previous runs if needed
        self.metadata.clear()

        for excel_file in excel_files:
            try:
                # Get all sheets
                # The user provided code uses pd.ExcelFile
                xls = pd.ExcelFile(excel_file)
                
                for sheet_name in xls.sheet_names:
                    table_name = self.sanitize_table_name(excel_file.name, sheet_name)
                    
                    if self.load_excel_to_table(excel_file, sheet_name, table_name):
                        results['success'].append(table_name)
                    else:
                        results['failed'].append(table_name)
                        
            except Exception as e:
                logger.error(f"❌ Failed to process {excel_file}: {e}")
                results['failed'].append(str(excel_file))
        
        logger.info(f"✅ Sync complete: {len(results['success'])} success, {len(results['failed'])} failed")
        return results
    
    def remove_orphaned_tables(self):
        """
        Remove SQL tables that no longer have corresponding Excel files
        """
        # Get current Excel-based table names
        current_tables = set()
        for excel_file in self.get_excel_files():
            try:
                xls = pd.ExcelFile(excel_file)
                for sheet_name in xls.sheet_names:
                    table_name = self.sanitize_table_name(excel_file.name, sheet_name)
                    current_tables.add(table_name)
            except:
                pass
        
        # Get all tables in database
        inspector = inspect(self.engine)
        db_tables = set(inspector.get_table_names())
        
        # Find orphaned tables
        orphaned = db_tables - current_tables
        
        # Drop orphaned tables
        with self.engine.begin() as conn:
            for table_name in orphaned:
                logger.info(f"🗑️ Removing orphaned table: {table_name}")
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
        
        return list(orphaned)
    
    def get_table_info(self, table_name: str) -> Dict:
        """Get schema information for a table"""
        inspector = inspect(self.engine)
        
        if table_name not in inspector.get_table_names():
            return None
        
        columns = inspector.get_columns(table_name)
        
        return {
            'table_name': table_name,
            'columns': [
                {
                    'name': col['name'],
                    'type': str(col['type']),
                    'nullable': col['nullable']
                }
                for col in columns
            ]
        }
    
    def list_all_tables(self) -> List[str]:
        """List all tables in database"""
        inspector = inspect(self.engine)
        return inspector.get_table_names()
