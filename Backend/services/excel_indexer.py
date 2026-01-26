import os
import pandas as pd
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict, Tuple
import json

class ExcelSchemaIndexer:
    def __init__(self, excel_folder: str, index_path: str):
        self.excel_folder = excel_folder
        self.index_path = index_path
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.metadata = []
        self.index = None
        
    def extract_sheet_schema(self, file_path: str, sheet_name: str) -> Dict:
        """Extract schema information from a sheet"""
        try:
            # Read first 100 rows for schema analysis
            df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=100)
            
            # Get column information
            columns_info = []
            for col in df.columns:
                col_info = {
                    'name': str(col),
                    'dtype': str(df[col].dtype),
                    'sample_values': df[col].dropna().head(3).tolist(),
                    'null_count': int(df[col].isnull().sum())
                }
                columns_info.append(col_info)
            
            # Create rich text description for embedding
            schema_text = f"File: {os.path.basename(file_path)}, Sheet: {sheet_name}\n"
            schema_text += f"Columns: {', '.join([c['name'] for c in columns_info])}\n"
            schema_text += "Column Details:\n"
            for col in columns_info:
                schema_text += f"- {col['name']} ({col['dtype']}): Examples: {col['sample_values']}\n"
            
            return {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'sheet_name': sheet_name,
                'columns': columns_info,
                'row_count': len(df),
                'schema_text': schema_text
            }
        except Exception as e:
            print(f"Error extracting schema from {file_path}[{sheet_name}]: {e}")
            return None
    
    def index_all_sheets(self):
        """Index all sheets from all Excel files"""
        print("🧹 Clearing old index files and state...")
        self.metadata = []
        self.index = None
        
        # Forcefully delete old files from disk
        if os.path.exists(self.index_path):
            for f in ['sheets.index', 'metadata.pkl']:
                f_path = os.path.join(self.index_path, f)
                if os.path.exists(f_path):
                    try:
                        os.remove(f_path)
                    except Exception as e:
                        print(f"⚠️ Could not delete {f}: {e}")

        print("🔍 Starting fresh Excel indexing...")
        excel_files = list(Path(self.excel_folder).glob("*.xlsx"))
        print(f"Found {len(excel_files)} Excel files")
        
        all_schemas = []
        
        for excel_file in excel_files:
            if excel_file.name.startswith('~$'):  # Skip temp files
                continue
                
            try:
                # Get all sheet names
                xl_file = pd.ExcelFile(excel_file)
                sheet_names = xl_file.sheet_names
                print(f"  📄 {excel_file.name}: {len(sheet_names)} sheets")
                
                for sheet_name in sheet_names:
                    schema = self.extract_sheet_schema(str(excel_file), sheet_name)
                    if schema:
                        all_schemas.append(schema)
                        print(f"    ✓ Indexed: {sheet_name}")
                        
            except Exception as e:
                print(f"  ✗ Error reading {excel_file.name}: {e}")
        
        if not all_schemas:
            print("⚠️  No sheets found to index! Clearing index...")
            self.metadata = []
            self.index = None
            # Handle index folder cleaning
            if os.path.exists(self.index_path):
                for f in ['sheets.index', 'metadata.pkl']:
                    f_path = os.path.join(self.index_path, f)
                    if os.path.exists(f_path):
                        os.remove(f_path)
            return
        
        # Create embeddings
        print(f"\n🧠 Creating embeddings for {len(all_schemas)} sheets...")
        schema_texts = [s['schema_text'] for s in all_schemas]
        embeddings = self.embedding_model.encode(schema_texts, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        self.metadata = all_schemas
        
        # Save to disk
        self.save_index()
        print(f"✅ Indexing complete! Indexed {len(all_schemas)} sheets")
    
    def save_index(self):
        """Save FAISS index and metadata"""
        os.makedirs(self.index_path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(self.index_path, 'sheets.index'))
        
        # Save metadata
        with open(os.path.join(self.index_path, 'metadata.pkl'), 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f"💾 Index saved to {self.index_path}")
    
    def load_index(self):
        """Load existing FAISS index"""
        try:
            self.index = faiss.read_index(os.path.join(self.index_path, 'sheets.index'))
            with open(os.path.join(self.index_path, 'metadata.pkl'), 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"✅ Loaded index with {len(self.metadata)} sheets")
            return True
        except:
            print("⚠️  No existing index found")
            return False
    
    def search_relevant_sheets(self, query: str, top_k: int = 3) -> List[Dict]:
        """Find most relevant sheets for a query with keyword boosting"""
        if self.index is None:
            return []
        
        # 1. Broaden search to get candidates
        # We fetch more results (e.g. 10) to allow re-ranking to pull up relevant files
        broad_k = min(len(self.metadata), 10)
        query_embedding = self.embedding_model.encode([query])
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), broad_k)
        
        candidates = []
        unique_indices = set()
        
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.metadata) and idx not in unique_indices:
                unique_indices.add(idx)
                result = self.metadata[idx].copy()
                # Dynamically update file_path based on current environment
                result['file_path'] = os.path.join(self.excel_folder, result['file_name'])
                result['relevance_score'] = float(dist) # Lower is better in L2
                candidates.append(result)
        
        # 2. Re-rank based on keyword matching
        # If the user explicitly names a file/plant (e.g. "Dadri"), strictly prioritize it.
        query_tokens = set(query.lower().split())
        
        for cand in candidates:
            filename = cand['file_name'].lower()
            sheetname = cand['sheet_name'].lower()
            
            # Calculate keyword matches
            matches = 0
            for token in query_tokens:
                # Ignore short words to prevent false positives (the, for, and...)
                if len(token) < 3: 
                    continue
                    
                # Exact substring match
                if token in filename or token in sheetname:
                    matches += 1
            
            # Apply massive boost for matches
            # L2 distances are usually < 2.0. Subtracting 10.0 puts matches firmly at the top.
            if matches > 0:
                cand['relevance_score'] -= (matches * 10.0)
                
        # 3. Sort by adjusted score (ascending)
        candidates.sort(key=lambda x: x['relevance_score'])
        
        # Return top_k
        return candidates[:top_k]