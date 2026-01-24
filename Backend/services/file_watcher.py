import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from services.excel_indexer import ExcelSchemaIndexer

class ExcelFileHandler(FileSystemEventHandler):
    def __init__(self, indexer: ExcelSchemaIndexer):
        self.indexer = indexer
        self.last_modified = time.time()
        
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.xlsx') and not os.path.basename(event.src_path).startswith('~$'):
            # Debounce: wait 2 seconds before reindexing
            current_time = time.time()
            if current_time - self.last_modified > 2:
                print(f"📝 File modified: {event.src_path}")
                print("🔄 Re-indexing...")
                self.indexer.index_all_sheets()
                self.last_modified = current_time
    
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.xlsx') and not os.path.basename(event.src_path).startswith('~$'):
            print(f"📄 New file detected: {event.src_path}")
            print("🔄 Re-indexing...")
            self.indexer.index_all_sheets()
            self.last_modified = time.time()

    def on_deleted(self, event):
        if not event.is_directory and event.src_path.endswith('.xlsx') and not os.path.basename(event.src_path).startswith('~$'):
            print(f"🗑️ File deleted: {event.src_path}")
            print("🔄 Re-indexing...")
            self.indexer.index_all_sheets()
            self.last_modified = time.time()

def start_file_watcher(excel_folder: str, indexer: ExcelSchemaIndexer):
    """Start watching Excel folder for changes"""
    event_handler = ExcelFileHandler(indexer)
    observer = Observer()
    observer.schedule(event_handler, excel_folder, recursive=False)
    observer.start()
    print(f"👀 Watching folder: {excel_folder}")
    return observer