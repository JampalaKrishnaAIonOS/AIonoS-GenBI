"""
File Watcher for Excel → SQL Sync

Monitors Excel folder for:
- New files → Create tables
- Modified files → Update tables
- Deleted files → Drop tables
"""

import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from services.excel_to_sql_sync import ExcelToSQLSync
import logging

logger = logging.getLogger(__name__)

class ExcelSQLFileHandler(FileSystemEventHandler):
    def __init__(self, sync_service: ExcelToSQLSync):
        self.sync_service = sync_service
        self.last_modified = time.time()
        
    def on_modified(self, event):
        """Handle file modification"""
        if self._should_process(event):
            logger.info(f"📝 File modified: {event.src_path}")
            self._debounced_sync()
    
    def on_created(self, event):
        """Handle new file creation"""
        if self._should_process(event):
            logger.info(f"➕ New file detected: {event.src_path}")
            self._debounced_sync()
    
    def on_deleted(self, event):
        """Handle file deletion"""
        if event.src_path.endswith('.xlsx'):
            logger.info(f"🗑️ File deleted: {event.src_path}")
            # Remove orphaned tables
            self.sync_service.remove_orphaned_tables()
    
    def _should_process(self, event):
        """Check if event should trigger sync"""
        return (
            not event.is_directory 
            and event.src_path.endswith('.xlsx')
            and not os.path.basename(event.src_path).startswith('~$')
        )
    
    def _debounced_sync(self):
        """Sync with 2-second debounce to avoid rapid re-syncs"""
        current_time = time.time()
        if current_time - self.last_modified > 2:
            logger.info("🔄 Syncing Excel files to database...")
            self.sync_service.sync_all_excel_files()
            self.last_modified = current_time


def start_file_watcher_sql(excel_folder: str, sync_service: ExcelToSQLSync):
    """Start watching Excel folder for changes"""
    event_handler = ExcelSQLFileHandler(sync_service)
    observer = Observer()
    observer.schedule(event_handler, excel_folder, recursive=False)
    observer.start()
    logger.info(f"👀 Watching folder: {excel_folder}")
    return observer
