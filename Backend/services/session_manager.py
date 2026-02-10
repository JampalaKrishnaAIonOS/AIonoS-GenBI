from typing import Dict, List, Any
from datetime import datetime, timedelta

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = timedelta(hours=2)
    
    def create_session(self, session_id: str):
        """Create new session"""
        self.sessions[session_id] = {
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'conversation_history': [],
            'last_used_sheet': None,
            'last_df': None,
            'last_code': None,
            'context': {}
        }
    
    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get or create session"""
        if session_id not in self.sessions:
            self.create_session(session_id)
        else:
            # Update last activity
            self.sessions[session_id]['last_activity'] = datetime.now()
        
        return self.sessions[session_id]
    
    def add_to_history(self, session_id: str, role: str, content: str):
        """Add message to conversation history"""
        session = self.get_session(session_id)
        session['conversation_history'].append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 10 messages
        self.trim_history(session_id, max_messages=10)

    def trim_history(self, session_id: str, max_messages: int = 10):
        """Trim conversation history to keep only the last N messages"""
        session = self.get_session(session_id)
        if len(session['conversation_history']) > max_messages:
            session['conversation_history'] = session['conversation_history'][-max_messages:]
    
    def update_context(self, session_id: str, **kwargs):
        """Update session context"""
        session = self.get_session(session_id)
        
        # Robust update: Check for last_result specifically
        if "last_result" in kwargs:
            lr = kwargs["last_result"]
            
            # Use safe type checking instead of .get() on possible DataFrame
            import pandas as pd
            is_df = isinstance(lr, (pd.DataFrame, pd.Series))
            
            # Always update context with the last result regardless of type
            session["context"]["last_result"] = lr
            
            # If it's a data object, mark as last successful for plotting
            if is_df:
                session["last_df"] = lr
                session["last_successful_result"] = lr
                session["stale_error"] = False
            elif isinstance(lr, dict) and lr.get("type") in ["dataframe", "series"]:
                session["last_successful_result"] = lr
                session["stale_error"] = False
        
        if "last_sql" in kwargs:
            session["last_sql"] = kwargs["last_sql"]
        
        session['context'].update(kwargs)

        session['last_activity'] = datetime.now()

    def get_last_result(self, session_id: str):
        """Retrieve the last execution result from session context"""
        session = self.get_session(session_id)
        return session.get("context", {}).get("last_result")
    
    def set_last_dataframe(self, session_id: str, df: Any):
        """Specifically store a dataframe for plotting/reuse"""
        session = self.get_session(session_id)
        session['last_df'] = df
        
    def get_last_dataframe(self, session_id: str):
        """Retrieve the last stored dataframe"""
        session = self.get_session(session_id)
        return session.get('last_df')
    
    def set_last_sql(self, session_id: str, sql: str):
        """Specifically store a SQL query for reuse"""
        session = self.get_session(session_id)
        session['last_sql'] = sql
    
    def get_last_sql(self, session_id: str):
        """Retrieve the last stored SQL query"""
        session = self.get_session(session_id)
        return session.get('last_sql')
    
    def set_last_table(self, session_id: str, table_data: Dict[str, Any]):
        """Specifically store a table object for plotting/reuse"""
        session = self.get_session(session_id)
        session['last_table'] = table_data
        
    def get_last_table(self, session_id: str):
        """Retrieve the last stored table object"""
        session = self.get_session(session_id)
        return session.get('last_table')
    
    def cleanup_old_sessions(self):
        """Remove expired sessions"""
        current_time = datetime.now()
        expired = [
            sid for sid, session in self.sessions.items()
            if current_time - session['last_activity'] > self.session_timeout
        ]
        
        for sid in expired:
            del self.sessions[sid]
        
        if expired:
            print(f"🧹 Cleaned up {len(expired)} expired sessions")

# Global instance for shared access
session_manager = SessionManager()