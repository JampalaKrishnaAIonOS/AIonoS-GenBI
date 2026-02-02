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
            'last_dataframe': None,
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
        if len(session['conversation_history']) > 10:
            session['conversation_history'] = session['conversation_history'][-10:]
    
    def update_context(self, session_id: str, **kwargs):
        """Update session context"""
        session = self.get_session(session_id)
        
        # Robust update: Check for last_result specifically
        if "last_result" in kwargs:
            lr = kwargs["last_result"]
            
            # Always update context with the last result regardless of type
            session["context"]["last_result"] = lr
            
            # Allow chart to be stored as a successful result to enable "plot it" to work on pure charts if needed,
            # though generally "plot it" works on data. 
            # User specific request: permit 'chart' type.
            if lr.get("type") in ["dataframe", "series", "chart"]:
                session["last_successful_result"] = lr
                session["stale_error"] = False  # Clear false errors
        
        session['context'].update(kwargs)

        session['last_activity'] = datetime.now()

    def get_last_result(self, session_id: str):
        """Retrieve the last execution result from session context"""
        session = self.get_session(session_id)
        return session.get("context", {}).get("last_result")
    
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