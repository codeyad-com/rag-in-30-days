"""
Database Manager - Only stores user data and admin credentials
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)



class JSONDatabase:
    """Simple JSON file database for user data only."""
    
    def __init__(self, db_path: Path = Path("app_data.json")):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database file if not exists."""
        if not self.db_path.exists():
            self._save(self._get_default_data())
    
    def _get_default_data(self) -> Dict[str, Any]:
        """Return default database structure."""
        return {
            "admin": {
                "username": "admin",
                "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
                "created_at": datetime.now().isoformat(),
                "last_login": None,
            },
            "stats": {
                "total_queries": 0,
                "total_time_ms": 0,
                "avg_response_time_ms": 0,
                "successful_queries": 0,
                "failed_queries": 0,
            },
            "documents": [],
            "chat_history": [],
        }
    
    def _load(self) -> Dict[str, Any]:
        """Load data from JSON file."""
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return self._get_default_data()
    
    def _save(self, data: Dict[str, Any]):
        """Save data to JSON file."""
        try:
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save database: {e}")
    
    # ==================== Admin Methods ====================
    
    def verify_admin(self, username: str, password: str) -> bool:
        """Verify admin credentials."""
        data = self._load()
        admin = data.get("admin", {})
        
        if username != admin.get("username"):
            return False
        
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        return password_hash == admin.get("password_hash", "")
    
    def update_admin_last_login(self, username: str):
        """Update last login timestamp."""
        data = self._load()
        if data["admin"].get("username") == username:
            data["admin"]["last_login"] = datetime.now().isoformat()
            self._save(data)
    
    def update_admin_password(self, username: str, old_password: str, new_password: str) -> bool:
        """Update admin password."""
        if not self.verify_admin(username, old_password):
            return False
        
        data = self._load()
        data["admin"]["password_hash"] = hashlib.sha256(new_password.encode()).hexdigest()
        self._save(data)
        return True
    
    # ==================== Stats Methods ====================
    
    def increment_query(self, response_time_ms: float, success: bool = True):
        """Update query statistics."""
        data = self._load()
        
        data["stats"]["total_queries"] += 1
        data["stats"]["total_time_ms"] += response_time_ms
        data["stats"]["avg_response_time_ms"] = (
            data["stats"]["total_time_ms"] / data["stats"]["total_queries"]
        )
        
        if success:
            data["stats"]["successful_queries"] += 1
        else:
            data["stats"]["failed_queries"] += 1
        
        self._save(data)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get all statistics."""
        data = self._load()
        return data.get("stats", {})
    
    # ==================== Document Methods ====================
    
    def add_document(self, doc_name: str, chunks: int, size_mb: float = 0) -> bool:
        """Add a document to tracking."""
        data = self._load()
        
        # Check if already exists
        for doc in data["documents"]:
            if doc["name"] == doc_name:
                doc["chunks"] = chunks
                doc["size_mb"] = size_mb
                doc["updated_at"] = datetime.now().isoformat()
                self._save(data)
                return True
        
        # Add new document
        data["documents"].append({
            "name": doc_name,
            "chunks": chunks,
            "size_mb": size_mb,
            "indexed_at": datetime.now().isoformat(),
        })
        self._save(data)
        return True
    
    def remove_document(self, doc_name: str) -> bool:
        """Remove a document from tracking."""
        data = self._load()
        data["documents"] = [doc for doc in data["documents"] if doc["name"] != doc_name]
        self._save(data)
        return True
    
    def get_documents(self) -> List[Dict[str, Any]]:
        """Get all indexed documents."""
        data = self._load()
        return data.get("documents", [])
    
    def clear_all_documents(self) -> bool:
        """Clear all document tracking."""
        data = self._load()
        data["documents"] = []
        self._save(data)
        return True
    
    # ==================== Chat History Methods ====================
    
    def add_chat_message(self, role: str, content: str, sources: List[Dict] = None):
        """Add a message to chat history."""
        data = self._load()
        data["chat_history"].append({
            "role": role,
            "content": content,
            "sources": sources or [],
            "timestamp": datetime.now().isoformat(),
        })
        
        # Keep only last 100 messages
        if len(data["chat_history"]) > 100:
            data["chat_history"] = data["chat_history"][-100:]
        
        self._save(data)
    
    def get_chat_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get chat history."""
        data = self._load()
        return data.get("chat_history", [])[-limit:]
    
    def clear_chat_history(self) -> bool:
        """Clear all chat history."""
        data = self._load()
        data["chat_history"] = []
        self._save(data)
        return True


# Create global database instance
db = JSONDatabase()