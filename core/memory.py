"""Memory management for agents and supervisor."""

import json
import sqlite3
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

from .config import Config

# Optional Redis import
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class MemoryBackend:
    """Base class for memory backends."""
    
    def store_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Store a message in memory."""
        raise NotImplementedError
    
    def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve messages for a session."""
        raise NotImplementedError
    
    def clear_session(self, session_id: str) -> None:
        """Clear all messages for a session."""
        raise NotImplementedError


class InMemorySQLiteBackend(MemoryBackend):
    """SQLite in-memory backend."""
    
    def __init__(self, config: Config):
        self.connection = sqlite3.connect(':memory:', check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize the database schema."""
        with self.lock:
            self.connection.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.connection.execute('''
                CREATE INDEX IF NOT EXISTS idx_session_id ON messages(session_id)
            ''')
            self.connection.commit()
    
    def store_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Store a message in SQLite."""
        with self.lock:
            self.connection.execute('''
                INSERT INTO messages (session_id, role, content, metadata)
                VALUES (?, ?, ?, ?)
            ''', (session_id, role, content, json.dumps(metadata or {})))
            self.connection.commit()
    
    def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve messages from SQLite."""
        with self.lock:
            query = '''
                SELECT role, content, metadata, timestamp
                FROM messages
                WHERE session_id = ?
                ORDER BY timestamp ASC
            '''
            if limit:
                query += f' LIMIT {limit}'
            
            rows = self.connection.execute(query, (session_id,)).fetchall()
            
            messages = []
            for row in rows:
                messages.append({
                    'role': row['role'],
                    'content': row['content'],
                    'metadata': json.loads(row['metadata']),
                    'timestamp': row['timestamp']
                })
            
            return messages
    
    def clear_session(self, session_id: str) -> None:
        """Clear session messages from SQLite."""
        with self.lock:
            self.connection.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
            self.connection.commit()
    
    def __del__(self):
        """Close connection on cleanup."""
        if hasattr(self, 'connection'):
            self.connection.close()


class PythonDictBackend(MemoryBackend):
    """Python dictionary-based in-memory backend."""
    
    def __init__(self, config: Config):
        self.messages: Dict[str, List[Dict[str, Any]]] = {}
        self.lock = threading.Lock()
    
    def store_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Store a message in memory."""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        with self.lock:
            if session_id not in self.messages:
                self.messages[session_id] = []
            self.messages[session_id].append(message)
    
    def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve messages from memory."""
        with self.lock:
            messages = self.messages.get(session_id, [])
            if limit:
                return messages[-limit:]  # Get last N messages
            return messages
    
    def clear_session(self, session_id: str) -> None:
        """Clear session messages from memory."""
        with self.lock:
            if session_id in self.messages:
                del self.messages[session_id]


class RedisMemoryBackend(MemoryBackend):
    """Redis-based memory backend (optional)."""
    
    def __init__(self, config: Config):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is not available. Install with: pip install redis")
        
        memory_config = config.memory.redis
        if not memory_config:
            raise ValueError("Redis memory configuration not found")
        
        self.redis_client = redis.Redis(
            host=memory_config.get('host', 'localhost'),
            port=memory_config.get('port', 6379),
            db=memory_config.get('db', 0),
            password=memory_config.get('password'),
            decode_responses=True
        )
        
        # Test connection
        try:
            self.redis_client.ping()
        except redis.ConnectionError:
            raise ConnectionError("Failed to connect to Redis server")
    
    def store_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Store a message in Redis."""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        key = f"session:{session_id}"
        self.redis_client.lpush(key, json.dumps(message))
        self.redis_client.expire(key, 86400 * 7)  # Expire after 7 days
    
    def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve messages from Redis."""
        key = f"session:{session_id}"
        messages = self.redis_client.lrange(key, 0, limit - 1 if limit else -1)
        
        # Reverse to get chronological order
        messages.reverse()
        return [json.loads(msg) for msg in messages]
    
    def clear_session(self, session_id: str) -> None:
        """Clear session messages from Redis."""
        key = f"session:{session_id}"
        self.redis_client.delete(key)


class SQLiteMemoryBackend(MemoryBackend):
    """SQLite-based memory backend."""
    
    def __init__(self, config: Config):
        memory_config = config.memory.sqlite
        if not memory_config:
            raise ValueError("SQLite memory configuration not found")
        
        self.db_path = memory_config.get('database_url', 'sqlite:///./agent_memory.db')
        if self.db_path.startswith('sqlite:///'):
            self.db_path = self.db_path[10:]  # Remove sqlite:/// prefix
        
        # Create database directory if it doesn't exist
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
    
    def _init_database(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_session_id ON messages(session_id)
            ''')
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper error handling."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def store_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Store a message in SQLite."""
        with self._get_connection() as conn:
            conn.execute('''
                INSERT INTO messages (session_id, role, content, metadata)
                VALUES (?, ?, ?, ?)
            ''', (session_id, role, content, json.dumps(metadata or {})))
    
    def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve messages from SQLite."""
        with self._get_connection() as conn:
            query = '''
                SELECT role, content, metadata, timestamp
                FROM messages
                WHERE session_id = ?
                ORDER BY timestamp ASC
            '''
            if limit:
                query += f' LIMIT {limit}'
            
            rows = conn.execute(query, (session_id,)).fetchall()
            
            messages = []
            for row in rows:
                messages.append({
                    'role': row['role'],
                    'content': row['content'],
                    'metadata': json.loads(row['metadata']),
                    'timestamp': row['timestamp']
                })
            
            return messages
    
    def clear_session(self, session_id: str) -> None:
        """Clear session messages from SQLite."""
        with self._get_connection() as conn:
            conn.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))


class MemoryManager:
    """Memory manager that handles different backends."""
    
    def __init__(self, config: Config):
        self.config = config
        self.backend = self._create_backend()
    
    def _create_backend(self) -> MemoryBackend:
        """Create the appropriate memory backend."""
        backend_type = self.config.supervisor.memory_backend
        
        if backend_type == "in_memory_sqlite":
            return InMemorySQLiteBackend(self.config)
        elif backend_type == "in_memory_dict":
            return PythonDictBackend(self.config)
        elif backend_type == "sqlite":
            return SQLiteMemoryBackend(self.config)
        elif backend_type == "redis":
            return RedisMemoryBackend(self.config)
        else:
            # Default to in-memory SQLite
            return InMemorySQLiteBackend(self.config)
    
    def store_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Store a message."""
        self.backend.store_message(session_id, role, content, metadata)
    
    def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve messages for a session."""
        return self.backend.get_messages(session_id, limit)
    
    def clear_session(self, session_id: str) -> None:
        """Clear session messages."""
        self.backend.clear_session(session_id)
    
    def get_conversation_context(self, session_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """Get conversation context for LLM."""
        messages = self.get_messages(session_id, limit)
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
        ]
