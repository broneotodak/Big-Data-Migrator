"""
Supabase database connection management and operations.
"""
import os
import time
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from functools import wraps

import httpx
from supabase import create_client, Client
from tenacity import retry, stop_after_attempt, wait_exponential

from app.utils.logging_config import get_logger

logger = get_logger(__name__)

class SupabaseManager:
    """
    Manages Supabase database connections and operations with connection pooling,
    retry logic, and circuit breaker pattern.
    """
    
    # Circuit breaker settings
    MAX_FAILURES = 3
    RESET_TIMEOUT = 60  # seconds
    
    # Connection pool settings
    MAX_CONNECTIONS = 10
    CONNECTION_TIMEOUT = 30  # seconds
    
    def __init__(self):
        """Initialize the Supabase manager with connection pooling."""
        self._client: Optional[Client] = None
        self._connection_pool: List[Client] = []
        self._failure_count = 0
        self._last_failure_time = None
        self._circuit_open = False
        
        # Initialize connection
        self._initialize_connection()
        
    def _initialize_connection(self) -> None:
        """Initialize the Supabase client with environment variables."""
        try:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_KEY")
            
            if not supabase_url or not supabase_key:
                raise ValueError("Missing Supabase configuration. Set SUPABASE_URL and SUPABASE_KEY environment variables.")
            
            # Create main client
            self._client = create_client(supabase_url, supabase_key)
            
            # Initialize connection pool
            for _ in range(self.MAX_CONNECTIONS):
                client = create_client(supabase_url, supabase_key)
                self._connection_pool.append(client)
                
            logger.info("Supabase connection pool initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Supabase connection: {str(e)}")
            raise
            
    def _get_connection(self) -> Client:
        """Get a connection from the pool with circuit breaker pattern."""
        if self._circuit_open:
            if time.time() - self._last_failure_time > self.RESET_TIMEOUT:
                self._circuit_open = False
                self._failure_count = 0
            else:
                raise ConnectionError("Circuit breaker is open. Too many recent failures.")
                
        if not self._connection_pool:
            raise ConnectionError("No available connections in the pool")
            
        return self._connection_pool.pop()
        
    def _release_connection(self, client: Client) -> None:
        """Release a connection back to the pool."""
        if len(self._connection_pool) < self.MAX_CONNECTIONS:
            self._connection_pool.append(client)
            
    def _handle_operation_failure(self) -> None:
        """Handle operation failure with circuit breaker pattern."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._failure_count >= self.MAX_FAILURES:
            self._circuit_open = True
            logger.error("Circuit breaker opened due to repeated failures")
            
    def _handle_operation_success(self) -> None:
        """Handle successful operation."""
        self._failure_count = 0
        self._circuit_open = False
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def execute_query(self, 
                     query: str, 
                     params: Optional[Dict[str, Any]] = None,
                     timeout: Optional[float] = None) -> Any:
        """
        Execute a database query with retry logic.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            timeout: Query timeout in seconds
            
        Returns:
            Query results
            
        Raises:
            ConnectionError: If connection fails
            QueryError: If query execution fails
        """
        client = None
        try:
            client = self._get_connection()
            result = client.rpc(query, params).execute()
            self._handle_operation_success()
            return result
            
        except Exception as e:
            self._handle_operation_failure()
            logger.error(f"Query execution failed: {str(e)}")
            raise
            
        finally:
            if client:
                self._release_connection(client)
                
    def execute_transaction(self, 
                          operations: List[Dict[str, Any]],
                          timeout: Optional[float] = None) -> Any:
        """
        Execute a transaction with multiple operations.
        
        Args:
            operations: List of operations to execute
            timeout: Transaction timeout in seconds
            
        Returns:
            Transaction results
            
        Raises:
            TransactionError: If transaction fails
        """
        client = None
        try:
            client = self._get_connection()
            
            # Start transaction
            client.postgrest.rpc('begin_transaction').execute()
            
            results = []
            for op in operations:
                result = client.rpc(op['query'], op.get('params')).execute()
                results.append(result)
                
            # Commit transaction
            client.postgrest.rpc('commit_transaction').execute()
            self._handle_operation_success()
            return results
            
        except Exception as e:
            if client:
                # Rollback transaction
                try:
                    client.postgrest.rpc('rollback_transaction').execute()
                except:
                    pass
                    
            self._handle_operation_failure()
            logger.error(f"Transaction failed: {str(e)}")
            raise
            
        finally:
            if client:
                self._release_connection(client)
                
    def check_connection_health(self) -> Dict[str, Any]:
        """
        Check the health of the database connection.
        
        Returns:
            Dictionary with connection health information
        """
        try:
            start_time = time.time()
            self.execute_query("SELECT 1")
            response_time = time.time() - start_time
            
            health = {
                "status": "healthy",
                "response_time": response_time,
                "active_connections": len(self._connection_pool),
                "circuit_breaker": "open" if self._circuit_open else "closed",
                "failure_count": self._failure_count
            }
            
            return health
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker": "open" if self._circuit_open else "closed",
                "failure_count": self._failure_count
            }
            
    def close(self) -> None:
        """Close all database connections."""
        try:
            for client in self._connection_pool:
                client.close()
            self._connection_pool.clear()
            logger.info("All database connections closed")
            
        except Exception as e:
            logger.error(f"Error closing database connections: {str(e)}")
            
    def __del__(self):
        """Cleanup when the object is deleted."""
        self.close() 