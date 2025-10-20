"""
Redis Manager for Kernel Session Management

This module provides Redis-based session management for kernel workers,
allowing persistent storage of kernel-to-worker mappings and session state.
"""

import redis.asyncio as redis
import json
import logging
import asyncio
from typing import Optional, Dict, Any
from config.settings import REDIS_URL, ENABLE_REDIS

logger = logging.getLogger(__name__)

class RedisManager:
    """
    Manages Redis connections and operations for kernel session management
    """
    
    def __init__(self, redis_url: str = None):
        """
        Initialize Redis manager
        
        Args:
            redis_url: Redis connection URL (defaults to settings)
        """
        self.redis_url = redis_url or REDIS_URL
        self.redis = None
        # Use Redis only if explicitly enabled
        self.use_redis = ENABLE_REDIS
        logger.info(f"🔧 Redis manager initialized with URL: {self.redis_url}")
        logger.info(f"📊 Redis enabled: {self.use_redis}")
        
        # Initialize in-memory storage for fallback
        self._kernel_workers = {}  # In-memory storage as fallback
        self._kernel_metadata = {}  # In-memory storage as fallback
        self._active_kernels = set()  # In-memory storage as fallback
    
    async def connect(self):
        """
        Establish connection to Redis
        """
        # Only connect to Redis if explicitly enabled
        if not self.use_redis:
            logger.info("⚠️ Redis explicitly disabled, using in-memory storage")
            return
            
        try:
            self.redis = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            # Test connection
            await self.redis.ping()
            logger.info("✅ Redis connection established")
        except Exception as e:
            logger.warning(f"⚠️ Failed to connect to Redis, falling back to in-memory storage: {e}")
            self.use_redis = False
            # In-memory storage is already initialized in __init__
    
    async def disconnect(self):
        """
        Close Redis connection
        """
        if self.redis and self.use_redis:
            await self.redis.close()
            logger.info("🧹 Redis connection closed")
        else:
            # Clear in-memory storage
            self._kernel_workers = {}
            self._kernel_metadata = {}
            self._active_kernels = set()
            logger.info("🧹 In-memory storage cleared")
    
    async def register_kernel(self, kernel_id: str, worker_id: str, 
                            client_id: str = None) -> bool:
        """
        Register a kernel session with its worker
        
        Args:
            kernel_id: Unique kernel identifier
            worker_id: Worker identifier handling this kernel
            client_id: Optional client identifier
            
        Returns:
            True if registration successful
        """
        try:
            if self.use_redis and self.redis:
                # Store kernel-to-worker mapping
                await self.redis.hset("kernel_workers", kernel_id, worker_id)
                
                # Store kernel metadata
                metadata = {
                    "worker_id": worker_id,
                    "client_id": client_id,
                    "created_at": str(asyncio.get_event_loop().time()),
                    "last_activity": str(asyncio.get_event_loop().time())
                }
                await self.redis.hset("kernel_metadata", kernel_id, json.dumps(metadata))
                
                # Add to active kernels set
                await self.redis.sadd("active_kernels", kernel_id)
            else:
                # Use in-memory storage
                self._kernel_workers[kernel_id] = worker_id
                self._kernel_metadata[kernel_id] = {
                    "worker_id": worker_id,
                    "client_id": client_id,
                    "created_at": str(asyncio.get_event_loop().time()),
                    "last_activity": str(asyncio.get_event_loop().time())
                }
                self._active_kernels.add(kernel_id)
            
            logger.info(f"📝 Registered kernel {kernel_id} with worker {worker_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to register kernel {kernel_id}: {e}")
            return False
    
    async def unregister_kernel(self, kernel_id: str) -> bool:
        """
        Unregister a kernel session
        
        Args:
            kernel_id: Kernel identifier to unregister
            
        Returns:
            True if unregistration successful
        """
        try:
            if self.use_redis and self.redis:
                # Remove from all mappings
                await self.redis.hdel("kernel_workers", kernel_id)
                await self.redis.hdel("kernel_metadata", kernel_id)
                await self.redis.srem("active_kernels", kernel_id)
            else:
                # Use in-memory storage
                self._kernel_workers.pop(kernel_id, None)
                self._kernel_metadata.pop(kernel_id, None)
                self._active_kernels.discard(kernel_id)
            
            logger.info(f"🗑️ Unregistered kernel {kernel_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to unregister kernel {kernel_id}: {e}")
            return False
    
    async def get_worker_for_kernel(self, kernel_id: str) -> Optional[str]:
        """
        Get the worker ID for a kernel session
        
        Args:
            kernel_id: Kernel identifier
            
        Returns:
            Worker ID or None if not found
        """
        try:
            if self.use_redis and self.redis:
                worker_id = await self.redis.hget("kernel_workers", kernel_id)
                return worker_id
            else:
                # Use in-memory storage
                return self._kernel_workers.get(kernel_id)
        except Exception as e:
            logger.error(f"❌ Failed to get worker for kernel {kernel_id}: {e}")
            return None
    
    async def get_kernel_metadata(self, kernel_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a kernel session
        
        Args:
            kernel_id: Kernel identifier
            
        Returns:
            Kernel metadata or None if not found
        """
        try:
            if self.use_redis and self.redis:
                metadata_str = await self.redis.hget("kernel_metadata", kernel_id)
                if metadata_str:
                    return json.loads(metadata_str)
                return None
            else:
                # Use in-memory storage
                return self._kernel_metadata.get(kernel_id)
        except Exception as e:
            logger.error(f"❌ Failed to get metadata for kernel {kernel_id}: {e}")
            return None
    
    async def update_kernel_activity(self, kernel_id: str) -> bool:
        """
        Update last activity timestamp for a kernel
        
        Args:
            kernel_id: Kernel identifier
            
        Returns:
            True if update successful
        """
        try:
            if self.use_redis and self.redis:
                metadata = await self.get_kernel_metadata(kernel_id)
                if metadata:
                    metadata["last_activity"] = str(asyncio.get_event_loop().time())
                    await self.redis.hset("kernel_metadata", kernel_id, json.dumps(metadata))
                    return True
                return False
            else:
                # Use in-memory storage
                if kernel_id in self._kernel_metadata:
                    self._kernel_metadata[kernel_id]["last_activity"] = str(asyncio.get_event_loop().time())
                    return True
                return False
        except Exception as e:
            logger.error(f"❌ Failed to update activity for kernel {kernel_id}: {e}")
            return False
    
    async def get_active_kernels(self) -> set:
        """
        Get all active kernel IDs
        
        Returns:
            Set of active kernel IDs
        """
        try:
            if self.use_redis and self.redis:
                kernels = await self.redis.smembers("active_kernels")
                return set(kernels) if kernels else set()
            else:
                # Use in-memory storage
                return self._active_kernels.copy()
        except Exception as e:
            logger.error(f"❌ Failed to get active kernels: {e}")
            return set()
    
    async def cleanup_expired_kernels(self, max_age: float = 3600.0) -> int:
        """
        Clean up expired kernel sessions
        
        Args:
            max_age: Maximum age in seconds before considering kernel expired
            
        Returns:
            Number of kernels cleaned up
        """
        try:
            active_kernels = await self.get_active_kernels()
            current_time = asyncio.get_event_loop().time()
            cleaned_count = 0
            
            for kernel_id in active_kernels:
                metadata = await self.get_kernel_metadata(kernel_id)
                if metadata:
                    last_activity = float(metadata.get("last_activity", 0))
                    if current_time - last_activity > max_age:
                        await self.unregister_kernel(kernel_id)
                        cleaned_count += 1
                        logger.info(f"🧹 Cleaned up expired kernel: {kernel_id}")
            
            if cleaned_count > 0:
                logger.info(f"🧹 Cleaned up {cleaned_count} expired kernels")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup expired kernels: {e}")
            return 0

# Global Redis manager instance
redis_manager = RedisManager()