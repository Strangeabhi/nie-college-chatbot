"""
Performance Optimization and Caching System for NIE Chatbot
Handles embedding caching, query caching, and response optimization
"""

import json
import pickle
import hashlib
import time
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
import numpy as np
from functools import wraps
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceCache:
    def __init__(self, cache_dir: str = "cache"):
        """Initialize the performance cache system"""
        self.cache_dir = cache_dir
        self.ensure_cache_dir()
        
        # Cache configurations
        self.cache_durations = {
            'embeddings': timedelta(days=7),      # Embeddings cache for 7 days
            'queries': timedelta(hours=6),        # Query responses for 6 hours
            'fallback': timedelta(hours=12),      # Fallback responses for 12 hours
            'similarities': timedelta(hours=1)    # Similarity calculations for 1 hour
        }
        
        # In-memory caches for frequently accessed data
        self.memory_caches = {
            'embeddings': {},
            'faq_data': {},
            'similarities': {}
        }
        
        logger.info(f"Performance cache initialized with directory: {self.cache_dir}")

    def ensure_cache_dir(self):
        """Ensure cache directory exists"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            logger.info(f"Created cache directory: {self.cache_dir}")

    def get_cache_key(self, data: Any, prefix: str = "") -> str:
        """Generate a cache key from data"""
        data_str = json.dumps(data, sort_keys=True) if isinstance(data, (dict, list)) else str(data)
        return f"{prefix}_{hashlib.md5(data_str.encode()).hexdigest()}"

    def get_cache_path(self, cache_key: str) -> str:
        """Get full path for cache file"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")

    def is_cache_valid(self, cache_key: str, cache_type: str) -> bool:
        """Check if cache entry is still valid"""
        cache_path = self.get_cache_path(cache_key)
        
        if not os.path.exists(cache_path):
            return False
        
        # Check file age
        file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_path))
        return file_age < self.cache_durations.get(cache_type, timedelta(hours=1))

    def get_from_cache(self, cache_key: str, cache_type: str = "queries") -> Optional[Any]:
        """Retrieve data from cache"""
        try:
            # Check in-memory cache first
            if cache_type in self.memory_caches and cache_key in self.memory_caches[cache_type]:
                logger.debug(f"Cache hit (memory): {cache_key}")
                return self.memory_caches[cache_type][cache_key]
            
            # Check disk cache
            if self.is_cache_valid(cache_key, cache_type):
                cache_path = self.get_cache_path(cache_key)
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Store in memory cache for faster access
                if cache_type in self.memory_caches:
                    self.memory_caches[cache_type][cache_key] = data
                
                logger.debug(f"Cache hit (disk): {cache_key}")
                return data
            
            logger.debug(f"Cache miss: {cache_key}")
            return None
            
        except Exception as e:
            logger.warning(f"Cache retrieval error: {e}")
            return None

    def save_to_cache(self, cache_key: str, data: Any, cache_type: str = "queries"):
        """Save data to cache"""
        try:
            # Save to memory cache
            if cache_type in self.memory_caches:
                self.memory_caches[cache_type][cache_key] = data
            
            # Save to disk cache
            cache_path = self.get_cache_path(cache_key)
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            
            logger.debug(f"Cache saved: {cache_key}")
            
        except Exception as e:
            logger.warning(f"Cache save error: {e}")

    def cache_query_response(self, query: str, response: str, confidence: float) -> str:
        """Cache a query response"""
        cache_key = self.get_cache_key(query, "query")
        cache_data = {
            'response': response,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        self.save_to_cache(cache_key, cache_data, 'queries')
        return cache_key

    def get_cached_response(self, query: str) -> Optional[Tuple[str, float]]:
        """Get cached response for query"""
        cache_key = self.get_cache_key(query, "query")
        cached_data = self.get_from_cache(cache_key, 'queries')
        
        if cached_data:
            return cached_data['response'], cached_data['confidence']
        return None

    def cache_embeddings(self, embeddings: np.ndarray, faq_hash: str):
        """Cache embeddings with FAQ hash"""
        cache_key = self.get_cache_key(faq_hash, "embeddings")
        self.save_to_cache(cache_key, embeddings, 'embeddings')
        logger.info(f"Embeddings cached with key: {cache_key}")

    def get_cached_embeddings(self, faq_hash: str) -> Optional[np.ndarray]:
        """Get cached embeddings"""
        cache_key = self.get_cache_key(faq_hash, "embeddings")
        return self.get_from_cache(cache_key, 'embeddings')

    def cache_similarities(self, query: str, similarities: np.ndarray):
        """Cache similarity calculations"""
        cache_key = self.get_cache_key(query, "similarities")
        self.save_to_cache(cache_key, similarities, 'similarities')

    def get_cached_similarities(self, query: str) -> Optional[np.ndarray]:
        """Get cached similarities"""
        cache_key = self.get_cache_key(query, "similarities")
        return self.get_from_cache(cache_key, 'similarities')

    def cache_fallback_response(self, query: str, response: str, confidence: float):
        """Cache fallback response"""
        cache_key = self.get_cache_key(query, "fallback")
        cache_data = {
            'response': response,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        self.save_to_cache(cache_key, cache_data, 'fallback')

    def get_cached_fallback(self, query: str) -> Optional[Tuple[str, float]]:
        """Get cached fallback response"""
        cache_key = self.get_cache_key(query, "fallback")
        cached_data = self.get_from_cache(cache_key, 'fallback')
        
        if cached_data:
            return cached_data['response'], cached_data['confidence']
        return None

    def clear_cache(self, cache_type: str = None):
        """Clear cache entries"""
        try:
            if cache_type:
                # Clear specific cache type
                if cache_type in self.memory_caches:
                    self.memory_caches[cache_type].clear()
                
                # Clear disk cache files
                for filename in os.listdir(self.cache_dir):
                    if filename.startswith(f"{cache_type}_") and filename.endswith('.pkl'):
                        os.remove(os.path.join(self.cache_dir, filename))
                
                logger.info(f"Cleared {cache_type} cache")
            else:
                # Clear all caches
                for cache in self.memory_caches.values():
                    cache.clear()
                
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith('.pkl'):
                        os.remove(os.path.join(self.cache_dir, filename))
                
                logger.info("Cleared all caches")
                
        except Exception as e:
            logger.error(f"Cache clear error: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            'memory_cache_sizes': {k: len(v) for k, v in self.memory_caches.items()},
            'disk_cache_files': len([f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]),
            'cache_dir': self.cache_dir,
            'cache_durations': {k: str(v) for k, v in self.cache_durations.items()}
        }
        return stats


def cached_query_response(cache_instance: PerformanceCache):
    """Decorator to cache query responses"""
    def decorator(func):
        @wraps(func)
        def wrapper(query, *args, **kwargs):
            # Try to get from cache first
            cached = cache_instance.get_cached_response(query)
            if cached:
                logger.debug(f"Returning cached response for: {query}")
                return cached
            
            # Execute function and cache result
            result = func(query, *args, **kwargs)
            if isinstance(result, tuple) and len(result) == 2:
                response, confidence = result
                cache_instance.cache_query_response(query, response, confidence)
            
            return result
        return wrapper
    return decorator


def cached_fallback_response(cache_instance: PerformanceCache):
    """Decorator to cache fallback responses"""
    def decorator(func):
        @wraps(func)
        def wrapper(query, *args, **kwargs):
            # Try to get from cache first
            cached = cache_instance.get_cached_fallback(query)
            if cached:
                logger.debug(f"Returning cached fallback for: {query}")
                return cached
            
            # Execute function and cache result
            result = func(query, *args, **kwargs)
            if isinstance(result, tuple) and len(result) == 2:
                response, confidence = result
                cache_instance.cache_fallback_response(query, response, confidence)
            
            return result
        return wrapper
    return decorator


# Global cache instance
performance_cache = None

def get_cache():
    """Get global cache instance"""
    global performance_cache
    if performance_cache is None:
        performance_cache = PerformanceCache()
    return performance_cache

if __name__ == "__main__":
    # Test the performance cache
    cache = get_cache()
    
    # Test caching
    test_query = "What are the CSE cutoffs?"
    test_response = "KCET cutoff for CSE is 8726"
    test_confidence = 0.9
    
    # Cache a response
    cache.cache_query_response(test_query, test_response, test_confidence)
    
    # Retrieve cached response
    cached = cache.get_cached_response(test_query)
    print(f"Cached response: {cached}")
    
    # Get cache stats
    stats = cache.get_cache_stats()
    print(f"Cache stats: {json.dumps(stats, indent=2)}")
