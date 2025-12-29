"""
Cache System Initialization for Portfolio Optimization
Integrates the multi-level caching system with the Flask application
"""

import logging
import time
from cache_manager import get_cache
from cache_warmer import start_cache_warming, get_cache_warming_status

logger = logging.getLogger(__name__)

def initialize_cache_system():
    """Initialize the complete caching system for portfolio optimization"""
    start_time = time.time()
    logger.info("Initializing portfolio optimization cache system...")
    
    try:
        # Initialize the cache manager
        cache = get_cache()
        logger.info("Cache manager initialized successfully")
        
        # Start cache warming system (runs in background)
        start_cache_warming()
        
        # Log initial status
        status = get_cache_warming_status()
        logger.info(f"Cache system initialized in {time.time() - start_time:.2f} seconds")
        logger.info(f"Cache warming status: {status}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize cache system: {e}")
        return False

def get_cache_status():
    """Get comprehensive cache system status"""
    try:
        cache = get_cache()
        cache_stats = cache.stats()
        warming_status = get_cache_warming_status()
        
        return {
            'cache_initialized': True,
            'cache_stats': cache_stats,
            'warming_status': warming_status,
            'performance_summary': {
                'l1_hit_ratio': cache_stats['hit_ratios']['l1'],
                'l2_hit_ratio': cache_stats['hit_ratios']['l2'],
                'overall_hit_ratio': cache_stats['hit_ratios']['overall'],
                'memory_usage_mb': cache_stats['l1_cache']['memory_usage_mb'],
                'memory_utilization': cache_stats['l1_cache']['memory_utilization']
            }
        }
    except Exception as e:
        logger.error(f"Failed to get cache status: {e}")
        return {
            'cache_initialized': False,
            'error': str(e)
        }

if __name__ == "__main__":
    # Test initialization
    logging.basicConfig(level=logging.INFO)
    success = initialize_cache_system()
    
    if success:
        print("✅ Cache system initialized successfully!")
        
        # Wait a moment for initial warming
        time.sleep(5)
        
        # Show status
        status = get_cache_status()
        print(f"📊 Cache Status: {status['performance_summary']}")
    else:
        print("❌ Cache system initialization failed!")
