"""
Cache Warming System for Portfolio Optimization
Pre-loads popular data into cache for optimal performance
"""

import time
import threading
import logging
from datetime import datetime, timedelta
from typing import List, Dict

from cache_manager import get_cache
from portfolio_optimization import get_stock_data, _forecast_single_ticker
from ticker_lists import get_ticker_group

logger = logging.getLogger(__name__)

class CacheWarmer:
    """Intelligent cache warming system for portfolio optimization"""
    
    def __init__(self):
        self.cache = get_cache()
        self.warming_active = False
        self.popular_tickers = [
            # FAANG + major tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX',
            # Major indices components
            'SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VOO',
            # Blue chip stocks
            'BRK-B', 'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA',
            # Popular growth stocks
            'CRM', 'ADBE', 'PYPL', 'INTC', 'AMD', 'ORCL', 'IBM', 'CSCO'
        ]
        self.common_date_ranges = [
            ('1Y', 365),
            ('2Y', 730),
            ('5Y', 1825)
        ]
        
    def warm_popular_data(self):
        """Warm cache with popular ticker data"""
        if self.warming_active:
            logger.info("Cache warming already in progress, skipping")
            return
            
        self.warming_active = True
        start_time = time.time()
        
        try:
            logger.info("Starting cache warming for popular data")
            
            # Warm individual popular tickers
            self._warm_individual_tickers()
            
            # Warm popular index groups
            self._warm_index_groups()
            
            # Warm forecast data for top tickers
            self._warm_forecasts()
            
            elapsed = time.time() - start_time
            logger.info(f"Cache warming completed in {elapsed:.1f} seconds")
            
        except Exception as e:
            logger.error(f"Cache warming failed: {e}")
        finally:
            self.warming_active = False
            
    def _warm_individual_tickers(self):
        """Warm cache for individual popular tickers"""
        logger.info(f"Warming cache for {len(self.popular_tickers)} popular tickers")
        
        for name, days in self.common_date_ranges:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            for ticker in self.popular_tickers[:20]:  # Top 20 most popular
                try:
                    # This will cache the data if not already cached
                    data = get_stock_data([ticker], start_date, end_date)
                    if not data.empty:
                        logger.debug(f"Warmed cache for {ticker} ({name})")
                except Exception as e:
                    logger.warning(f"Failed to warm cache for {ticker}: {e}")
                    
    def _warm_index_groups(self):
        """Warm cache for popular index groups"""
        popular_groups = ['sp500', 'nasdaq100', 'dow30']
        
        for group in popular_groups:
            try:
                tickers = get_ticker_group(group)
                if tickers:
                    # Warm 1-year data for the full index
                    end_date = datetime.now().strftime('%Y-%m-%d')
                    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
                    
                    logger.info(f"Warming cache for {group} ({len(tickers)} tickers)")
                    data = get_stock_data(tickers[:50], start_date, end_date)  # Top 50 to avoid overwhelming
                    if not data.empty:
                        logger.info(f"Successfully warmed {group} cache")
                        
            except Exception as e:
                logger.warning(f"Failed to warm cache for {group}: {e}")
                
    def _warm_forecasts(self):
        """Warm forecast cache for top tickers"""
        logger.info("Warming forecast cache for top tickers")
        
        # Get recent data for forecasting
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')  # 2 years
        
        try:
            # Get data for top 10 tickers
            top_tickers = self.popular_tickers[:10]
            data = get_stock_data(top_tickers, start_date, end_date)
            
            if not data.empty:
                for ticker in data.columns:
                    try:
                        # Warm both lightweight and Prophet forecasts
                        _forecast_single_ticker(ticker, data[ticker], use_lightweight=True)
                        _forecast_single_ticker(ticker, data[ticker], use_lightweight=False)
                        logger.debug(f"Warmed forecast cache for {ticker}")
                    except Exception as e:
                        logger.warning(f"Failed to warm forecast for {ticker}: {e}")
                        
        except Exception as e:
            logger.warning(f"Failed to warm forecast cache: {e}")
            
    def start_background_warming(self):
        """Start background cache warming with simple timer-based approach"""
        logger.info("Starting background cache warming system")
        
        def background_warmer():
            """Background thread for periodic cache warming"""
            while True:
                try:
                    # Warm cache every 4 hours
                    time.sleep(4 * 3600)  # 4 hours
                    
                    if not self.warming_active:
                        logger.info("Background cache warming triggered")
                        self.warm_popular_data()
                        
                except Exception as e:
                    logger.error(f"Background cache warming error: {e}")
                    time.sleep(3600)  # Wait 1 hour on error
                    
        # Start background thread
        warmer_thread = threading.Thread(target=background_warmer, daemon=True)
        warmer_thread.start()
        logger.info("Background cache warming system started")
        
    def _refresh_hot_data(self):
        """Refresh frequently accessed data during market hours"""
        if not self._is_market_hours():
            return
            
        logger.debug("Refreshing hot data during market hours")
        
        # Refresh data for most popular tickers with short TTL
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')  # Last month
        
        try:
            hot_tickers = self.popular_tickers[:5]  # Top 5 most popular
            get_stock_data(hot_tickers, start_date, end_date)
            logger.debug(f"Refreshed hot data for {len(hot_tickers)} tickers")
        except Exception as e:
            logger.warning(f"Failed to refresh hot data: {e}")
            
    def _is_market_hours(self) -> bool:
        """Check if current time is during market hours (9:30 AM - 4:00 PM EST)"""
        now = datetime.now()
        # Simplified check - assumes Eastern Time
        return 9 <= now.hour < 16 and now.weekday() < 5  # Monday-Friday, 9 AM - 4 PM
        
    def get_warming_status(self) -> Dict:
        """Get current cache warming status"""
        cache_stats = self.cache.stats()
        return {
            'warming_active': self.warming_active,
            'cache_entries': cache_stats['l1_cache']['entries'],
            'memory_usage_mb': cache_stats['l1_cache']['memory_usage_mb'],
            'hit_ratios': cache_stats['hit_ratios'],
            'popular_tickers_count': len(self.popular_tickers),
            'background_warming_enabled': True
        }
        
    def force_warm_portfolio(self, tickers: List[str]):
        """Force warm cache for a specific portfolio"""
        logger.info(f"Force warming cache for portfolio: {tickers[:10]}{'...' if len(tickers) > 10 else ''}")
        
        # Warm multiple date ranges for the portfolio
        for name, days in self.common_date_ranges:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            try:
                data = get_stock_data(tickers, start_date, end_date)
                if not data.empty:
                    logger.info(f"Force warmed portfolio cache for {name} ({len(data.columns)} tickers)")
                    
                    # Also warm forecasts for the portfolio
                    for ticker in data.columns[:20]:  # Top 20 to avoid overwhelming
                        try:
                            _forecast_single_ticker(ticker, data[ticker], use_lightweight=True)
                        except Exception as e:
                            logger.debug(f"Failed to warm forecast for {ticker}: {e}")
                            
            except Exception as e:
                logger.warning(f"Failed to force warm portfolio for {name}: {e}")

# Global cache warmer instance
_global_warmer = None

def get_cache_warmer() -> CacheWarmer:
    """Get global cache warmer instance (singleton)"""
    global _global_warmer
    if _global_warmer is None:
        _global_warmer = CacheWarmer()
    return _global_warmer

def start_cache_warming():
    """Start the cache warming system"""
    warmer = get_cache_warmer()
    
    # Initial warming
    logger.info("Performing initial cache warming...")
    warmer.warm_popular_data()
    
    # Start background scheduler
    warmer.start_background_warming()
    
    logger.info("Cache warming system fully initialized")

def warm_portfolio_cache(tickers: List[str]):
    """Convenience function to warm cache for a specific portfolio"""
    warmer = get_cache_warmer()
    warmer.force_warm_portfolio(tickers)

def get_cache_warming_status() -> Dict:
    """Get current cache warming status"""
    warmer = get_cache_warmer()
    return warmer.get_warming_status()
