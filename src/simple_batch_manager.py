"""
Simple batch management for ticker grouping and batch creation.

This module provides fast, simple batch creation strategies focused on performance
without complex analysis or grouping algorithms.
"""

import logging
import math
from typing import List, Optional, Tuple
import psutil
import numpy as np

from batch_forecasting_config import BatchConfig

logger = logging.getLogger(__name__)


class SimpleBatchManager:
    """
    Simple batch management focused on performance without complex analysis.
    
    This class creates ticker batches using simple, fast strategies:
    1. Single batch (default): Process all tickers together
    2. Size-based chunking: Split into equal-sized chunks if memory constraints exist
    """
    
    def __init__(self, config: BatchConfig):
        """
        Initialize batch manager with configuration.
        
        Args:
            config: Batch configuration settings
        """
        self.config = config
        self.logger = logging.getLogger('batch_forecasting.batch_manager')
        
    def create_batches(self, 
                      tickers: List[str], 
                      max_batch_size: Optional[int] = None,
                      memory_limit_mb: Optional[int] = None) -> List[List[str]]:
        """
        Create batches using simple, fast strategies.
        
        Args:
            tickers: List of ticker symbols to batch
            max_batch_size: Override for maximum batch size
            memory_limit_mb: Memory limit for automatic batch sizing
            
        Returns:
            List of ticker batches
        """
        if not tickers:
            self.logger.warning("Empty ticker list provided")
            return []
        
        self.logger.info(f"Creating batches for {len(tickers)} tickers using {self.config.strategy} strategy")
        
        # Determine effective batch size
        effective_max_size = max_batch_size or self.config.max_batch_size
        
        # Check memory constraints if enabled
        if memory_limit_mb and self._should_limit_batch_size(len(tickers), memory_limit_mb):
            estimated_safe_size = self._estimate_safe_batch_size(len(tickers), memory_limit_mb)
            if effective_max_size is None or estimated_safe_size < effective_max_size:
                effective_max_size = estimated_safe_size
                self.logger.info(f"Memory constraint applied: limiting batch size to {effective_max_size}")
        
        # Apply batching strategy
        if self.config.strategy == "single_batch" and (effective_max_size is None or len(tickers) <= effective_max_size):
            batches = self._single_batch_strategy(tickers)
        else:
            # Use chunking strategy
            chunk_size = effective_max_size or self.config.chunk_size
            batches = self._chunk_based_strategy(tickers, chunk_size)
        
        self.logger.info(f"Created {len(batches)} batches with sizes: {[len(batch) for batch in batches]}")
        return batches
    
    def _single_batch_strategy(self, tickers: List[str]) -> List[List[str]]:
        """
        Return all tickers as a single batch.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Single batch containing all tickers
        """
        self.logger.debug(f"Using single batch strategy for {len(tickers)} tickers")
        return [tickers]
    
    def _chunk_based_strategy(self, tickers: List[str], max_batch_size: int) -> List[List[str]]:
        """
        Split tickers into equal-sized chunks.
        
        Args:
            tickers: List of ticker symbols
            max_batch_size: Maximum size for each batch
            
        Returns:
            List of ticker batches
        """
        self.logger.debug(f"Using chunk-based strategy: {len(tickers)} tickers, max size {max_batch_size}")
        
        if max_batch_size <= 0:
            raise ValueError(f"Batch size must be positive: {max_batch_size}")
        
        batches = []
        for i in range(0, len(tickers), max_batch_size):
            batch = tickers[i:i + max_batch_size]
            batches.append(batch)
        
        return batches
    
    def _should_limit_batch_size(self, num_tickers: int, memory_limit_mb: int) -> bool:
        """
        Determine if batch size should be limited based on memory constraints.
        
        Args:
            num_tickers: Number of tickers to process
            memory_limit_mb: Memory limit in MB
            
        Returns:
            True if batch size should be limited
        """
        try:
            # Get current memory usage
            memory_info = psutil.virtual_memory()
            available_mb = memory_info.available / (1024 * 1024)
            
            # Estimate memory needed for batch processing
            # Rough estimate: ~1MB per ticker for features + model data
            estimated_memory_mb = num_tickers * 1.5  # Conservative estimate
            
            should_limit = (available_mb < memory_limit_mb) or (estimated_memory_mb > available_mb * 0.7)
            
            if should_limit:
                self.logger.info(f"Memory constraint detected: available={available_mb:.0f}MB, "
                               f"estimated_need={estimated_memory_mb:.0f}MB, limit={memory_limit_mb}MB")
            
            return should_limit
            
        except Exception as e:
            self.logger.warning(f"Failed to check memory constraints: {e}")
            return False
    
    def _estimate_safe_batch_size(self, num_tickers: int, memory_limit_mb: int) -> int:
        """
        Estimate safe batch size based on available memory.
        
        Args:
            num_tickers: Total number of tickers
            memory_limit_mb: Memory limit in MB
            
        Returns:
            Estimated safe batch size
        """
        try:
            memory_info = psutil.virtual_memory()
            available_mb = memory_info.available / (1024 * 1024)
            
            # Use the more restrictive limit
            effective_limit = min(memory_limit_mb, available_mb * 0.7)
            
            # Conservative estimate: 1.5MB per ticker
            safe_batch_size = max(1, int(effective_limit / 1.5))
            
            # Don't exceed the total number of tickers
            safe_batch_size = min(safe_batch_size, num_tickers)
            
            self.logger.debug(f"Estimated safe batch size: {safe_batch_size} "
                            f"(available: {available_mb:.0f}MB, limit: {effective_limit:.0f}MB)")
            
            return safe_batch_size
            
        except Exception as e:
            self.logger.warning(f"Failed to estimate safe batch size: {e}")
            # Conservative fallback
            return min(10, num_tickers)
    
    def optimize_batch_sizes(self, 
                           tickers: List[str], 
                           target_batches: int) -> List[List[str]]:
        """
        Create optimally sized batches for a target number of batches.
        
        Args:
            tickers: List of ticker symbols
            target_batches: Target number of batches to create
            
        Returns:
            List of optimally sized ticker batches
        """
        if target_batches <= 0:
            raise ValueError(f"Target batches must be positive: {target_batches}")
        
        if target_batches >= len(tickers):
            # One ticker per batch
            return [[ticker] for ticker in tickers]
        
        # Calculate optimal batch size
        base_size = len(tickers) // target_batches
        remainder = len(tickers) % target_batches
        
        batches = []
        start_idx = 0
        
        for i in range(target_batches):
            # Distribute remainder across first few batches
            batch_size = base_size + (1 if i < remainder else 0)
            end_idx = start_idx + batch_size
            
            batch = tickers[start_idx:end_idx]
            batches.append(batch)
            
            start_idx = end_idx
        
        self.logger.info(f"Created {len(batches)} optimized batches with sizes: {[len(b) for b in batches]}")
        return batches
    
    def get_batch_statistics(self, batches: List[List[str]]) -> dict:
        """
        Get statistics about created batches.
        
        Args:
            batches: List of ticker batches
            
        Returns:
            Dictionary with batch statistics
        """
        if not batches:
            return {
                'num_batches': 0,
                'total_tickers': 0,
                'avg_batch_size': 0,
                'min_batch_size': 0,
                'max_batch_size': 0,
                'batch_sizes': []
            }
        
        batch_sizes = [len(batch) for batch in batches]
        total_tickers = sum(batch_sizes)
        
        return {
            'num_batches': len(batches),
            'total_tickers': total_tickers,
            'avg_batch_size': total_tickers / len(batches),
            'min_batch_size': min(batch_sizes),
            'max_batch_size': max(batch_sizes),
            'batch_sizes': batch_sizes,
            'strategy_used': self.config.strategy
        }
    
    def validate_batches(self, batches: List[List[str]], original_tickers: List[str]) -> bool:
        """
        Validate that batches contain all original tickers without duplicates.
        
        Args:
            batches: List of ticker batches
            original_tickers: Original list of tickers
            
        Returns:
            True if validation passes
        """
        try:
            # Flatten batches
            flattened = [ticker for batch in batches for ticker in batch]
            
            # Check for missing tickers
            missing = set(original_tickers) - set(flattened)
            if missing:
                self.logger.error(f"Missing tickers in batches: {missing}")
                return False
            
            # Check for duplicate tickers
            if len(flattened) != len(set(flattened)):
                duplicates = [ticker for ticker in flattened if flattened.count(ticker) > 1]
                self.logger.error(f"Duplicate tickers in batches: {set(duplicates)}")
                return False
            
            # Check for extra tickers
            extra = set(flattened) - set(original_tickers)
            if extra:
                self.logger.error(f"Extra tickers in batches: {extra}")
                return False
            
            self.logger.debug("Batch validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Batch validation failed: {e}")
            return False