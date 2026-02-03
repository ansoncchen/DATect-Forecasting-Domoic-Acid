"""
Cache Manager for DATect API
Manages pre-computed results for Google Cloud deployment
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages pre-computed cache file access"""
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.enabled = self._should_enable_cache()
        
        if self.enabled and not self.cache_dir.exists():
            logger.warning(f"Cache directory {cache_dir} not found. Run precompute_cache.py first.")
    
    def _should_enable_cache(self) -> bool:
        """Check if cache should be enabled based on environment"""
        if os.getenv("CACHE_DIR") == "/app/cache":
            return True
        if os.getenv("ENABLE_PRECOMPUTED_CACHE", "").lower() == "true":
            return True
        if os.getenv("NODE_ENV") == "production":
            return True
        
        return False
            
        
        
    def get_retrospective_forecast(self, task: str, model_type: str) -> Optional[List[Dict]]:
        """
        Get cached retrospective forecast results.
        
        Args:
            task: "regression" or "classification"
            model_type: "xgboost" or "linear"
            
        Returns:
            List of forecast results or None if not cached
        """
        if not self.enabled:
            return None
        cache_file = self.cache_dir / "retrospective" / f"{task}_{model_type}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                # Clean any inf/nan values from cached data
                import math
                def clean_cached_data(item):
                    if isinstance(item, dict):
                        cleaned = {}
                        for k, v in item.items():
                            if isinstance(v, float):
                                if math.isinf(v) or math.isnan(v):
                                    cleaned[k] = None
                                else:
                                    cleaned[k] = v
                            else:
                                cleaned[k] = v
                        return cleaned
                    return item
                
                cleaned_data = [clean_cached_data(item) for item in data]
                logger.info(f"Served cached retrospective forecast: {task}+{model_type} ({len(cleaned_data)} records)")
                return cleaned_data
            except Exception as e:
                logger.error(f"Failed to load cached retrospective forecast: {e}")
                
        logger.warning(f"No cached data for retrospective forecast: {task}+{model_type}")
        return None
        
    def get_spectral_analysis(self, site: Optional[str] = None) -> Optional[List[Dict]]:
        """
        Get cached spectral analysis results.
        
        Args:
            site: Site name or None for aggregate
            
        Returns:
            Spectral analysis plots or None if not cached
        """
        if not self.enabled:
            return None
        site_name = site or "all_sites"
        cache_file = self.cache_dir / "spectral" / f"{site_name}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                logger.info(f"Served cached spectral analysis: {site_name}")
                return data
            except Exception as e:
                logger.error(f"Failed to load cached spectral analysis: {e}")
                
        logger.warning(f"No cached spectral analysis for site: {site_name}")
        return None

    def get_correlation_heatmap(self, site: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Get cached correlation heatmap results for a site.

        Args:
            site: Site name or None for all sites (not currently cached)

        Returns:
            Correlation heatmap JSON or None if not cached
        """
        if not self.enabled or site is None:
            return None

        cache_file = self.cache_dir / "visualizations" / f"{site}_correlation.json"

        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                logger.info(f"Served cached correlation heatmap: {site}")
                return data
            except Exception as e:
                logger.error(f"Failed to load cached correlation heatmap: {e}")

        logger.warning(f"No cached correlation heatmap for site: {site}")
        return None

# Global cache manager instance
cache_manager = CacheManager()