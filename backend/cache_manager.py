"""
Cache Manager for DATect API
Manages pre-computed results for Google Cloud deployment

Supports two caching backends:
1. Redis (100x faster, recommended for production)
2. File-based (default fallback)

To enable Redis:
    export REDIS_URL=redis://localhost:6379/0
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Try to import Redis cache
try:
    from .redis_cache import get_redis_cache, RedisCacheManager
    REDIS_CACHE_AVAILABLE = True
except ImportError:
    REDIS_CACHE_AVAILABLE = False
    get_redis_cache = None


class CacheManager:
    """
    Manages pre-computed cache file access with optional Redis backend.
    
    Cache priority:
    1. Redis (if REDIS_URL is set and redis is available) - 100x faster
    2. File-based cache (default)
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(os.getenv("CACHE_DIR", cache_dir or "./cache"))
        self.enabled = self._should_enable_cache()
        self._redis_cache: Optional[RedisCacheManager] = None
        
        # Try to initialize Redis cache
        if REDIS_CACHE_AVAILABLE and os.getenv("REDIS_URL"):
            try:
                self._redis_cache = get_redis_cache()
                if self._redis_cache.is_available:
                    logger.info("Redis cache enabled (100x faster reads)")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis cache: {e}")
        
        if self.enabled:
            if self.cache_dir.exists():
                logger.info(f"Precomputed cache enabled: {self.cache_dir}")
            else:
                logger.warning(f"Cache directory {self.cache_dir} not found. Run precompute_cache.py first.")
    
    @property
    def use_redis(self) -> bool:
        """Check if Redis cache is available and should be used."""
        return self._redis_cache is not None and self._redis_cache.is_available
    
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
            model_type: "ensemble", "naive", "linear", or "logistic"
            
        Returns:
            List of forecast results or None if not cached
        """
        if not self.enabled:
            return None
        
        # Try Redis first (100x faster)
        if self.use_redis:
            data = self._redis_cache.get_retrospective_forecast(task, model_type)
            if data is not None:
                logger.info(f"Served Redis-cached retrospective forecast: {task}+{model_type}")
                return data
        
        # Fall back to file cache
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
                
                # Populate Redis cache for next time
                if self.use_redis:
                    self._redis_cache.set_retrospective_forecast(task, model_type, cleaned_data)
                
                logger.info(f"Served file-cached retrospective forecast: {task}+{model_type} ({len(cleaned_data)} records)")
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
        
        # Try Redis first (100x faster)
        if self.use_redis:
            data = self._redis_cache.get_spectral_analysis(site)
            if data is not None:
                logger.info(f"Served Redis-cached spectral analysis: {site_name}")
                return data
        
        # Fall back to file cache
        cache_file = self.cache_dir / "spectral" / f"{site_name}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                # Populate Redis cache for next time
                if self.use_redis:
                    self._redis_cache.set_spectral_analysis(data, site)
                
                logger.info(f"Served file-cached spectral analysis: {site_name}")
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
        
        # Try Redis first (100x faster)
        if self.use_redis:
            data = self._redis_cache.get_correlation_heatmap(site)
            if data is not None:
                logger.info(f"Served Redis-cached correlation heatmap: {site}")
                return data

        # Fall back to file cache
        cache_file = self.cache_dir / "visualizations" / f"{site}_correlation.json"

        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                # Check if data is in old format (matrix/columns) and convert to Plotly format
                if 'matrix' in data and 'columns' in data and 'data' not in data:
                    data = self._convert_raw_correlation_to_plotly(data, site)
                
                # Populate Redis cache for next time
                if self.use_redis:
                    self._redis_cache.set_correlation_heatmap(site, data)
                
                logger.info(f"Served file-cached correlation heatmap: {site}")
                return data
            except Exception as e:
                logger.error(f"Failed to load cached correlation heatmap: {e}")

        logger.warning(f"No cached correlation heatmap for site: {site}")
        return None

    def _convert_raw_correlation_to_plotly(self, raw_data: Dict[str, Any], site: str) -> Dict[str, Any]:
        """
        Convert old raw correlation format to Plotly heatmap format.
        
        Args:
            raw_data: Dict with 'matrix' and 'columns' keys
            site: Site name for title
            
        Returns:
            Plotly-ready dict with 'data' and 'layout' keys
        """
        import pandas as pd
        
        columns = raw_data['columns']
        matrix_dict = raw_data['matrix']
        
        # Reconstruct correlation matrix from dict format
        corr_matrix = pd.DataFrame(matrix_dict)[columns].reindex(columns)
        
        title = f'Correlation Heatmap - {site}'
        
        # Build annotations for heatmap values
        annotations = []
        for i in range(len(columns)):
            for j in range(len(columns)):
                value = corr_matrix.iloc[i, j]
                color = "white" if abs(value) > 0.7 else "black"
                annotations.append({
                    "x": columns[j],
                    "y": columns[i],
                    "text": f"{value:.2f}",
                    "font": {"color": color, "size": 10},
                    "showarrow": False
                })
        
        return {
            "data": [{
                "type": "heatmap",
                "z": corr_matrix.values.tolist(),
                "x": columns,
                "y": columns,
                "colorscale": [
                    [0.0, "rgb(178, 24, 43)"],
                    [0.25, "rgb(239, 138, 98)"],
                    [0.5, "rgb(255, 255, 255)"],
                    [0.75, "rgb(103, 169, 207)"],
                    [1.0, "rgb(33, 102, 172)"]
                ],
                "zmid": 0,
                "zmin": -1,
                "zmax": 1,
                "colorbar": {
                    "title": "Correlation (r)",
                    "titleside": "right",
                    "tickmode": "linear",
                    "tick0": -1,
                    "dtick": 0.5
                }
            }],
            "layout": {
                "title": {
                    "text": title,
                    "font": {"size": 20}
                },
                "height": 600,
                "width": 800,
                "xaxis": {
                    "side": "bottom",
                    "tickangle": -45,
                    "tickfont": {"size": 12}
                },
                "yaxis": {
                    "side": "left",
                    "tickfont": {"size": 12}
                },
                "annotations": annotations,
                "margin": {"l": 150, "r": 150, "t": 100, "b": 150}
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Get cache health status."""
        status = {
            "file_cache_enabled": self.enabled,
            "file_cache_path": str(self.cache_dir),
            "file_cache_exists": self.cache_dir.exists(),
        }
        
        if self.use_redis:
            status["redis"] = self._redis_cache.health_check()
        else:
            status["redis"] = {"status": "disabled"}
        
        return status

# Global cache manager instance
cache_manager = CacheManager()