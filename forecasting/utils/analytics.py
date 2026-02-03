"""
DuckDB Analytics Utilities
==========================

High-performance analytical queries using DuckDB for 10-100x faster
aggregations compared to Pandas on large datasets.
"""

import os
from typing import Optional, List, Dict, Any
import pandas as pd

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    duckdb = None
    DUCKDB_AVAILABLE = False


class AnalyticsEngine:
    """
    Fast analytical query engine using DuckDB.
    
    DuckDB is 10-100x faster than Pandas for:
    - Aggregations (GROUP BY, COUNT, AVG, etc.)
    - Window functions
    - Joins on large tables
    - Parquet file queries
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize analytics engine.
        
        Args:
            data_path: Optional path to Parquet data file for direct queries
        """
        self.data_path = data_path
        self._conn = None
        
        if not DUCKDB_AVAILABLE:
            raise ImportError("DuckDB not installed. Install with: pip install duckdb")
    
    @property
    def conn(self):
        """Lazy connection initialization."""
        if self._conn is None:
            self._conn = duckdb.connect(database=':memory:')
        return self._conn
    
    def close(self):
        """Close the connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
    
    def query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame."""
        return self.conn.execute(sql).fetchdf()
    
    def query_parquet(self, sql: str, parquet_path: Optional[str] = None) -> pd.DataFrame:
        """
        Query a Parquet file directly without loading into memory.
        
        Args:
            sql: SQL query with 'data' as the table alias
            parquet_path: Path to Parquet file (uses default if not specified)
        
        Example:
            engine.query_parquet("SELECT site, AVG(da) FROM data GROUP BY site")
        """
        path = parquet_path or self.data_path
        if not path:
            raise ValueError("No parquet path specified")
        
        # Replace 'data' with parquet_scan
        full_sql = sql.replace('data', f"parquet_scan('{path}')")
        return self.query(full_sql)
    
    def get_site_statistics(self, parquet_path: Optional[str] = None) -> pd.DataFrame:
        """
        Get comprehensive statistics for each site.
        Much faster than Pandas groupby for large datasets.
        """
        path = parquet_path or self.data_path
        
        return self.query(f"""
            SELECT 
                site,
                COUNT(*) as n_records,
                MIN(date) as first_date,
                MAX(date) as last_date,
                AVG(da) as mean_da,
                MEDIAN(da) as median_da,
                STDDEV(da) as std_da,
                MAX(da) as max_da,
                COUNT(CASE WHEN da > 20 THEN 1 END) as spike_count,
                COUNT(CASE WHEN da > 20 THEN 1 END) * 100.0 / COUNT(*) as spike_percentage
            FROM parquet_scan('{path}')
            GROUP BY site
            ORDER BY site
        """)
    
    def get_temporal_aggregates(
        self, 
        parquet_path: Optional[str] = None,
        group_by: str = 'month'
    ) -> pd.DataFrame:
        """
        Get temporal aggregates (monthly, quarterly, yearly).
        
        Args:
            parquet_path: Path to data file
            group_by: 'month', 'quarter', or 'year'
        """
        path = parquet_path or self.data_path
        
        if group_by == 'month':
            date_expr = "DATE_TRUNC('month', date)"
        elif group_by == 'quarter':
            date_expr = "DATE_TRUNC('quarter', date)"
        else:
            date_expr = "DATE_TRUNC('year', date)"
        
        return self.query(f"""
            SELECT 
                {date_expr} as period,
                site,
                COUNT(*) as n_samples,
                AVG(da) as mean_da,
                MAX(da) as max_da,
                AVG("modis-sst") as mean_sst,
                AVG("modis-chla") as mean_chla
            FROM parquet_scan('{path}')
            GROUP BY {date_expr}, site
            ORDER BY period, site
        """)
    
    def get_correlation_matrix(
        self, 
        parquet_path: Optional[str] = None,
        site: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix using DuckDB (faster for large datasets).
        """
        path = parquet_path or self.data_path
        
        where_clause = f"WHERE site = '{site}'" if site else ""
        
        # Get numeric columns
        df = self.query(f"""
            SELECT * 
            FROM parquet_scan('{path}')
            {where_clause}
            LIMIT 1
        """)
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ['lat', 'lon']]
        
        # For correlation, we need to load data into memory
        # DuckDB doesn't have built-in correlation matrix
        data = self.query(f"""
            SELECT {', '.join([f'"{c}"' for c in numeric_cols])}
            FROM parquet_scan('{path}')
            {where_clause}
        """)
        
        return data.corr()
    
    def get_spike_analysis(
        self, 
        parquet_path: Optional[str] = None,
        threshold: float = 20.0
    ) -> pd.DataFrame:
        """
        Analyze DA spike events efficiently.
        """
        path = parquet_path or self.data_path
        
        return self.query(f"""
            WITH spike_events AS (
                SELECT 
                    date,
                    site,
                    da,
                    "modis-sst" as sst,
                    "modis-chla" as chla,
                    LAG(da) OVER (PARTITION BY site ORDER BY date) as prev_da,
                    LEAD(da) OVER (PARTITION BY site ORDER BY date) as next_da
                FROM parquet_scan('{path}')
                WHERE da > {threshold}
            )
            SELECT 
                site,
                COUNT(*) as spike_count,
                AVG(da) as mean_spike_da,
                MAX(da) as max_spike_da,
                AVG(sst) as mean_spike_sst,
                AVG(chla) as mean_spike_chla,
                AVG(prev_da) as mean_da_before_spike,
                AVG(next_da) as mean_da_after_spike
            FROM spike_events
            GROUP BY site
            ORDER BY spike_count DESC
        """)
    
    def register_dataframe(self, df: pd.DataFrame, name: str):
        """Register a DataFrame for SQL queries."""
        self.conn.register(name, df)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Convenience functions for quick queries
def quick_site_stats(parquet_path: str) -> pd.DataFrame:
    """Get site statistics with a single function call."""
    with AnalyticsEngine(parquet_path) as engine:
        return engine.get_site_statistics()


def quick_temporal_stats(parquet_path: str, group_by: str = 'month') -> pd.DataFrame:
    """Get temporal aggregates with a single function call."""
    with AnalyticsEngine(parquet_path) as engine:
        return engine.get_temporal_aggregates(group_by=group_by)
