#!/usr/bin/env python3
"""
SQLite3 setup script for Streamlit Cloud compatibility.
This script ensures pysqlite3 replaces sqlite3 before any other imports.
"""

import sys
import os

def setup_sqlite():
    """Setup SQLite3 compatibility for ChromaDB on Streamlit Cloud."""
    try:
        # Import pysqlite3 and replace sqlite3 completely
        import pysqlite3
        
        # Replace in sys.modules
        sys.modules["sqlite3"] = pysqlite3
        sys.modules["sqlite3.dbapi2"] = pysqlite3
        
        # Set environment variables for ChromaDB
        os.environ["CHROMA_DB_IMPL"] = "duckdb+parquet"
        os.environ["ANONYMIZED_TELEMETRY"] = "False"
        
        print("✅ SQLite3 setup completed successfully")
        
        # Verify the setup
        import sqlite3
        version = sqlite3.sqlite_version_info
        print(f"✅ SQLite version: {'.'.join(map(str, version))}")
        
        if version >= (3, 35, 0):
            print("✅ SQLite version is compatible with ChromaDB")
            return True
        else:
            print(f"⚠️ SQLite version {'.'.join(map(str, version))} may be incompatible")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to setup SQLite3: {e}")
        return False

if __name__ == "__main__":
    setup_sqlite()
