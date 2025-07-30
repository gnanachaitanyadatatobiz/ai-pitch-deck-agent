#!/usr/bin/env python3
"""
Test script to verify CrewAI compatibility without ChromaDB
"""

import os
import sys

# Configure environment to avoid ChromaDB issues
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_SERVER_NOFILE"] = "1"
os.environ["ALLOW_RESET"] = "TRUE"

# Comprehensive ChromaDB mock to prevent import issues
import sys
import types

def create_comprehensive_chromadb_mock():
    """Create a comprehensive mock chromadb module with all required submodules."""

    # Main chromadb module
    chromadb = types.ModuleType('chromadb')
    chromadb.Documents = list
    chromadb.EmbeddingFunction = object
    chromadb.Embeddings = list

    # chromadb.api submodule
    chromadb_api = types.ModuleType('chromadb.api')
    chromadb_api.ClientAPI = object
    chromadb_api.AdminAPI = object
    chromadb.api = chromadb_api

    # chromadb.config submodule
    chromadb_config = types.ModuleType('chromadb.config')
    chromadb_config.Settings = dict
    chromadb.config = chromadb_config

    # chromadb.utils submodule
    chromadb_utils = types.ModuleType('chromadb.utils')
    chromadb_utils.embedding_functions = types.ModuleType('chromadb.utils.embedding_functions')
    chromadb.utils = chromadb_utils

    # chromadb.errors submodule
    chromadb_errors = types.ModuleType('chromadb.errors')
    chromadb_errors.ChromaError = Exception
    chromadb.errors = chromadb_errors

    return chromadb

# Pre-emptively add comprehensive mock chromadb to sys.modules
if 'chromadb' not in sys.modules:
    mock_chromadb = create_comprehensive_chromadb_mock()
    sys.modules['chromadb'] = mock_chromadb
    sys.modules['chromadb.api'] = mock_chromadb.api
    sys.modules['chromadb.config'] = mock_chromadb.config
    sys.modules['chromadb.utils'] = mock_chromadb.utils
    sys.modules['chromadb.utils.embedding_functions'] = mock_chromadb.utils.embedding_functions
    sys.modules['chromadb.errors'] = mock_chromadb.errors

def test_crewai_imports():
    """Test that CrewAI can be imported without ChromaDB issues."""
    try:
        print("Testing CrewAI imports...")
        
        # Test basic CrewAI imports
        from crewai import Agent, Task, Crew, Process, LLM
        print("✅ CrewAI core imports successful")
        
        # Test CrewAI tools
        from crewai_tools import SerperDevTool
        print("✅ CrewAI tools imports successful")
        
        # Test basic agent creation
        test_llm = LLM(
            model="gpt-4o-mini",
            api_key="test-key",  # This won't actually be used
            temperature=0.7
        )
        
        test_agent = Agent(
            role="Test Agent",
            goal="Test agent creation",
            backstory="A test agent for compatibility testing",
            llm=test_llm,
            verbose=False
        )
        print("✅ Agent creation successful")
        
        # Test tool creation
        search_tool = SerperDevTool()
        print("✅ SerperDevTool creation successful")
        
        print("\n🎉 All CrewAI compatibility tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ CrewAI compatibility test failed: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_crewai_imports()
    sys.exit(0 if success else 1)
