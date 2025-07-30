#!/usr/bin/env python3
"""
Test script to verify CrewAI compatibility without ChromaDB
"""

import os
import sys

# Configure environment to avoid ChromaDB issues
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_SERVER_NOFILE"] = "1"

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
