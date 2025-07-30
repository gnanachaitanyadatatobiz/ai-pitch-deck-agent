"""
Knowledge Agent Module
Implements CrewAI agent that uses RAG technique to compare research output with existing pitch deck data.
"""

import os
import logging

# Try to import CrewAI components with fallback
try:
    from crewai import Agent, Task, Crew, Process
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    # Create fallback classes
    class Agent:
        def __init__(self, role="", goal="", backstory="", tools=None, allow_delegation=False, verbose=True):
            self.role = role
            self.goal = goal
            self.backstory = backstory
            self.tools = tools or []
            self.allow_delegation = allow_delegation
            self.verbose = verbose

    class Task:
        def __init__(self, description="", expected_output="", agent=None):
            self.description = description
            self.expected_output = expected_output
            self.agent = agent

    class Crew:
        def __init__(self, agents=None, tasks=None, process=None, verbose=True):
            self.agents = agents or []
            self.tasks = tasks or []
            self.process = process
            self.verbose = verbose

        def kickoff(self, inputs=None):
            # Simple fallback implementation
            return "Knowledge analysis completed using fallback mode."
# BaseTool import - try multiple paths
try:
    from crewai_tools import BaseTool
except ImportError:
    try:
        from langchain.tools import BaseTool
    except ImportError:
        # Create a simple BaseTool fallback
        class BaseTool:
            def __init__(self, name="", description=""):
                self.name = name
                self.description = description
from pydantic import BaseModel, Field
from vector_database import VectorDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GetCompanyDataTool(BaseTool):
    name: str = "Get Similar Company Context"
    description: str = "Retrieves and summarizes context for companies similar to the user's startup based on industry and problem description."

    def _run(self, query: str) -> str:
        try:
            db = VectorDatabase()
            # The tool now performs a general search, not a targeted company lookup
            context = db.search_by_query(query)
            if not context:
                return "No relevant context found for the query."
            return context
        except Exception as e:
            return f"Error retrieving context: {str(e)}"

    def run(self, query: str) -> str:
        """Public run method for compatibility."""
        return self._run(query)

class KnowledgeAgent:
    def __init__(self):
        self.llm = None  # LLM is set by the Streamlit app

        # Create agent without tools to avoid validation issues
        # We'll handle the tool functionality manually in the analyze_startup method
        try:
            self.agent = Agent(
                role="Startup Ecosystem Analyst",
                goal="Provide foundational context and risk analysis by finding comparable companies in a vector database",
                backstory="You are an expert analyst with a deep understanding of the startup ecosystem. You specialize in using a knowledge base of past startups to identify patterns, risks, and opportunities for new ventures. You are meticulous in citing your sources.",
                tools=[],  # No tools to avoid validation issues
                allow_delegation=False,
                verbose=True,
            )
            logger.info("✅ KnowledgeAgent created successfully without tools")
        except Exception as e:
            logger.error(f"Failed to create Agent: {e}")
            # Use fallback agent
            self.agent = None
        self.knowledge_task = Task(
            description="""
            Your primary goal is to find relevant context for a new startup from the knowledge base.
            The user will provide their startup details. Your task is to formulate a search query based on their
            industry and the problem they are solving. Use this query with the 'Get Similar Company Context' tool.

            Your final output MUST contain two distinct, clearly marked parts:
            1.  A brief summary of the context you found and why it's relevant.
            2.  The full, raw, verbatim context you retrieved from the tool, enclosed between the exact markers
                '--- RAW CONTEXT START ---' and '--- RAW CONTEXT END ---'.

            Here is an example of the query you should generate for the tool:
            'A startup in the [industry] space solving the problem of [problem description]'
            """,
            expected_output="""
            A brief analysis followed by the full raw context. Example:

            Based on the startup's focus on AI-powered language learning, I have retrieved context from similar companies in the EdTech space. This data provides insights into their product descriptions and market positioning.

            --- RAW CONTEXT START ---
            [Full text of document 1 snippet]
            ---
            [Full text of document 2 snippet]
            --- RAW CONTEXT END ---
            """,
            agent=self.agent
        )

    def quick_company_check(self, company_name: str) -> dict:
        """
        Checks if a company exists in the vector database and returns chunk count.
        """
        db = VectorDatabase()
        exists = db.check_company_exists(company_name)
        chunk_count = db.get_company_document_count(company_name)
        return {"exists": exists, "chunk_count": chunk_count}

    def analyze_startup(self, startup_data: dict, research_output: str = "") -> str:
        """
        Runs the knowledge agent's analysis task using the provided startup data.
        Optionally, can use research_output for context (not used in current prompt).
        """
        try:
            industry = startup_data.get("industry_type", "")
            problem = startup_data.get("key_problem_solved", "")
            query = f"A startup in the {industry} space solving the problem of {problem}"

            # If agent is None, use fallback analysis
            if self.agent is None:
                return self._fallback_analysis(startup_data, query)

            # Since we removed tools, we'll do the database search manually
            try:
                db = VectorDatabase()
                context = db.search_by_query(query)

                # Create a simple analysis based on the context
                if context and len(context.strip()) > 50:
                    analysis = f"""
Based on the startup's focus on {industry} and solving {problem}, I found relevant context from similar companies.

--- RAW CONTEXT START ---
{context}
--- RAW CONTEXT END ---

This context provides insights into similar companies and their approaches in this space.
"""
                else:
                    analysis = f"""
Based on the startup's focus on {industry} and solving {problem}, no specific context was found in the knowledge base.

--- RAW CONTEXT START ---
No relevant context found for this specific industry and problem combination.
--- RAW CONTEXT END ---

This appears to be a novel approach or underrepresented area in our database.
"""

                return analysis

            except Exception as e:
                logger.error(f"Error in manual database search: {e}")
                return self._fallback_analysis(startup_data, query)

        except Exception as e:
            logger.error(f"Error in analyze_startup: {e}")
            return self._fallback_analysis(startup_data, "")

    def _fallback_analysis(self, startup_data, query):
        """Fallback analysis when database or agent fails."""
        industry = startup_data.get("industry_type", "Unknown")
        problem = startup_data.get("key_problem_solved", "Unknown")

        return f"""
Knowledge analysis completed in fallback mode for {industry} startup solving {problem}.

--- RAW CONTEXT START ---
Fallback analysis: Unable to access knowledge database. This startup operates in the {industry} space
and focuses on solving {problem}. Further research would be needed to identify comparable companies.
--- RAW CONTEXT END ---

Analysis completed using fallback methodology.
"""

    def quick_company_check(self, company_name: str) -> dict:
        """
        Quick check to see if a company exists in the knowledge base.
        Returns dictionary with 'exists' key and additional info.
        """
        try:
            db = VectorDatabase()
            # Search for the company name directly
            context = db.search_by_query(company_name)
            # If we get meaningful context back, the company likely exists
            exists = bool(context and len(context.strip()) > 50)

            return {
                'exists': exists,
                'context_length': len(context) if context else 0,
                'company_name': company_name
            }
        except Exception as e:
            logger.error(f"Error in quick_company_check: {e}")
            return {
                'exists': False,
                'context_length': 0,
                'company_name': company_name,
                'error': str(e)
            }
