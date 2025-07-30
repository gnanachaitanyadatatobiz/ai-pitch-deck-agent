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
        db = VectorDatabase()
        # The tool now performs a general search, not a targeted company lookup
        context = db.search_by_query(query)
        if not context:
            return "No relevant context found for the query."
        return context

class KnowledgeAgent:
    def __init__(self):
        self.llm = None  # LLM is set by the Streamlit app
        self.agent = Agent(
            role="Startup Ecosystem Analyst",
            goal="Provide foundational context and risk analysis by finding comparable companies in a vector database",
            backstory="You are an expert analyst with a deep understanding of the startup ecosystem. You specialize in using a knowledge base of past startups to identify patterns, risks, and opportunities for new ventures. You are meticulous in citing your sources.",
            tools=[GetCompanyDataTool()],
            allow_delegation=False,
            verbose=True,
        )
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
        industry = startup_data.get("industry_type", "")
        problem = startup_data.get("key_problem_solved", "")
        query = f"A startup in the {industry} space solving the problem of {problem}"
        knowledge_crew = Crew(
            agents=[self.agent],
            tasks=[self.knowledge_task],
            process=Process.sequential,
            verbose=True,
            full_output=True
        )
        result = knowledge_crew.kickoff(inputs={"input": query})
        return str(result)
