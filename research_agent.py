"""
Research Agent Module
Implements the Market & Competitor Research agent using CrewAI.
"""

import os
import logging
import requests
import json
# Try to import CrewAI components with fallback
try:
    from crewai import Agent, Task, Crew, Process, LLM
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    # Create fallback classes
    class Agent:
        def __init__(self, role="", goal="", backstory="", tools=None, allow_delegation=False, verbose=True, llm=None):
            self.role = role
            self.goal = goal
            self.backstory = backstory
            self.tools = tools or []
            self.allow_delegation = allow_delegation
            self.verbose = verbose
            self.llm = llm

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
            return "Research completed using fallback mode."

    class LLM:
        def __init__(self, model="", api_key="", temperature=0.7, max_tokens=2000):
            self.model = model
            self.api_key = api_key
            self.temperature = temperature
            self.max_tokens = max_tokens
from dotenv import load_dotenv

# Try to import SerperDevTool with fallback
try:
    from crewai_tools import SerperDevTool
    print("✅ SerperDevTool imported from crewai_tools")
except ImportError:
    try:
        from langchain_community.tools import SerperDevTool
        print("✅ SerperDevTool imported from langchain_community.tools")
    except ImportError:
        print("❌ SerperDevTool not found - using fallback implementation")

        class SerperDevTool:
            """Fallback SerperDevTool implementation using direct Serper API calls."""

            def __init__(self):
                self.api_key = os.getenv('SERPER_API_KEY')
                if not self.api_key:
                    raise ValueError("SERPER_API_KEY not found in environment variables")

            def run(self, query):
                """Run a search query using Serper API."""
                try:
                    url = "https://google.serper.dev/search"
                    headers = {
                        'X-API-KEY': self.api_key,
                        'Content-Type': 'application/json'
                    }
                    payload = {
                        'q': query,
                        'num': 10
                    }

                    response = requests.post(url, headers=headers, json=payload)
                    response.raise_for_status()

                    data = response.json()

                    # Format results similar to original SerperDevTool
                    results = []
                    if 'organic' in data:
                        for item in data['organic'][:5]:  # Top 5 results
                            results.append({
                                'title': item.get('title', ''),
                                'link': item.get('link', ''),
                                'snippet': item.get('snippet', '')
                            })

                    return {
                        'results': results,
                        'query': query,
                        'total_results': len(results)
                    }

                except Exception as e:
                    logger.error(f"❌ Serper API call failed: {e}")
                    return {
                        'results': [],
                        'query': query,
                        'error': str(e)
                    }

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchAgent:
    """
    A CrewAI agent responsible for conducting market and competitor research.
    """
    def __init__(self):
        """Initialize the Research Agent with tools and LLM."""
        
        # Configure LLM
        self.llm = LLM(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7,
            max_tokens=2000
        )

        # Initialize tools
        self.search_tool = SerperDevTool()

        # Initialize the agent
        self.agent = Agent(
            role="Market & Competitor Researcher",
            goal="Conduct comprehensive and up-to-date research on the market and competitors for a given startup.",
            backstory="""You are a highly skilled market analyst with a knack for digging up the latest trends, competitor strategies,
            and market opportunities. You use web search tools to find the most relevant and current information.""",
            tools=[self.search_tool],
            allow_delegation=False,
            verbose=True,
            llm=self.llm
        )

    def run_research_task(self, company_name: str, industry_type: str, knowledge_context: str) -> str:
        """
        Runs the research task for the given startup.

        Args:
            company_name: The name of the startup.
            industry_type: The industry of the startup.
            knowledge_context: The context provided by the Knowledge Agent.

        Returns:
            A detailed market research report.
        """
        try:
            task_description = (
                f"Conduct comprehensive market and competitor research for '{company_name}' "
                f"in the {industry_type} industry. "
                f"Use the following initial analysis from our internal database as a starting point:\n\n"
                f"--- Internal Analysis ---\n{knowledge_context}\n---\n\n"
                f"Your research should focus on:\n"
                f"1. Current market size, trends, and growth projections.\n"
                f"2. Detailed analysis of key competitors, including their strengths, weaknesses, and recent activities.\n"
                f"3. Identification of untapped opportunities or emerging threats in the market.\n"
                f"4. Customer acquisition strategies that are currently effective in this industry."
            )

            research_task = Task(
                description=task_description,
                expected_output="A detailed market research report with actionable insights, competitor analysis, and market trends.",
                agent=self.agent
            )

            # A temporary crew to run just this one task
            research_crew = Crew(
                agents=[self.agent],
                tasks=[research_task],
                process=Process.sequential,
                verbose=True
            )

            logger.info(f"🔬 Starting research task for {company_name}...")
            result = research_crew.kickoff()
            logger.info(f"✅ Research task for {company_name} completed.")
            
            return str(result)

        except Exception as e:
            logger.error(f"Error during research task for {company_name}: {e}")
            return f"Research failed for {company_name} due to an error: {str(e)}" 