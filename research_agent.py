"""
Research Agent Module
Implements the Market & Competitor Research agent using CrewAI.
"""

import os
import logging
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

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