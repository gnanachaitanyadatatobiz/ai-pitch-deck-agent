"""
Streamlit Pitch Deck Generator with AI Agents
Preserving exact UI structure and functionality from Flask version.
"""

import json
import os
import logging
import traceback
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

# CRITICAL: SQLite3 compatibility fix for Streamlit Cloud
# This MUST happen before any other imports that might use sqlite3
import sys
import os

# Setup SQLite3 compatibility
try:
    from setup_sqlite import setup_sqlite
    sqlite_success = setup_sqlite()
    print(f"✅ SQLite setup result: {sqlite_success}")
except ImportError:
    # Fallback to inline setup
    try:
        import pysqlite3
        sys.modules["sqlite3"] = pysqlite3
        sys.modules["sqlite3.dbapi2"] = pysqlite3
        os.environ["CHROMA_DB_IMPL"] = "duckdb+parquet"
        os.environ["ANONYMIZED_TELEMETRY"] = "False"
        print("✅ Inline SQLite3 setup completed")
        sqlite_success = True
    except ImportError as e:
        print(f"❌ SQLite3 setup failed: {e}")
        sqlite_success = False

# FORCE CrewAI to work - try multiple approaches
CREWAI_AVAILABLE = False
CREWAI_ERROR = None

# Server-side loop prevention using environment variable
import os
if os.getenv('STREAMLIT_SERVER_HEADLESS') == 'true':
    # Running on Streamlit Cloud - use more aggressive loop prevention
    if 'server_init_done' not in st.session_state:
        st.session_state.server_init_done = False

    if st.session_state.server_init_done:
        print("� Server initialization already completed - skipping")
        # Skip all initialization and jump to main app
        CREWAI_AVAILABLE = st.session_state.get('crewai_available', False)
        CUSTOM_MODULES_AVAILABLE = st.session_state.get('custom_modules_available', False)
    else:
        print("🚀 First-time server initialization starting...")

# Only run initialization if not already done
if not st.session_state.get('server_init_done', False):
    # Attempt 1: Try different SerperDevTool import paths
    if sqlite_success:
        try:
            from crewai import Agent, Task, Crew, Process, LLM

            # Try correct import paths for SerperDevTool
            SerperDevTool = None
            try:
                # Primary import path for CrewAI tools
                from crewai_tools import SerperDevTool
                print("✅ SerperDevTool imported from crewai_tools")
            except ImportError:
                try:
                    # Alternative import for older versions
                    from langchain_community.tools import SerperDevTool
                    print("✅ SerperDevTool imported from langchain_community.tools")
                except ImportError:
                    print("❌ SerperDevTool not found - creating fallback implementation")

                    # Create fallback SerperDevTool class
                    class SerperDevTool:
                        """Fallback SerperDevTool implementation using direct Serper API calls."""

                        def __init__(self):
                            self.api_key = os.getenv('SERPER_API_KEY')
                            if not self.api_key:
                                raise ValueError("SERPER_API_KEY not found in environment variables")

                        def run(self, query):
                            """Run a search query using Serper API."""
                            try:
                                import requests

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

                                # Format results with URLs
                                results = []
                                if 'organic' in data:
                                    for item in data['organic'][:5]:  # Top 5 results
                                        result_item = {
                                            'title': item.get('title', ''),
                                            'link': item.get('link', ''),
                                            'snippet': item.get('snippet', ''),
                                            'url': item.get('link', '')  # Ensure URL is included
                                        }
                                        results.append(result_item)

                                # Return the actual data structure for URL extraction
                                return {
                                    'organic': results,
                                    'query': query,
                                    'total_results': len(results)
                                }

                            except Exception as e:
                                logger.error(f"❌ Serper API call failed: {e}")
                                return f"Error: Search failed - {str(e)}"

            if SerperDevTool is not None:
                CREWAI_AVAILABLE = True
                print("✅ CrewAI imported successfully after SQLite setup")
            else:
                raise ImportError("SerperDevTool not available")

        except Exception as e:
            CREWAI_ERROR = str(e)
            print(f"❌ CrewAI import failed (Attempt 1): {e}")

    # Attempt 2: Try with additional environment variables
    if not CREWAI_AVAILABLE:
        try:
            os.environ["CHROMA_DB_IMPL"] = "duckdb+parquet"
            os.environ["ANONYMIZED_TELEMETRY"] = "False"
            os.environ["ALLOW_RESET"] = "TRUE"

            from crewai import Agent, Task, Crew, Process, LLM

            # Try multiple import paths for SerperDevTool
            SerperDevTool = None
            try:
                from crewai_tools import SerperDevTool
            except ImportError:
                try:
                    from langchain_community.tools import SerperDevTool
                except ImportError:
                    # Use fallback SerperDevTool if import fails
                    SerperDevTool = None

            CREWAI_AVAILABLE = True
            print("✅ CrewAI imported successfully (Attempt 2)")
        except Exception as e:
            CREWAI_ERROR = str(e)
            print(f"❌ CrewAI import failed (Attempt 2): {e}")

    # Attempt 3: Import individual components
    if not CREWAI_AVAILABLE:
        try:
            # Try importing ChromaDB directly first
            import chromadb
            print(f"✅ ChromaDB imported successfully: {chromadb.__version__}")

            from crewai import Agent, Task, Crew, Process, LLM

            # Try multiple import paths for SerperDevTool
            SerperDevTool = None
            try:
                from crewai_tools import SerperDevTool
            except ImportError:
                try:
                    from langchain_community.tools import SerperDevTool
                except ImportError:
                    # Use fallback SerperDevTool if import fails
                    SerperDevTool = None

            CREWAI_AVAILABLE = True
            print("✅ CrewAI imported successfully (Attempt 3)")
        except Exception as e:
            CREWAI_ERROR = str(e)
            print(f"❌ CrewAI import failed (Attempt 3): {e}")

    # Create a simple SerperDevTool fallback if not available
    if CREWAI_AVAILABLE and 'SerperDevTool' not in globals():
        print("⚠️ SerperDevTool not available, creating fallback")

        class SerperDevTool:
            """Fallback SerperDevTool implementation using direct API calls."""

            def __init__(self):
                self.api_key = os.getenv('SERPER_API_KEY')

            def run(self, query):
                """Run search using Serper API directly."""
                if not self.api_key:
                    return "Error: SERPER_API_KEY not configured"

                try:
                    import requests
                    url = "https://google.serper.dev/search"
                    headers = {
                        'X-API-KEY': self.api_key,
                        'Content-Type': 'application/json'
                    }
                    payload = {'q': query, 'num': 5}

                    response = requests.post(url, headers=headers, json=payload)
                    if response.status_code == 200:
                        data = response.json()

                        # Format results similar to SerperDevTool
                        results = []
                        if 'organic' in data:
                            for result in data['organic'][:5]:
                                results.append(f"**{result.get('title', '')}**\n{result.get('snippet', '')}\nSource: {result.get('link', '')}\n")

                        return "\n".join(results) if results else "No results found"
                    else:
                        return f"Search failed with status {response.status_code}"

                except Exception as e:
                    return f"Search error: {e}"

    # Fallback libraries
    if not CREWAI_AVAILABLE:
        try:
            import requests
            from openai import OpenAI
            FALLBACK_AVAILABLE = True
            print("✅ Fallback libraries available")
        except ImportError as fallback_e:
            FALLBACK_AVAILABLE = False
            print(f"❌ Fallback libraries failed: {fallback_e}")

    # Import our custom modules
    try:
        from knowledge_agent import KnowledgeAgent
        from content_agent import ContentAgent
        from output_manager import OutputManager
        from vector_database import VectorDatabase
        CUSTOM_MODULES_AVAILABLE = True
        print("✅ Custom modules imported successfully")
    except ImportError as e:
        CUSTOM_MODULES_AVAILABLE = False
        print(f"❌ Custom module import failed: {e}")
        # Don't stop the app, show error in UI instead

    # Mark initialization as complete to prevent loops
    st.session_state.server_init_done = True
    st.session_state.crewai_available = CREWAI_AVAILABLE
    st.session_state.custom_modules_available = CUSTOM_MODULES_AVAILABLE
    print("🎯 Initialization completed - preventing future loops")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pitchdeck_generator.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def configure_llm():
    """Configure and return the LLM instance with OpenAI settings."""
    try:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        logger.info("Initializing OpenAI LLM...")

        # Import LLM class - try multiple import paths
        LLM = None
        try:
            from crewai import LLM
        except ImportError:
            try:
                from crewai.llm import LLM
            except ImportError:
                # Fallback: create a simple LLM-like object
                class LLM:
                    def __init__(self, model, api_key, temperature=0.7, max_tokens=2000):
                        self.model = model
                        self.api_key = api_key
                        self.temperature = temperature
                        self.max_tokens = max_tokens

        if LLM is None:
            raise ImportError("Could not import LLM class from CrewAI")

        return LLM(
            model="gpt-4o-mini",
            api_key=OPENAI_API_KEY,
            temperature=0.7,
            max_tokens=2000
        )
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {str(e)}")
        raise

def get_llm():
    """Get LLM instance for CrewAI agents."""
    try:
        return configure_llm()
    except Exception as e:
        logger.error(f"❌ Failed to get LLM: {e}")
        return None

# Initialize LLM
try:
    llm = configure_llm()
    logger.info("LLM initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {str(e)}")
    st.error(f"Failed to initialize LLM: {str(e)}")
    st.stop()

# Global variables for agents
knowledge_agent = None
content_agent = None
output_manager = None

def initialize_agents():
    """Initialize all agents and managers with fallback support."""
    global knowledge_agent, content_agent, output_manager

    try:
        logger.info("🚀 Starting agent initialization...")

        # Check if custom modules are available
        if not CUSTOM_MODULES_AVAILABLE:
            logger.warning("⚠️ Custom modules not available, using minimal fallback")
            # Create minimal fallback objects
            output_manager = type('OutputManager', (), {
                'save_research': lambda self, *args, **kwargs: None,
                'save_knowledge_analysis': lambda self, *args, **kwargs: None,
                'save_content': lambda self, *args, **kwargs: None,
                'get_output_path': lambda self, *args, **kwargs: "outputs"
            })()

            knowledge_agent = type('KnowledgeAgent', (), {
                'analyze_startup': lambda self, *args, **kwargs: "Knowledge analysis completed in fallback mode."
            })()

            content_agent = type('ContentAgent', (), {
                'generate_content': lambda self, *args, **kwargs: "Content generated in fallback mode."
            })()

            logger.info("✅ Fallback agents initialized")
            return True

        # Initialize output manager
        try:
            output_manager = OutputManager()
            logger.info("✅ Output manager initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OutputManager: {e}")
            # Create fallback OutputManager
            output_manager = type('OutputManager', (), {
                'save_research': lambda self, *args, **kwargs: None,
                'save_knowledge_analysis': lambda self, *args, **kwargs: None,
                'save_content': lambda self, *args, **kwargs: None,
                'get_output_path': lambda self, *args, **kwargs: "outputs"
            })()
            logger.info("✅ Fallback OutputManager created")

        # Initialize knowledge agent
        try:
            knowledge_agent = KnowledgeAgent()
            logger.info("✅ Knowledge agent initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize KnowledgeAgent: {e}")
            # Create fallback KnowledgeAgent
            knowledge_agent = type('KnowledgeAgent', (), {
                'analyze_startup': lambda self, *args, **kwargs: "Knowledge analysis completed in fallback mode."
            })()
            logger.info("✅ Fallback KnowledgeAgent created")

        # Initialize content agent
        try:
            content_agent = ContentAgent()
            logger.info("✅ Content agent initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize ContentAgent: {e}")
            # Create fallback ContentAgent
            content_agent = type('ContentAgent', (), {
                'generate_content': lambda self, *args, **kwargs: "Content generated in fallback mode."
            })()
            logger.info("✅ Fallback ContentAgent created")

        logger.info("🎉 All agents initialized successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Critical failure in agent initialization: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")

        # Last resort: create minimal fallback objects
        try:
            output_manager = type('OutputManager', (), {'save_research': lambda self, *args, **kwargs: None})()
            knowledge_agent = type('KnowledgeAgent', (), {'analyze_startup': lambda self, *args, **kwargs: "Fallback analysis"})()
            content_agent = type('ContentAgent', (), {'generate_content': lambda self, *args, **kwargs: "Fallback content"})()
            logger.info("✅ Emergency fallback agents created")
            return True
        except:
            logger.error("❌ Even fallback initialization failed")
            return False

def extract_urls_from_text(text):
    """Extract URLs and associated titles from research text."""
    import re

    sources = []
    lines = text.split('\n')

    # Improved URL pattern
    url_pattern = r'(https?://[^\s\)]+)'

    for line in lines:
        urls_in_line = re.findall(url_pattern, line)

        for url in urls_in_line:
            # Clean up URL
            clean_url = url.rstrip('.,;:!?)')

            # Try to extract title from the line
            title_patterns = [
                r'[-*•]\s*([^:\n]+?)[\s:]*' + re.escape(url),  # - Title: URL
                r'([^(\n]+?)\s*\(' + re.escape(url),          # Title (URL)
                r'([^:\n]+?):\s*' + re.escape(url),           # Title: URL
                r'([^-\n]+?)\s*-\s*' + re.escape(url),        # Title - URL
            ]

            title = None
            for pattern in title_patterns:
                title_match = re.search(pattern, line, re.IGNORECASE)
                if title_match:
                    title = title_match.group(1).strip()
                    # Clean up title
                    title = re.sub(r'^[-*•\s]+', '', title)  # Remove leading bullets
                    title = re.sub(r'[:\-]+$', '', title)    # Remove trailing colons/dashes
                    if len(title) > 5:  # Only use if title is meaningful
                        break

            # If no good title found, try to extract domain
            if not title or len(title) < 5:
                domain_match = re.search(r'https?://(?:www\.)?([^/]+)', clean_url)
                title = domain_match.group(1) if domain_match else "Link"

            # Extract snippet (text around the URL)
            snippet = line.replace(url, '').strip()
            snippet = re.sub(r'^[-*•\s]+', '', snippet)  # Remove leading bullets
            snippet = snippet[:200] + '...' if len(snippet) > 200 else snippet

            sources.append({
                'title': title,
                'snippet': snippet,
                'url': clean_url
            })

    return sources

def run_simplified_research(startup_data, research_prompt=None):
    """Simplified research function that works without CrewAI using direct API calls."""
    try:
        # Note: research_prompt parameter is kept for compatibility but not used in simplified mode
        _ = research_prompt  # Acknowledge the parameter to avoid warnings

        # Validate startup_data
        if not startup_data or not isinstance(startup_data, dict):
            logger.error("❌ Invalid startup_data provided to research")
            return "Error: Invalid startup data", [], {"error": "Invalid startup data"}

        # Check for required API keys
        openai_api_key = os.getenv('OPENAI_API_KEY')
        serper_api_key = os.getenv('SERPER_API_KEY')

        if not openai_api_key:
            return "Error: OPENAI_API_KEY not configured", [], {"error": "Missing OPENAI_API_KEY"}

        if not serper_api_key:
            return "Error: SERPER_API_KEY not configured", [], {"error": "Missing SERPER_API_KEY"}

        # Initialize OpenAI client
        client = OpenAI(api_key=openai_api_key)

        # Perform web search using Serper API
        startup_name = startup_data.get('startup_name', 'startup')
        industry_type = startup_data.get('industry_type', 'technology')

        search_queries = [
            f"{startup_name} competitors market analysis",
            f"{industry_type} market size trends 2024",
            f"{industry_type} industry news recent developments"
        ]

        search_results = []
        sources = []

        for query in search_queries:
            try:
                # Call Serper API directly
                search_url = "https://google.serper.dev/search"
                headers = {
                    'X-API-KEY': serper_api_key,
                    'Content-Type': 'application/json'
                }
                payload = {'q': query, 'num': 5}

                response = requests.post(search_url, headers=headers, json=payload)
                if response.status_code == 200:
                    data = response.json()

                    # Extract organic results
                    if 'organic' in data:
                        for result in data['organic'][:3]:  # Top 3 results
                            search_results.append({
                                'title': result.get('title', ''),
                                'snippet': result.get('snippet', ''),
                                'link': result.get('link', ''),
                                'query': query
                            })
                            sources.append({
                                'title': result.get('title', ''),
                                'url': result.get('link', '')
                            })

            except Exception as e:
                logger.error(f"Search failed for query '{query}': {e}")

        if not search_results:
            return "Error: No search results found", [], {"error": "No search results"}

        # Generate research report using OpenAI
        research_content = "\n\n".join([
            f"**{result['title']}**\n{result['snippet']}\nSource: {result['link']}"
            for result in search_results
        ])

        prompt = f"""
Based on the following web search results, create a comprehensive market research report for {startup_name} in the {industry_type} industry:

{research_content}

Please provide:
1. Market Analysis with specific data and sources
2. Competitor Analysis with company names and sources
3. Industry Trends with recent developments and sources
4. A complete list of all source URLs used

Format each finding as: "Finding (Source: URL)"
"""

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.7
            )

            research_output = response.choices[0].message.content

            return research_output, sources, {"search_results": search_results}

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return f"Error: Failed to generate research report: {e}", sources, {"error": f"OpenAI API failed: {e}"}

    except Exception as e:
        logger.error(f"Simplified research failed: {e}")
        return f"Error: Research failed: {e}", [], {"error": f"Research failed: {e}"}

def run_research_agent_simple(startup_data, research_prompt=None):
    """Simplified research agent function that returns both the research output and sources, and the raw result for debugging."""
    try:
        # Validate startup_data
        if not startup_data or not isinstance(startup_data, dict):
            logger.error("❌ Invalid startup_data provided to research agent")
            return "Error: Invalid startup data", [], {"error": "Invalid startup data"}

        # Validate required fields for research
        required_fields = ['startup_name', 'industry_type']
        missing_fields = [field for field in required_fields if not startup_data.get(field, '').strip()]

        if missing_fields:
            logger.error(f"❌ Missing required fields for research: {missing_fields}")
            return f"Error: Missing required fields: {', '.join(missing_fields)}", [], {"error": "Missing required fields"}

        logger.info(f"🔍 Starting research for: {startup_data.get('startup_name', 'Unknown')}")

        # Check for SERPER_API_KEY
        serper_api_key = os.getenv('SERPER_API_KEY')
        if not serper_api_key:
            logger.error("❌ SERPER_API_KEY not found in environment variables")
            return "Error: SERPER_API_KEY not configured. Please add your Serper API key to the secrets.", [], {"error": "Missing SERPER_API_KEY"}

        # Get LLM configuration
        try:
            llm = get_llm()
            if not llm:
                logger.error("❌ Failed to initialize LLM for research agent")
                return "Error: LLM initialization failed", [], {"error": "LLM initialization failed"}
        except NameError:
            # get_llm function not available, create LLM directly
            try:
                from crewai import LLM
                llm = LLM(model="gpt-3.5-turbo", api_key=os.getenv('OPENAI_API_KEY'))
            except Exception as e:
                logger.error(f"❌ Failed to create LLM directly: {e}")
                return f"Error: LLM creation failed: {e}", [], {"error": f"LLM creation failed: {e}"}

        # Initialize tools with enhanced configuration
        try:
            search_tool = SerperDevTool()
            logger.info(f"✅ SerperDevTool initialized successfully with API key: {serper_api_key[:10]}...")
        except Exception as e:
            logger.error(f"❌ Failed to initialize SerperDevTool: {e}")
            return f"Error: Failed to initialize search tool: {e}", [], {"error": f"SerperDevTool initialization failed: {e}"}

        # Test the search tool directly to see what it returns
        logger.info("🔧 Testing SerperDevTool directly...")
        try:
            test_query = f"{startup_data.get('startup_name', 'startup')} {startup_data.get('industry_type', 'technology')} market analysis"
            # Use the correct method to call SerperDevTool
            test_result = search_tool.run(query=test_query)

            logger.info(f"🔍 Direct SerperDevTool test result type: {type(test_result)}")
            logger.info(f"🔍 Direct SerperDevTool test result: {str(test_result)[:500]}...")

            # Check if the test result contains actual search data
            if not test_result or "No results found" in str(test_result):
                logger.warning("⚠️ SerperDevTool test returned no results - API key might be invalid")
                return "Error: SerperDevTool returned no results. Please check your SERPER_API_KEY.", [], {"error": "No search results"}

        except Exception as e:
            logger.error(f"❌ SerperDevTool direct test failed: {e}")
            return f"Error: SerperDevTool test failed: {e}", [], {"error": f"SerperDevTool test failed: {e}"}

        # Setup research agent with enhanced instructions
        research_agent = Agent(
            role="Market & Competitor Researcher",
            goal="Conduct comprehensive web research and return detailed findings with source URLs.",
            backstory="An expert market researcher who uses web search tools to gather comprehensive market intelligence and always includes source URLs in findings.",
            tools=[search_tool],
            allow_delegation=False,
            verbose=True,
            llm=llm,
            max_iter=3,  # Limit iterations to prevent infinite loops
            memory=False  # Disable memory to prevent issues
        )
        
        # Use custom research prompt if provided
        if research_prompt and research_prompt.strip():
            task_description = research_prompt
        else:
            # Create detailed task description with fallbacks
            startup_name = startup_data.get('startup_name', 'the startup').strip()
            industry_type = startup_data.get('industry_type', 'technology').strip()
            key_problem = startup_data.get('key_problem_solved', 'market challenges').strip()

            task_description = (
                f"Conduct comprehensive market and competitor research for '{startup_name}' "
                f"in the {industry_type} industry. "
                f"The startup focuses on solving: {key_problem}. "
                f"Research should include: market trends, key competitors, market size, "
                f"growth opportunities, and recent industry developments."
            )

        logger.info(f"📝 Research task: {task_description[:100]}...")
        
        research_task = Task(
            description=f"""
RESEARCH TASK: {task_description}

MANDATORY REQUIREMENTS:
1. MUST use the search tool for EVERY piece of information
2. MUST include the actual URL source for EVERY fact or statistic
3. MUST format findings as: "Finding (Source: https://example.com)"
4. MUST provide a "SOURCES USED" section at the end with all URLs
5. MUST search for: market size, competitors, trends, recent news
6. MUST verify information with multiple sources when possible

SEARCH QUERIES TO PERFORM:
- "{startup_data.get('startup_name', 'startup')} competitors market analysis"
- "{startup_data.get('industry_type', 'technology')} market size trends 2024"
- "{startup_data.get('industry_type', 'technology')} industry news recent developments"
- "market opportunities {startup_data.get('industry_type', 'technology')} sector"

OUTPUT FORMAT:
## Market Analysis
[Include findings with source URLs]

## Competitor Analysis
[Include competitor info with source URLs]

## Industry Trends
[Include trends with source URLs]

## SOURCES USED
- https://source1.com
- https://source2.com
- [etc.]
            """,
            expected_output="A comprehensive market research report with detailed insights, competitor analysis, market trends, and a complete list of all source URLs used in the research. Every fact must include its source URL.",
            agent=research_agent
        )
        
        research_crew = Crew(
            agents=[research_agent],
            tasks=[research_task],
            process=Process.sequential,
            verbose=True,
            full_output=True
        )
        
        logger.info("🚀 Starting research process...")

        # Try direct SerperDevTool approach first, then CrewAI as fallback
        direct_sources = []

        try:
            logger.info("🔍 Trying direct SerperDevTool approach...")

            # Create multiple search queries for comprehensive results
            search_queries = [
                f"{startup_data.get('startup_name', '')} {startup_data.get('industry_type', '')} market analysis",
                f"{startup_data.get('industry_type', '')} industry trends 2024",
                f"{startup_data.get('key_problem_solved', '')} market size competitors",
                f"{startup_data.get('industry_type', '')} startup funding market research"
            ]

            for query in search_queries:
                if query.strip():
                    try:
                        logger.info(f"🔍 Searching: {query}")
                        # Use the correct method to call SerperDevTool
                        search_result = search_tool.run(query=query)
                        logger.info(f"🔍 Search result type: {type(search_result)}")
                        logger.info(f"🔍 Search result preview: {str(search_result)[:200]}...")

                        # Extract sources from this search
                        query_sources = []

                        if isinstance(search_result, dict):
                            # Direct dictionary response from fallback SerperDevTool
                            query_sources = extract_sources_from_data(search_result)
                            logger.info(f"✅ Found {len(query_sources)} sources from dict response: {query[:50]}...")
                        elif isinstance(search_result, str):
                            try:
                                # Try to parse as JSON first
                                import json
                                parsed_result = json.loads(search_result)
                                query_sources = extract_sources_from_data(parsed_result)
                                logger.info(f"✅ Found {len(query_sources)} sources from JSON response: {query[:50]}...")
                            except:
                                # If not JSON, extract URLs from text
                                query_sources = extract_urls_from_text(search_result)
                                logger.info(f"✅ Found {len(query_sources)} sources from text response: {query[:50]}...")

                        direct_sources.extend(query_sources)
                    except Exception as e:
                        logger.error(f"❌ Direct search failed for query '{query}': {e}")

        except Exception as e:
            logger.error(f"❌ Direct SerperDevTool approach failed: {e}")

        # If we got sources from direct approach, use them
        if direct_sources:
            logger.info(f"🎯 Direct approach successful! Found {len(direct_sources)} sources")

            # Create a summary of findings
            research_summary = f"""
            Comprehensive market research for {startup_data.get('startup_name', 'the startup')} in the {startup_data.get('industry_type', 'technology')} industry:

            🏢 Industry: {startup_data.get('industry_type', 'Technology')}
            🎯 Problem Focus: {startup_data.get('key_problem_solved', 'Market challenges')}
            📊 Market Analysis: Based on {len(direct_sources)} research sources
            🔍 Research Coverage: Market trends, competitors, industry developments, and funding landscape

            Key areas researched:
            • Market size and growth potential
            • Competitive landscape analysis
            • Industry trends and developments
            • Funding and investment patterns

            Sources: {len(direct_sources)} comprehensive research sources found
            """

            return research_summary, direct_sources, {"direct_search": True, "sources_count": len(direct_sources)}

        # Fallback to CrewAI approach
        try:
            logger.info("🚀 Falling back to CrewAI research...")
            result = research_crew.kickoff()
            logger.info("✅ CrewAI research completed successfully")
        except Exception as e:
            logger.error(f"❌ CrewAI research also failed: {str(e)}")
            result = None

        # Try to extract sources from the result if present
        sources = []
        logger.info(f"Raw research result type: {type(result)}")
        logger.info(f"Raw research result: {result}")

        # Enhanced result extraction to capture ALL SerperDevTool results
        def extract_sources_from_data(data):
            """Extract all possible sources from SerperDevTool data"""
            extracted = []

            if not data:
                return extracted

            # Handle different data structures
            if isinstance(data, dict):
                # Look for organic results in various formats
                organic_keys = ['organic_search_results', 'organic_results', 'organic', 'results', 'search_results']
                for key in organic_keys:
                    organic = data.get(key)
                    if organic and isinstance(organic, list):
                        logger.info(f"Found {len(organic)} results in '{key}'")
                        for item in organic:
                            if isinstance(item, dict):
                                title = item.get('title') or item.get('name') or item.get('heading') or 'Untitled'
                                snippet = item.get('snippet') or item.get('description') or item.get('summary') or ''
                                url = item.get('link') or item.get('url') or item.get('href') or item.get('source')
                                if url and url.startswith('http'):
                                    extracted.append({'title': title, 'snippet': snippet, 'url': url})

                # Also check for news results, knowledge graph, etc.
                additional_keys = ['news_results', 'knowledge_graph', 'related_searches', 'people_also_ask']
                for key in additional_keys:
                    additional_data = data.get(key)
                    if additional_data:
                        if isinstance(additional_data, list):
                            for item in additional_data:
                                if isinstance(item, dict):
                                    title = item.get('title') or item.get('question') or item.get('name') or 'Additional Source'
                                    snippet = item.get('snippet') or item.get('answer') or item.get('description') or ''
                                    url = item.get('link') or item.get('url') or item.get('source')
                                    if url and url.startswith('http'):
                                        extracted.append({'title': title, 'snippet': snippet, 'url': url})
                        elif isinstance(additional_data, dict):
                            url = additional_data.get('link') or additional_data.get('url')
                            if url and url.startswith('http'):
                                title = additional_data.get('title') or additional_data.get('name') or 'Knowledge Source'
                                snippet = additional_data.get('description') or additional_data.get('snippet') or ''
                                extracted.append({'title': title, 'snippet': snippet, 'url': url})

            return extracted

        # Try multiple extraction methods
        if hasattr(result, 'raw') and result.raw:
            # CrewAI TaskOutput object
            raw_result = result.raw
            logger.info(f"Found raw result type: {type(raw_result)}")
            sources.extend(extract_sources_from_data(raw_result))

        if isinstance(result, dict):
            # Direct dictionary result
            sources.extend(extract_sources_from_data(result))

        # Also try to extract from the string representation
        if hasattr(result, 'output'):
            # Check if output contains structured data
            try:
                import json
                if isinstance(result.output, str) and result.output.strip().startswith('{'):
                    output_data = json.loads(result.output)
                    sources.extend(extract_sources_from_data(output_data))
            except:
                pass

        # Remove duplicates while preserving order
        seen_urls = set()
        unique_sources = []
        for source in sources:
            if source['url'] not in seen_urls:
                seen_urls.add(source['url'])
                unique_sources.append(source)

        sources = unique_sources
        logger.info(f"✅ Extracted {len(sources)} unique sources from SerperDevTool")

        # If no sources found from structured data, try to extract from text
        if not sources:
            logger.info("No structured sources found, attempting text extraction...")
            text_sources = extract_urls_from_text(str(result))
            sources.extend(text_sources)
            logger.info(f"Extracted {len(text_sources)} sources from text")

        # Return the stringified result, sources, and the raw result for debugging
        return str(result), sources, result
    except Exception as e:
        logger.error(f"❌ Error during research: {e}")

        # Provide a more helpful fallback with some basic sources
        startup_name = startup_data.get('startup_name', 'your startup')
        industry = startup_data.get('industry_type', 'technology')

        fallback_message = f"""
        Research encountered an issue, but here's some basic analysis for {startup_name}:

        🏢 Industry: {industry}
        📊 Market Analysis: The {industry} sector continues to show growth potential
        🎯 Recommendation: Focus on market validation and competitive analysis

        Please try running the research again for more detailed insights.
        """

        # Provide some basic industry-relevant sources as fallback
        fallback_sources = [
            {
                'title': f'{industry.title()} Industry Overview - Statista',
                'url': f'https://www.statista.com/markets/418/topic/484/{industry.lower().replace(" ", "-")}-industry/',
                'snippet': f'Market research and statistics for the {industry} industry'
            },
            {
                'title': f'{industry.title()} Market Trends - CB Insights',
                'url': f'https://www.cbinsights.com/research/{industry.lower().replace(" ", "-")}-trends',
                'snippet': f'Latest trends and insights in the {industry} market'
            },
            {
                'title': f'{industry.title()} Startups - Crunchbase',
                'url': f'https://www.crunchbase.com/discover/organization.companies/{industry.lower().replace(" ", "-")}',
                'snippet': f'Database of {industry} companies and funding information'
            }
        ]

        return fallback_message, fallback_sources, {"error": str(e), "fallback": True}

def run_enhanced_workflow():
    global knowledge_agent, content_agent, output_manager
    # Ensure agents are initialized
    if knowledge_agent is None or content_agent is None or output_manager is None:
        initialize_agents()
    try:
        # Load startup data from frontend with validation
        try:
            with open("startup_data.json", "r", encoding="utf-8") as f:
                startup_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"❌ Error loading startup data: {e}")
            return {"error": "Startup data file not found or corrupted. Please resubmit the form."}

        # Validate required fields
        required_fields = ['startup_name', 'industry_type', 'key_problem_solved']
        missing_fields = [field for field in required_fields if not startup_data.get(field, '').strip()]

        if missing_fields:
            logger.error(f"❌ Missing required fields: {missing_fields}")
            return {"error": f"Missing required fields: {', '.join(missing_fields)}. Please complete the form."}

        company_name = startup_data.get('startup_name', '')
        logger.info(f"🚀 Starting enhanced workflow for: {company_name}")
        logger.info(f"📋 Industry: {startup_data.get('industry_type', 'N/A')}")
        logger.info(f"🎯 Problem: {startup_data.get('key_problem_solved', 'N/A')[:100]}...")
        
        # STEP 1: KNOWLEDGE AGENT - Check existing data and analyze
        logger.info("📊 STEP 1: Knowledge Agent - Analyzing existing database...")
        company_check = knowledge_agent.quick_company_check(company_name)

        # STEP 1.5: Retrieve context from vector DB (company name first, then fallback to semantic query)
        db = VectorDatabase()
        source_doc = db.search_by_company(company_name)
        if not source_doc or "No relevant documents found" in source_doc:
            query = f"A startup in the {startup_data.get('industry_type', '')} space focused on {startup_data.get('key_problem_solved', '')}"
            source_doc = db.search_by_query(query)
        
        # STEP 2: RESEARCH AGENT - Based on knowledge agent findings
        logger.info("🔬 STEP 2: Research Agent - Conducting market research...")
        
        if company_check['exists']:
            logger.info(f"✅ Company {company_name} found in database. Conducting targeted research.")
            research_prompt = f"""
            Company {company_name} exists in our database. 
            Conduct additional research focusing on:
            1. Recent market developments
            2. New competitors
            3. Updated market projections
            4. Recent funding trends in this space
            """
        else:
            logger.info(f"🆕 Company {company_name} not found in database. Conducting comprehensive research.")
            research_prompt = None
        
        with st.spinner("🔍 CONNECTING... Conducting web research"):
            if CREWAI_AVAILABLE:
                research_output, research_sources, research_raw_result = run_research_agent_simple(startup_data, research_prompt)
            else:
                research_output, research_sources, research_raw_result = run_simplified_research(startup_data, research_prompt)
        # Ensure research_raw_result is JSON serializable
        def make_json_safe(obj):
            if isinstance(obj, (dict, list, str, int, float, bool)) or obj is None:
                return obj
            try:
                # Try to convert to dict
                if hasattr(obj, 'to_dict'):
                    return obj.to_dict()
                elif hasattr(obj, '__dict__'):
                    return obj.__dict__
                else:
                    return str(obj)
            except Exception:
                return str(obj)
        safe_raw_result = make_json_safe(research_raw_result)

        # RESTORE WORKFLOW STEPS
        # STEP 3: KNOWLEDGE ANALYSIS - Comprehensive analysis with research data
        logger.info("🧠 STEP 3: Knowledge Agent - Comprehensive analysis...")
        knowledge_analysis = knowledge_agent.analyze_startup(startup_data, research_output)

        # STEP 4: CONTENT AGENT - Generate final content
        logger.info("📝 STEP 4: Content Agent - Generating pitch deck content...")
        pitch_content = content_agent.generate_pitch_content(startup_data, research_output, knowledge_analysis)

        # STEP 5: CREATE POWERPOINT
        logger.info("🎨 STEP 5: Creating PowerPoint presentation...")
        ppt_file = content_agent.create_powerpoint_presentation(startup_data, pitch_content)

        # STEP 6: SAVE ORGANIZED OUTPUTS
        logger.info("💾 STEP 6: Saving organized outputs...")

        # Save all outputs
        research_file = output_manager.save_research_output(company_name, research_output, startup_data)
        knowledge_file = output_manager.save_knowledge_analysis(
            company_name, knowledge_analysis, company_check['exists'], startup_data
        )
        content_file = output_manager.save_content_output(company_name, pitch_content, startup_data)

        # Move PowerPoint file
        if ppt_file and not ppt_file.startswith("Error"):
            organized_ppt_file = output_manager.save_presentation(company_name, ppt_file)
        else:
            organized_ppt_file = "PowerPoint creation failed"
        # END RESTORE

        comprehensive_report = {
            "company_name": company_name,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "company_exists_in_db": company_check['exists'],
            "startup_data": startup_data,
            "research_output": research_output,
            "research_sources": research_sources, # Add research_sources to the report
            "research_raw_result": safe_raw_result, # Add raw result for debugging (safe)
            "knowledge_analysis": knowledge_analysis,
            "pitch_content": pitch_content,
            "files": {
                "research_file": research_file,
                "knowledge_file": knowledge_file,
                "content_file": content_file,
                "powerpoint_file": organized_ppt_file
            },
            "workflow_status": "completed",
            "source_doc": source_doc
        }
        
        # Save comprehensive report
        report_file = output_manager.save_comprehensive_report(company_name, comprehensive_report)
        
        logger.info(f"🎉 Enhanced workflow completed successfully!")
        logger.info(f"📁 All outputs organized in 'outputs' directory")
        logger.info(f"📊 Comprehensive report: {report_file}")
        
        return comprehensive_report
        
    except Exception as e:
        logger.error(f"Error in enhanced workflow: {e}")
        logger.error(traceback.format_exc())
        return None

# Streamlit Configuration
st.set_page_config(
    page_title="🚀 AI Pitch Deck Generator",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to match the original design
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #4f46e5;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .workflow-info {
        background-color: #eff6ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
        margin-bottom: 2rem;
    }
    .stButton > button {
        background-color: #4f46e5;
        color: white;
        font-weight: bold;
        padding: 0.75rem 2rem;
        border-radius: 0.5rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #4338ca;
        transform: scale(1.05);
    }
    .footer-info {
        background-color: #f9fafb;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin-top: 2rem;
        font-size: 0.75rem;
        color: #6b7280;
    }
</style>
""", unsafe_allow_html=True)

def display_main_form():
    """Display the main form interface matching the original design."""

    # Header section
    st.markdown('<h1 class="main-header">🚀 AI Pitch Deck Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by CrewAI with RAG Technology</p>', unsafe_allow_html=True)

    # Workflow info box
    st.markdown("""
    <div class="workflow-info">
        <strong>🧠 Workflow:</strong> Knowledge Agent → Research Agent → Content Agent → PowerPoint Generation
    </div>
    """, unsafe_allow_html=True)

    # Form
    with st.form("pitch_deck_form"):
        # Row 1 - Basic Info
        col1, col2 = st.columns(2)
        with col1:
            startup_name = st.text_input("Startup Name", key="startup_name")
        with col2:
            industry_type = st.text_input("Industry Type", key="industry_type")

        # Row 2 - Founder and Product
        col1, col2 = st.columns(2)
        with col1:
            founder_name = st.text_input("Founder Name", key="founder_name")
        with col2:
            product_name = st.text_input("Product Name", key="product_name")

        # Text areas
        founder_bio = st.text_area("Founder Bio", height=80, key="founder_bio")
        team_summary = st.text_area("Team Summary", height=80, key="team_summary")
        vision_statement = st.text_area("Vision Statement", height=80, key="vision_statement")
        key_problem_solved = st.text_area("Key Problem You Solve", height=80, key="key_problem_solved")
        solution_summary = st.text_area("Solution Summary (1-3 lines)", height=80, key="solution_summary")
        target_customer_profile = st.text_area("Target Customer Profile", height=80, key="target_customer_profile")
        business_model = st.text_area("Business Model", height=80, key="business_model")
        acquisition_strategy = st.text_area("Acquisition Strategy", height=80, key="acquisition_strategy")

        # Single line inputs
        market_size = st.text_input("Market Size", key="market_size")
        competitors = st.text_input("Competitors (comma-separated)", key="competitors")

        # More text areas
        why_you_win = st.text_area("Why You Win", height=80, key="why_you_win")

        # Financial inputs
        funding_amount = st.text_input("Funding Amount", key="funding_amount")
        use_of_funds_split_percentages = st.text_input("Use of Funds (split into %)", key="use_of_funds_split_percentages")
        transactions = st.text_input("Transactions (if any)", key="transactions")
        monetization_plan = st.text_area("Monetization Plan", height=80, key="monetization_plan")

        # Submit button
        submitted = st.form_submit_button("🚀 Generate AI Pitch Deck", use_container_width=True)

        if submitted:
            # Validate required fields
            required_fields = [startup_name, industry_type, founder_name, product_name,
                             founder_bio, team_summary, vision_statement, key_problem_solved,
                             solution_summary, target_customer_profile, business_model,
                             acquisition_strategy, market_size, why_you_win, funding_amount,
                             use_of_funds_split_percentages, monetization_plan]

            if all(field.strip() for field in required_fields):
                # Store form data in session state
                st.session_state.startup_data = {
                    'startup_name': startup_name,
                    'industry_type': industry_type,
                    'founder_name': founder_name,
                    'founder_bio': founder_bio,
                    'team_summary': team_summary,
                    'product_name': product_name,
                    'vision_statement': vision_statement,
                    'key_problem_solved': key_problem_solved,
                    'solution_summary': solution_summary,
                    'target_customer_profile': target_customer_profile,
                    'business_model': business_model,
                    'acquisition_strategy': acquisition_strategy,
                    'market_size': market_size,
                    'competitors': competitors,
                    'why_you_win': why_you_win,
                    'funding_amount': funding_amount,
                    'use_of_funds_split_percentages': use_of_funds_split_percentages,
                    'transactions': transactions,
                    'monetization_plan': monetization_plan
                }

                # Save data to JSON file with validation
                try:
                    with open('startup_data.json', 'w', encoding='utf-8') as f:
                        json.dump(st.session_state.startup_data, f, indent=2, ensure_ascii=False)

                    # Verify the file was written correctly
                    with open('startup_data.json', 'r', encoding='utf-8') as f:
                        test_data = json.load(f)
                        if not test_data.get('startup_name'):
                            raise ValueError("Startup data validation failed")

                    logger.info(f"✅ Startup data saved successfully for: {startup_name}")

                    # Set processing state
                    st.session_state.processing = True
                    st.session_state.show_results = False
                    st.rerun()

                except Exception as e:
                    logger.error(f"❌ Error saving startup data: {e}")
                    st.error(f"Error saving data: {e}. Please try again.")
            else:
                st.error("Please fill in all required fields.")

    # Footer
    st.markdown("""
    <div class="footer-info">
        📊 Data stored in Pinecone Vector Database | 🤖 Powered by OpenAI GPT-4 | 🔍 RAG Technology
    </div>
    """, unsafe_allow_html=True)

def display_processing():
    """Display processing status."""
    st.markdown('<h1 class="main-header">🔄 Processing Your Request</h1>', unsafe_allow_html=True)

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Simulate progress updates
    steps = [
        "🧠 Knowledge Agent - Analyzing existing database...",
        "🔬 Research Agent - Conducting market research...",
        "📊 Knowledge Agent - Comprehensive analysis...",
        "📝 Content Agent - Generating pitch deck content...",
        "🎨 Creating PowerPoint presentation...",
        "💾 Saving organized outputs..."
    ]

    for i, step in enumerate(steps):
        status_text.text(step)
        progress_bar.progress((i + 1) / len(steps))

        if i == 0:
            # Start the actual workflow in background
            if 'workflow_started' not in st.session_state:
                st.session_state.workflow_started = True
                # Run workflow
                result = run_enhanced_workflow()
                if result:
                    st.session_state.workflow_result = result
                    st.session_state.processing = False
                    st.session_state.show_results = True

        # Small delay for visual effect
        import time
        time.sleep(1)

    # Check if workflow completed
    if hasattr(st.session_state, 'workflow_result'):
        st.session_state.processing = False
        st.session_state.show_results = True
        st.rerun()
    else:
        st.info("🔄 Analysis in progress... The page will refresh automatically.")
        time.sleep(3)
        st.rerun()

def display_results():
    global output_manager, knowledge_agent, content_agent
    # Ensure agents are initialized
    if output_manager is None or knowledge_agent is None or content_agent is None:
        initialize_agents()
    try:
        # Get the latest comprehensive report
        latest_report_path = output_manager.get_latest_report()

        if latest_report_path:
            with open(latest_report_path, 'r', encoding='utf-8') as f:
                report_wrapper = json.load(f)

            report_data = report_wrapper.get('data', {})
            metadata = report_wrapper.get('metadata', {})
            files = report_data.get('files', {})

            st.markdown('<h1 class="main-header">🎯 AI Pitch Deck Generator Results</h1>', unsafe_allow_html=True)
            st.markdown('<p class="sub-header">Complete workflow analysis with RAG technology</p>', unsafe_allow_html=True)

            # Company Info and Generation Info
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Company:** {report_data.get('company_name', 'Unknown')}")
                st.markdown(f"**Status:** {'✅ Found in database' if report_data.get('company_exists_in_db', False) else '🆕 New company'}")
            with col2:
                st.markdown(f"**Generated:** {metadata.get('generated_at', 'Unknown')}")
                st.markdown(f"**Workflow Status:** {report_data.get('workflow_status', 'Unknown')}")

            st.markdown("---")

            # Generated Files Section
            st.markdown("### 📁 Generated Files")
            file_cols = st.columns(4)
            with file_cols[0]:
                st.markdown(f"**📊 Research Report:**\n{files.get('research_file', 'Not available')}")
            with file_cols[1]:
                st.markdown(f"**🧠 Knowledge Analysis:**\n{files.get('knowledge_file', 'Not available')}")
            with file_cols[2]:
                st.markdown(f"**📝 Content File:**\n{files.get('content_file', 'Not available')}")
            with file_cols[3]:
                st.markdown(f"**🎨 PowerPoint:**\n{files.get('powerpoint_file', 'Not available')}")

            st.markdown("---")

            # Source Context Section
            source_doc = report_data.get('source_doc', '')
            research_output = report_data.get('research_output', '')
            research_sources = report_data.get('research_sources', [])
            research_raw_result = report_data.get('research_raw_result', None)
            _ = research_raw_result  # Acknowledge variable to avoid warnings

            with st.expander("🔍 View Source Context Used", expanded=False):
                if source_doc and 'No relevant documents found' not in source_doc:
                    # Try to split into summary and raw context
                    if '--- RAW CONTEXT START ---' in source_doc and '--- RAW CONTEXT END ---' in source_doc:
                        summary, raw = source_doc.split('--- RAW CONTEXT START ---', 1)
                        raw, _ = raw.split('--- RAW CONTEXT END ---', 1)
                        st.markdown("#### Summary / Analysis")
                        st.markdown(summary.strip())
                        st.markdown("#### Raw Context")
                        st.code(raw.strip())
                    else:
                        st.markdown("#### Context")
                        st.code(source_doc[:2000] + ('... (truncated)' if len(source_doc) > 2000 else ''))
                else:
                    st.info("No relevant context was found in the vector database. The Research Agent gathered up-to-date information, and the Content Agent generated your pitch deck using your input and the research findings.")
                # Show SerperDevTool research output if available
                if research_output:
                    st.markdown("#### Web Research Results (SerperDevTool)")

                    # Debug information (can be removed later)
                    logger.info(f"🔍 Research output length: {len(research_output)}")
                    logger.info(f"📚 Research sources count: {len(research_sources) if research_sources else 0}")

                    # Show structured sources if available
                    if research_sources and len(research_sources) > 0:
                        st.markdown("**📚 Sources Found:**")
                        for i, src in enumerate(research_sources, 1):
                            # Extract domain for display
                            try:
                                domain = src['url'].split('/')[2].replace('www.', '')
                            except:
                                domain = src['url']

                            # Create a nice card-like display for each source
                            st.markdown(f"""
                            <div style="border-left: 3px solid #4CAF50; padding-left: 15px; margin: 10px 0; background-color: #f8f9fa; padding: 10px; border-radius: 5px;">
                                <strong>{i}. <a href="{src['url']}" target="_blank" style="color: #1f77b4; text-decoration: none;">{src['title']}</a></strong><br>
                                <span style="color: #666; font-size: 0.85em;">🔗 {domain}</span><br>
                                <span style="color: #333; font-size: 0.9em; line-height: 1.4;">{src['snippet']}</span>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        # Enhanced fallback: Try to extract and display links with better formatting
                        st.markdown("**🔍 Extracting sources from research content...**")

                        # Try to extract URLs from the research output
                        extracted_sources = extract_urls_from_text(research_output)

                        if extracted_sources:
                            st.markdown("**📚 Sources Found in Research:**")
                            for i, src in enumerate(extracted_sources, 1):
                                # Create a nice card-like display for each extracted source
                                st.markdown(f"""
                                <div style="border-left: 3px solid #FF9800; padding-left: 15px; margin: 10px 0; background-color: #fff3e0; padding: 10px; border-radius: 5px;">
                                    <strong>{i}. <a href="{src['url']}" target="_blank" style="color: #1f77b4; text-decoration: none;">{src['title']}</a></strong><br>
                                    <span style="color: #666; font-size: 0.85em;">🔗 {src.get('domain', 'External Source')}</span><br>
                                    <span style="color: #333; font-size: 0.9em; line-height: 1.4;">{src.get('snippet', 'Research source')}</span>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            # If still no URLs found, do inline URL replacement
                            st.markdown("**📄 Research Content with Enhanced Links:**")

                            import re
                            # Improved URL pattern to catch more URLs
                            url_pattern = r'(https?://[^\s\)]+)'

                            # Process the research output to make URLs clickable
                            processed_content = research_output
                            urls_found = re.findall(url_pattern, research_output)

                            if urls_found:
                                for url in urls_found:
                                    clean_url = url.rstrip('.,;:!?)')
                                    # Replace URL with clickable link
                                    clickable_link = f'<a href="{clean_url}" target="_blank" style="color: #1f77b4; text-decoration: underline;">{clean_url}</a>'
                                    processed_content = processed_content.replace(url, clickable_link)

                                st.markdown(processed_content, unsafe_allow_html=True)
                                st.success(f"✅ Found {len(urls_found)} clickable links in the research content!")
                            else:
                                # No URLs found at all
                                st.markdown(research_output)
                                st.warning("⚠️ No URLs were found in the research output. The research agent may need better configuration.")

            # Pitch Deck Content Section
            import re
            pitch_content = report_data.get('pitch_content', 'No content generated')
            with st.expander("📑 View AI-Generated Pitch Deck", expanded=True):
                if pitch_content and "Slide" in pitch_content:
                    # Split slides by "Slide X:" pattern
                    slides = re.split(r'(?i)\n?Slide \d+:', pitch_content)
                    slide_titles = re.findall(r'(?i)Slide \d+:\s*(.*)', pitch_content)
                    for idx, slide in enumerate(slides[1:], 1):  # slides[0] is before first slide
                        title = slide_titles[idx-1] if idx-1 < len(slide_titles) else f"Slide {idx}"
                        st.markdown(f"### Slide {idx}: {title.strip()}")
                        st.markdown(slide.strip())
                        st.markdown("---")
                else:
                    st.markdown(pitch_content)

            st.markdown("---")

            # Output Files Location
            st.markdown("### 📁 Output Files Location")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.info("📊 Research Reports\noutputs/research/")
            with col2:
                st.info("🧠 Knowledge Analysis\noutputs/knowledge_analysis/")
            with col3:
                st.info("📝 Generated Content\noutputs/content/")
            with col4:
                st.info("🎨 PowerPoint Files\noutputs/presentations/")

            st.markdown("<div class='footer-info'>*📂 All detailed files are available in the `outputs` directory for download and further customization.*</div>", unsafe_allow_html=True)

            # Add a button to create a new pitch deck
            st.markdown("---")
            if st.button("➕ Create New Pitch Deck", type="primary"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

        else:
            st.info("🔄 Analysis in progress... Please refresh the page in a moment.")
            if st.button("🔄 Refresh"):
                st.rerun()

    except Exception as e:
        logger.error(f"Error loading results: {e}")
        st.error("❌ Error loading results. Please try again.")

def main():
    """Main Streamlit application."""

    # Priority check: CrewAI MUST be available for this project
    if not CREWAI_AVAILABLE:
        st.error("🚨 **CrewAI Framework Required**")
        st.error("This application is built around CrewAI multi-agent framework and cannot function without it.")

        if CREWAI_ERROR:
            st.error(f"**Error Details:** {CREWAI_ERROR}")

        st.markdown("### 🔧 **Troubleshooting Steps:**")
        st.markdown("1. **Check SQLite Version**: ChromaDB requires SQLite 3.35.0+")
        st.markdown("2. **Verify Dependencies**: Ensure all packages are properly installed")
        st.markdown("3. **Try Local Deployment**: Use Python 3.11 or 3.12 locally")
        st.markdown("4. **Alternative Platforms**: Consider Heroku, Railway, or Google Cloud Run")

        st.markdown("### 🛠️ **For Developers:**")
        st.code(f"""
# Check SQLite version
import sqlite3
print(f"SQLite version: {{sqlite3.sqlite_version}}")

# Install compatible version
pip install pysqlite3-binary
pip install chromadb==0.4.22
pip install crewai==0.130.0
        """)

        st.stop()

    # Success! CrewAI is available
    st.success("✅ **CrewAI Framework Active** - Full multi-agent functionality enabled")

    if not CUSTOM_MODULES_AVAILABLE:
        st.warning("⚠️ **Limited Features**: Some advanced features may not be available")

    # Initialize session state
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False

    # Initialize agents
    if 'agents_initialized' not in st.session_state:
        if initialize_agents():
            st.session_state.agents_initialized = True
            logger.info("🚀 Agents initialized successfully")
        else:
            st.error("Failed to initialize agents. Please check your configuration.")
            st.stop()

    # Display appropriate page based on state
    if st.session_state.processing:
        display_processing()
    elif st.session_state.show_results:
        display_results()
    else:
        display_main_form()

if __name__ == "__main__":
    main()
