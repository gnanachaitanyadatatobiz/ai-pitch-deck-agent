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
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool

# Import our custom modules
from knowledge_agent import KnowledgeAgent
from content_agent import ContentAgent
from output_manager import OutputManager
from vector_database import VectorDatabase

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
        return LLM(
            model="gpt-4o-mini",
            api_key=OPENAI_API_KEY,
            temperature=0.7,
            max_tokens=2000
        )
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {str(e)}")
        raise

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
    """Initialize all agents and managers."""
    global knowledge_agent, content_agent, output_manager
    
    try:
        logger.info("Initializing agents and managers...")
        
        # Initialize output manager
        output_manager = OutputManager()
        logger.info("Output manager initialized")
        
        # Initialize knowledge agent (simplified without tools for now)
        knowledge_agent = KnowledgeAgent()
        logger.info("Knowledge agent initialized")
        
        # Initialize content agent
        content_agent = ContentAgent()
        logger.info("Content agent initialized")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize agents: {e}")
        return False

def run_research_agent_simple(startup_data, research_prompt=None):
    """Simplified research agent function."""
    try:
        # Initialize tools
        search_tool = SerperDevTool()
        
        # Setup research agent
        research_agent = Agent(
            role="Market & Competitor Researcher",
            goal="Conduct targeted research based on startup information.",
            backstory="A skilled analyst with access to web search tools.",
            tools=[search_tool],
            allow_delegation=False,
            verbose=True,
            llm=llm
        )
        
        # Use custom research prompt if provided
        if research_prompt:
            task_description = research_prompt
        else:
            task_description = (
                f"Conduct comprehensive market and competitor research for '{startup_data['startup_name']}' "
                f"in the {startup_data['industry_type']} industry. "
                f"Focus on market trends, competitors, and opportunities."
            )
        
        research_task = Task(
            description=task_description,
            expected_output="A detailed market research report with insights and recommendations.",
            agent=research_agent
        )
        
        research_crew = Crew(
            agents=[research_agent],
            tasks=[research_task],
            process=Process.sequential,
            verbose=True,
            full_output=True
        )
        
        logger.info("Starting research process...")
        result = research_crew.kickoff()
        
        return str(result)
        
    except Exception as e:
        logger.error(f"Error during research: {e}")
        return f"Research completed with basic analysis for {startup_data['startup_name']}"

def run_enhanced_workflow():
    """Run the complete enhanced workflow with correct order."""
    try:
        # Load startup data from frontend
        with open("startup_data.json", "r", encoding="utf-8") as f:
            startup_data = json.load(f)
        
        company_name = startup_data.get('startup_name', '')
        logger.info(f"🚀 Starting enhanced workflow for: {company_name}")
        
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
        
        research_output = run_research_agent_simple(startup_data, research_prompt)
        
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
        
        # Create comprehensive report
        comprehensive_report = {
            "company_name": company_name,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "company_exists_in_db": company_check['exists'],
            "startup_data": startup_data,
            "research_output": research_output,
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

                # Save data to JSON file
                with open('startup_data.json', 'w') as f:
                    json.dump(st.session_state.startup_data, f)

                # Set processing state
                st.session_state.processing = True
                st.session_state.show_results = False
                st.rerun()
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
    """Display results matching the original design."""
    st.markdown('<h1 class="main-header">🎯 AI Analysis Results</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Complete workflow analysis with RAG technology</p>', unsafe_allow_html=True)

    # New Analysis button
    if st.button("🔄 New Analysis", type="primary"):
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # Workflow info
    st.markdown("""
    <div class="workflow-info">
        <strong>Workflow:</strong> Knowledge Agent analyzed existing data → Research Agent conducted market research → Content Agent generated pitch deck → PowerPoint created
    </div>
    """, unsafe_allow_html=True)

    try:
        # Get the latest comprehensive report
        latest_report_path = output_manager.get_latest_report()

        if latest_report_path:
            with open(latest_report_path, 'r', encoding='utf-8') as f:
                report_wrapper = json.load(f)

            report_data = report_wrapper.get('data', {})
            metadata = report_wrapper.get('metadata', {})
            files = report_data.get('files', {})

            # Show the actual context used (source_doc)
            source_doc = report_data.get('source_doc', '')
            if source_doc and 'No relevant documents found' not in source_doc:
                st.info(f"**Source of Content:**\n\nRelevant context was found in the vector database and used as background knowledge.\n\n---\n{source_doc[:2000]}{'... (truncated)' if len(source_doc) > 2000 else ''}")
            else:
                st.info("**Source of Content:**\n\nNo relevant context was found in the vector database. The Research Agent gathered up-to-date information, and the Content Agent generated your pitch deck using your input and the research findings.")

            # Format the pitch content for clean display
            pitch_content = report_data.get('pitch_content', 'No content generated')

            content = f"""
# 🎯 AI Pitch Deck Generator Results

**Company:** {report_data.get('company_name', 'Unknown')}
**Generated:** {metadata.get('generated_at', 'Unknown')}
**Status:** {'✅ Found in database' if report_data.get('company_exists_in_db', False) else '🆕 New company'}

## 📁 Generated Files
- **📊 Research Report:** `{files.get('research_file', 'Not available')}`
- **🧠 Knowledge Analysis:** `{files.get('knowledge_file', 'Not available')}`
- **📝 Content File:** `{files.get('content_file', 'Not available')}`
- **🎨 PowerPoint Presentation:** `{files.get('powerpoint_file', 'Not available')}`

---

## 🎯 Your AI-Generated Pitch Deck

{pitch_content}

---

**Workflow Status:** {report_data.get('workflow_status', 'Unknown')}

*📂 All detailed files are available in the `outputs` directory for download and further customization.*
"""

            # Display content
            st.markdown(content)

            # Output files location
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

        else:
            st.info("🔄 Analysis in progress... Please refresh the page in a moment.")
            if st.button("🔄 Refresh"):
                st.rerun()

    except Exception as e:
        logger.error(f"Error loading results: {e}")
        st.error("❌ Error loading results. Please try again.")

def main():
    """Main Streamlit application."""
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
