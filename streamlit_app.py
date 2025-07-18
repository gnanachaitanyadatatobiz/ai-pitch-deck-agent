"""
Streamlit Pitch Deck Generator - Full RAG & Agentic Workflow
"""

import streamlit as st
import json
import logging
from datetime import datetime
from crewai import Crew, Process, Task
from content_agent import ContentAgent
from research_agent import ResearchAgent
from vector_database import VectorDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def apply_custom_css():
    st.markdown("""
    <style>
        .main-header { text-align: center; color: #4f46e5; font-size: 2.5rem; font-weight: bold; margin-bottom: 1rem; }
        .sub-header { text-align: center; color: #6b7280; margin-bottom: 2rem; }
        .workflow-info { background-color: #eff6ff; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #3b82f6; margin-bottom: 2rem; }
        .stButton > button { background-color: #4f46e5; color: white; font-weight: bold; padding: 0.75rem 2rem; border-radius: 0.5rem; border: none; transition: all 0.3s; }
        .stButton > button:hover { background-color: #4338ca; transform: scale(1.05); }
        .footer-info { background-color: #f9fafb; padding: 1rem; border-radius: 0.5rem; text-align: center; margin-top: 2rem; font-size: 0.75rem; color: #6b7280; }
        .source-document-box { background-color: #f3f4f6; border: 1px solid #d1d5db; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem; }
    </style>
    """, unsafe_allow_html=True)

def run_agentic_workflow(startup_data: dict):
    logger.info("🏁 Starting agentic workflow...")
    # This function uses CrewAI agents for all pitch deck generation steps
    db = VectorDatabase()
    query = f"A startup in the {startup_data.get('industry_type', '')} space focused on {startup_data.get('key_problem_solved', '')}"
    logger.info(f"🚀 Querying vector database with: '{query}'")
    source_doc = db.search_by_query(query)
    logger.info("✅ Vector database query finished.")

    # Check if the context is relevant
    industry = startup_data.get('industry_type', '').lower()
    problem = startup_data.get('key_problem_solved', '').lower()
    context_is_relevant = False
    if source_doc and source_doc != "No relevant documents found for the query.":
        # Check if industry or problem keywords are in the context
        context_lower = source_doc.lower()
        if industry in context_lower or problem in context_lower:
            context_is_relevant = True
    
    if not context_is_relevant:
        logger.warning("⚠️ Retrieved context is not relevant to the user's input. Skipping context and using only user data.")
        source_doc = ""  # Use only user input
        st.session_state.context_warning = True
    else:
        st.session_state.context_warning = False

    research_agent_instance = ResearchAgent()
    content_agent_instance = ContentAgent()
    
    research_agent_instance.agent.llm = content_agent_instance.agent.llm

    research_task = Task(
        description=f"""
            Conduct comprehensive, up-to-date market and competitor research for a startup named '{startup_data['startup_name']}'.
            The startup is in the '{startup_data.get('industry_type', 'unknown')}' industry.

            Focus on:
            1.  Current market size (TAM, SAM, SOM).
            2.  Key market trends and drivers.
            3.  A detailed landscape of direct and indirect competitors.

            To guide your research, here is some initial context retrieved from our internal knowledge base:
            ---
            {{knowledge_analysis}}
            ---
        """,
        expected_output="A detailed market research report including current market size, competitor landscape, and recent industry trends.",
        agent=research_agent_instance.agent
    )

    content_task = Task(
        description=f"""
            Create a full, 12-slide investor-ready pitch deck for the startup '{startup_data['startup_name']}'.

            You MUST use the specific details provided below in the 'Startup Data' section to populate the slides. Do NOT use placeholders like '[Insert...]' or 'Year X'. Be specific and quantitative where data is provided.

            Synthesize this data with the market research (provided in the context) to create a cohesive and compelling narrative.

            **Startup Data:**
            ---
            {{startup_data}}
            ---
        """,
        expected_output="A complete, well-structured pitch deck in a clean, readable text format. Each slide should have a title and key bullet points, populated with the specific data provided.",
        agent=content_agent_instance.agent,
        context=[research_task]
    )

    main_crew = Crew(
        agents=[research_agent_instance.agent, content_agent_instance.agent],
        tasks=[research_task, content_task],
        process=Process.sequential,
        verbose=True
    )

    logger.info("🚀 Kicking off the Research and Content Agents...")
    final_result = main_crew.kickoff(inputs={'knowledge_analysis': source_doc, 'startup_data': json.dumps(startup_data)})
    logger.info("✅ Main agentic workflow completed.")
    
    return str(final_result), source_doc

def display_main_form():
    st.markdown('<h1 class="main-header">🚀 AI Pitch Deck Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by a Multi-Agent CrewAI System (All pitch deck content is generated by CrewAI agents)</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="workflow-info">
        <strong>Workflow:</strong> Direct RAG → Research Agent (CrewAI) → Content Agent (CrewAI) → Final Pitch Deck
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("pitch_deck_form"):
        col1, col2 = st.columns(2)
        with col1:
            startup_name = st.text_input("Startup Name", key="startup_name")
            product_name = st.text_input("Product Name", key="product_name")
            founder_name = st.text_input("Founder(s) Name(s)", key="founder_name")
            vision_statement = st.text_area("Vision Statement", height=80, key="vision_statement")
            key_problem_solved = st.text_area("Key Problem Solved", height=100, key="key_problem_solved")
            solution_summary = st.text_area("Solution Summary", height=100, key="solution_summary")
            business_model = st.text_area("Business Model (e.g., SaaS, Marketplace)", height=80, key="business_model")
            monetization_plan = st.text_area("Monetization Plan", height=80, key="monetization_plan")

        with col2:
            industry_type = st.text_input("Industry Type", key="industry_type")
            target_customer_profile = st.text_input("Target Customer Profile", key="target_customer_profile")
            founder_bio = st.text_area("Founder(s) Bio", height=80, key="founder_bio")
            team_summary = st.text_area("Team Summary (Key members, expertise)", height=80, key="team_summary")
            market_size = st.text_input("Market Size (TAM, SAM, SOM)", key="market_size")
            competitors = st.text_input("Key Competitors", key="competitors")
            why_you_win = st.text_area("Why You'll Win (Competitive Advantage)", height=80, key="why_you_win")
            acquisition_strategy = st.text_area("Customer Acquisition Strategy", height=80, key="acquisition_strategy")

        st.markdown("---")
        funding_amount = st.text_input("Funding Amount Sought", key="funding_amount")
        use_of_funds_split_percentages = st.text_area("Use of Funds (e.g., 40% Product, 40% Marketing, 20% Ops)", height=80, key="use_of_funds_split_percentages")
        transactions = st.text_area("Transactions / Traction (e.g., 1M downloads, 4.8 rating)", height=80, key="transactions")

        submitted = st.form_submit_button("🚀 Generate AI Pitch Deck", use_container_width=True)
        
        if submitted:
            startup_data = {
                'startup_name': startup_name, 'industry_type': industry_type,
                'founder_name': founder_name, 'founder_bio': founder_bio,
                'team_summary': team_summary, 'product_name': product_name,
                'vision_statement': vision_statement, 'key_problem_solved': key_problem_solved,
                'solution_summary': solution_summary, 'target_customer_profile': target_customer_profile,
                'business_model': business_model, 'acquisition_strategy': acquisition_strategy,
                'market_size': market_size, 'competitors': competitors,
                'why_you_win': why_you_win, 'funding_amount': funding_amount,
                'use_of_funds_split_percentages': use_of_funds_split_percentages,
                'transactions': transactions, 'monetization_plan': monetization_plan
            }
            
            required_fields = [startup_name, industry_type, key_problem_solved, solution_summary]
            if all(field.strip() for field in required_fields):
                st.session_state.startup_data = startup_data
                st.session_state.processing = True
                st.rerun()
            else:
                st.error("Please fill in at least the Startup Name, Industry, Problem, and Solution fields.")

def display_processing():
    st.header("Processing your request...")
    st.info("This process involves multiple AI agents and may take a few minutes. Please wait.")
    
    try:
        if 'crew_result' not in st.session_state:
            with st.spinner("Agents are working... Please do not close this window."):
                result, source_doc = run_agentic_workflow(st.session_state.startup_data)
                st.session_state.crew_result = result
                st.session_state.source_document = source_doc
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        logger.error(f"Workflow error: {e}")
        st.session_state.processing = False
        if st.button("Back to Form"):
            st.session_state.show_results = False
            st.rerun()
        return

    st.session_state.processing = False
    st.session_state.show_results = True
    st.rerun()

def display_results():
    st.header("Generated Pitch Deck Content (by CrewAI Agents)")

    # Source of Content section
    source_doc = st.session_state.get('source_document', None)
    context_warning = st.session_state.get('context_warning', False)
    if source_doc and source_doc.strip():
        st.info("**Source of Content:**\n\nRelevant context was found in the vector database and used as background knowledge for the Research Agent and Content Agent to generate your pitch deck.")
    else:
        st.info("**Source of Content:**\n\nNo relevant context was found in the vector database. The Research Agent gathered up-to-date information, and the Content Agent generated your pitch deck using your input and the research findings.")

    if 'crew_result' in st.session_state:
        st.markdown(st.session_state.crew_result)

    # Always show the source context (source_document) if available
    if source_doc is not None:
        with st.expander("View Source Context Used by CrewAI Agents"):
            st.text_area(label="Source Context", value=source_doc if source_doc else "No relevant context was found in the vector database.", height=300, disabled=True)

    if st.button("Generate a New Pitch Deck"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

def main():
    st.set_page_config(page_title="AI Pitch Deck Generator", layout="wide")
    apply_custom_css()

    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False

    if st.sidebar.button("Start Over"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    if st.session_state.processing:
        display_processing()
    elif st.session_state.show_results:
        display_results()
    else:
        display_main_form()

if __name__ == "__main__":
    main()
