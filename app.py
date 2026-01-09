"""
Streamlit UI for AutoPRISMA System
Run with: streamlit run app.py
"""
import streamlit as st
import requests
import time
from datetime import datetime, date
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="AutoPRISMA - Automated Systematic Review",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def start_review(research_question, databases, date_from, date_to, enable_hitl):
    """Start a new review via API."""
    payload = {
        "research_question": research_question,
        "databases": databases,
        "enable_hitl": enable_hitl
    }
    
    if date_from and date_to:
        payload["date_from"] = date_from.strftime("%Y-%m-%d")
        payload["date_to"] = date_to.strftime("%Y-%m-%d")
    
    try:
        response = requests.post(f"{API_BASE_URL}/reviews", json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to start review: {e}")
        return None


def get_review_status(job_id):
    """Get status of a review job."""
    try:
        response = requests.get(f"{API_BASE_URL}/reviews/{job_id}/status")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        return None


def get_review_result(job_id):
    """Get results of a completed review."""
    try:
        response = requests.get(f"{API_BASE_URL}/reviews/{job_id}/result")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to get result: {e}")
        return None


def download_report(job_id, format="markdown"):
    """Download report file."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/reviews/{job_id}/report",
            params={"format": format}
        )
        response.raise_for_status()
        return response.content
    except Exception as e:
        st.error(f"Failed to download report: {e}")
        return None


# ============================================================================
# MAIN UI
# ============================================================================

def main():
    # Header
    st.title("🔬 AutoPRISMA")
    st.markdown("### Automated PRISMA 2020 Systematic Literature Review System")
    
    st.markdown("""
    **AutoPRISMA** is a multi-agent AI system that automates the PRISMA 2020 systematic review workflow
    with full transparency, auditability, and traceability.
    
    ⚠️ **IMPORTANT**: This is a research prototype. All results require expert validation and should NOT be used
    for publication, clinical decisions, or policy-making without thorough human review.
    """)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Check API status
        try:
            response = requests.get(f"{API_BASE_URL}/health")
            if response.status_code == 200:
                st.success("✅ API Connected")
            else:
                st.error("❌ API Error")
        except:
            st.error("❌ API Not Running")
            st.info("Start the API server with: `python main.py`")
        
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["New Review", "Review Status", "About"]
        )
    
    # ========================================================================
    # PAGE: NEW REVIEW
    # ========================================================================
    
    if page == "New Review":
        st.header("📝 Start New Systematic Review")
        
        # Research question
        research_question = st.text_area(
            "Research Question",
            placeholder="e.g., What is the effectiveness of cognitive behavioral therapy for treating anxiety disorders in adults?",
            height=100,
            help="Enter your research question in clear, specific language"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Databases
            st.subheader("📚 Databases")
            databases = st.multiselect(
                "Select databases to search",
                options=["pubmed", "semantic_scholar", "arxiv", "openalex"],
                default=["semantic_scholar", "arxiv"],
                help="Select one or more academic databases"
            )
        
        with col2:
            # Date range
            st.subheader("📅 Date Range")
            use_date_range = st.checkbox("Limit by publication date")
            
            if use_date_range:
                date_from = st.date_input(
                    "From",
                    value=date(2015, 1, 1),
                    max_value=date.today()
                )
                date_to = st.date_input(
                    "To",
                    value=date.today(),
                    max_value=date.today()
                )
            else:
                date_from = None
                date_to = None
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            enable_hitl = st.checkbox(
                "Enable Human-in-the-Loop",
                value=False,
                help="Pause for human review at key checkpoints"
            )
            
            st.info("""
            **Human-in-the-Loop Checkpoints:**
            - After screening criteria definition
            - After abstract evaluation (for borderline cases)
            """)
        
        st.markdown("---")
        
        # Start button
        if st.button("🚀 Start Systematic Review", type="primary", use_container_width=True):
            if not research_question or len(research_question) < 10:
                st.error("Please enter a valid research question (at least 10 characters)")
            elif not databases:
                st.error("Please select at least one database")
            else:
                with st.spinner("Starting review..."):
                    result = start_review(
                        research_question,
                        databases,
                        date_from,
                        date_to,
                        enable_hitl
                    )
                    
                    if result:
                        st.success("✅ Review started successfully!")
                        st.info(f"**Job ID:** `{result['job_id']}`")
                        st.info("Go to 'Review Status' page to monitor progress")
                        
                        # Store in session state
                        if 'job_ids' not in st.session_state:
                            st.session_state.job_ids = []
                        st.session_state.job_ids.append(result['job_id'])
    
    # ========================================================================
    # PAGE: REVIEW STATUS
    # ========================================================================
    
    elif page == "Review Status":
        st.header("📊 Review Status & Results")
        
        # Job ID input
        col1, col2 = st.columns([3, 1])
        
        with col1:
            job_id = st.text_input(
                "Job ID",
                placeholder="Enter job ID or select from recent reviews",
                help="Enter the job ID from your review"
            )
        
        with col2:
            # Recent reviews dropdown
            if 'job_ids' in st.session_state and st.session_state.job_ids:
                recent_job = st.selectbox(
                    "Recent Reviews",
                    options=[""] + st.session_state.job_ids
                )
                if recent_job:
                    job_id = recent_job
        
        if job_id:
            # Refresh button
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                if st.button("🔄 Refresh", use_container_width=True):
                    st.rerun()
            
            # Get status
            status = get_review_status(job_id)
            
            if status:
                # Status display
                status_emoji = {
                    "pending": "⏳",
                    "running": "⚙️",
                    "completed": "✅",
                    "failed": "❌"
                }
                
                st.subheader(f"{status_emoji.get(status['status'], '❓')} Status: {status['status'].upper()}")
                
                # Progress bar
                if status['status'] in ['pending', 'running']:
                    progress = status.get('progress', 0.0)
                    st.progress(progress)
                    st.info(f"Current stage: {status.get('current_stage', 'Initializing')}")
                    
                    # Auto-refresh
                    if st.checkbox("Auto-refresh (every 5 seconds)", value=True):
                        time.sleep(5)
                        st.rerun()
                
                # Details
                with st.expander("📋 Job Details", expanded=True):
                    st.write(f"**Job ID:** `{status['job_id']}`")
                    st.write(f"**Started:** {status['started_at']}")
                    if status.get('completed_at'):
                        st.write(f"**Completed:** {status['completed_at']}")
                    if status.get('error_message'):
                        st.error(f"**Error:** {status['error_message']}")
                
                # If completed, show results
                if status['status'] == 'completed':
                    st.markdown("---")
                    st.subheader("📄 Results")
                    
                    result = get_review_result(job_id)
                    
                    if result:
                        summary = result.get('summary', {})
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Papers Retrieved",
                                summary.get('total_papers_retrieved', 0)
                            )
                        
                        with col2:
                            st.metric(
                                "Papers Included",
                                summary.get('papers_included', 0),
                                delta=None
                            )
                        
                        with col3:
                            st.metric(
                                "Themes Identified",
                                summary.get('themes_identified', 0)
                            )
                        
                        with col4:
                            st.metric(
                                "Research Gaps",
                                summary.get('research_gaps', 0)
                            )
                        
                        # Screening breakdown
                        st.markdown("### Screening Breakdown")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Included", summary.get('papers_included', 0))
                        with col2:
                            st.metric("Excluded", summary.get('papers_excluded', 0))
                        with col3:
                            st.metric("Borderline", summary.get('papers_borderline', 0))
                        
                        # Download options
                        st.markdown("### 📥 Downloads")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if st.button("Download Report (Markdown)", use_container_width=True):
                                content = download_report(job_id, "markdown")
                                if content:
                                    st.download_button(
                                        "Save Markdown",
                                        content,
                                        file_name=f"review_{job_id}.md",
                                        mime="text/markdown"
                                    )
                        
                        with col2:
                            if st.button("Download Report (HTML)", use_container_width=True):
                                content = download_report(job_id, "html")
                                if content:
                                    st.download_button(
                                        "Save HTML",
                                        content,
                                        file_name=f"review_{job_id}.html",
                                        mime="text/html"
                                    )
                        
                        with col3:
                            if st.button("View Audit Trail", use_container_width=True):
                                st.info(f"Audit trail available at: {result.get('audit_trail_path')}")
                        
                        # PRISMA Flow Diagram
                        if result.get('diagram_path'):
                            st.markdown("### PRISMA Flow Diagram")
                            try:
                                st.image(result['diagram_path'], use_column_width=True)
                            except:
                                st.warning("Diagram image not available")
            
            else:
                st.error("Failed to retrieve status. Please check the job ID.")
    
    # ========================================================================
    # PAGE: ABOUT
    # ========================================================================
    
    elif page == "About":
        st.header("ℹ️ About AutoPRISMA")
        
        st.markdown("""
        ## System Architecture
        
        AutoPRISMA implements a complete PRISMA 2020-compliant systematic review workflow using
        a **multi-agent architecture** built with LangGraph and LangChain.
        
        ### 🤖 Agents
        
        1. **Query Strategist** - PICO extraction, Boolean queries, MeSH terms
        2. **Literature Retrieval** - Multi-database search (PubMed, Semantic Scholar, arXiv, OpenAlex)
        3. **Screening Criteria** - Inclusion/exclusion rules, reproducible protocol
        4. **Abstract Evaluator** - Batch screening with confidence scoring
        5. **Synthesis & Analysis** - Theme extraction, gaps, contradictions
        6. **Report Generator** - PRISMA flow diagram, structured report
        7. **Orchestrator** - Workflow coordination, state management
        
        ### 🔍 Key Features
        
        - ✅ **PRISMA 2020 Compliant** - Full alignment with PRISMA guidelines
        - ✅ **Multi-Database Search** - Semantic Scholar, PubMed, arXiv, OpenAlex
        - ✅ **Audit Trail** - Complete provenance for every decision
        - ✅ **Reproducible** - Controlled temperature, deterministic workflow
        - ✅ **Explainable** - Clear reasoning for all decisions
        - ✅ **Human-in-the-Loop** - Optional checkpoints for expert review
        
        ### ⚠️ Limitations
        
        - This is a **research prototype** for educational purposes
        - Results require **expert validation**
        - NOT suitable for publication without human review
        - LLMs can make errors - always verify critical decisions
        - Full-text analysis not implemented (abstract-level only)
        
        ### 📖 Technical Stack
        
        - **Backend**: FastAPI, Uvicorn
        - **Frontend**: Streamlit
        - **Agents**: LangGraph, LangChain, OpenAI GPT-4
        - **Vector Store**: ChromaDB / FAISS
        - **APIs**: Semantic Scholar, PubMed, arXiv, OpenAlex
        
        ### 📚 References
        
        - PRISMA 2020 Guidelines: [prisma-statement.org](http://www.prisma-statement.org/)
        - LangGraph: [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/)
        
        ### 📝 Citation
        
        If you use AutoPRISMA in your research, please cite:
        
        ```
        AutoPRISMA: Automated PRISMA 2020 Systematic Literature Review System
        [Course Project - 2026]
        ```
        
        ---
        
        **Disclaimer**: This system is for educational and demonstration purposes only.
        All results must be validated by qualified experts before any real-world use.
        """)


# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
