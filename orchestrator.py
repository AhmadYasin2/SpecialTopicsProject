"""
Agent 7: Orchestrator (Optional but essential)
Responsible for: State management, agent coordination, dependency control,
error recovery, and progress tracking using LangGraph.
"""
from typing import Dict, Any, List, Literal
import logging
from datetime import datetime
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import asyncio

from config import settings
from state import PRISMAState, create_initial_state
from audit_trail import AuditTrail

# Import all agents
from agent_query_strategist import QueryStrategistAgent
from agent_literature_retrieval import LiteratureRetrievalAgent
from agent_screening_criteria import ScreeningCriteriaAgent
from agent_abstract_evaluator import AbstractEvaluatorAgent
from agent_synthesis_analysis import SynthesisAnalysisAgent
from agent_report_generator import ReportGeneratorAgent

logger = logging.getLogger(__name__)


# ============================================================================
# ORCHESTRATOR AGENT
# ============================================================================

class AutoPRISMAOrchestrator:
    """
    Agent 7: Orchestrator
    
    Coordinates the entire PRISMA workflow using LangGraph for:
    - Deterministic agent ordering
    - State management and checkpointing
    - Error recovery
    - Human-in-the-loop intervention points
    - Progress tracking
    """
    
    def __init__(self):
        # Initialize all agents
        self.query_strategist = QueryStrategistAgent()
        self.literature_retrieval = LiteratureRetrievalAgent()
        self.screening_criteria = ScreeningCriteriaAgent()
        self.abstract_evaluator = AbstractEvaluatorAgent()
        self.synthesis_analysis = SynthesisAnalysisAgent()
        self.report_generator = ReportGeneratorAgent()
        
        # Build the workflow graph
        self.graph = self._build_graph()
        
        # Audit trail
        self.audit_trail = None
    
    def _build_graph(self) -> StateGraph:
        """
        Build LangGraph workflow with proper agent ordering and dependencies.
        """
        # Create workflow graph
        workflow = StateGraph(PRISMAState)
        
        # Add agent nodes
        workflow.add_node("query_strategist", self._run_query_strategist)
        workflow.add_node("literature_retrieval", self._run_literature_retrieval)
        workflow.add_node("screening_criteria", self._run_screening_criteria)
        workflow.add_node("abstract_evaluator", self._run_abstract_evaluator)
        workflow.add_node("synthesis_analysis", self._run_synthesis_analysis)
        workflow.add_node("report_generator", self._run_report_generator)
        
        # Add human-in-the-loop checkpoint nodes
        workflow.add_node("hitl_checkpoint", self._hitl_checkpoint)
        
        # Define edges (workflow sequence)
        workflow.set_entry_point("query_strategist")
        
        workflow.add_edge("query_strategist", "literature_retrieval")
        workflow.add_edge("literature_retrieval", "screening_criteria")
        workflow.add_edge("screening_criteria", "hitl_checkpoint")  # Optional human review
        
        # Conditional edge after HITL checkpoint
        workflow.add_conditional_edges(
            "hitl_checkpoint",
            self._should_continue_after_hitl,
            {
                "continue": "abstract_evaluator",
                "end": END
            }
        )
        
        # Conditional edge after abstract evaluation
        workflow.add_conditional_edges(
            "abstract_evaluator",
            self._should_run_synthesis,
            {
                "continue": "synthesis_analysis",
                "skip": "report_generator"
            }
        )
        
        workflow.add_edge("synthesis_analysis", "report_generator")
        workflow.add_edge("report_generator", END)
        
        # Compile with checkpointing for state persistence
        checkpointer = MemorySaver()
        return workflow.compile(checkpointer=checkpointer)
    
    # ========================================================================
    # AGENT WRAPPER METHODS
    # ========================================================================
    
    def _run_query_strategist(self, state: PRISMAState) -> PRISMAState:
        """Execute Query Strategist agent and update state."""
        try:
            logger.info("🔍 Running Query Strategist Agent...")
            updates = self.query_strategist.run(state)
            
            if self.audit_trail:
                self.audit_trail.log_query_generation(
                    pico=updates["pico_criteria"].model_dump(mode='json') if updates.get("pico_criteria") else {},
                    queries=[q.model_dump(mode='json') for q in updates.get("search_queries", [])]
                )
            
            return {**state, **updates}
        except Exception as e:
            logger.error(f"Query Strategist failed: {e}")
            return {**state, "error_message": str(e), "workflow_status": "failed"}
    
    def _run_literature_retrieval(self, state: PRISMAState) -> PRISMAState:
        """Execute Literature Retrieval agent and update state."""
        try:
            logger.info("📚 Running Literature Retrieval Agent...")
            updates = asyncio.run(self.literature_retrieval.run(state))
            
            if self.audit_trail:
                stats = updates.get("retrieval_stats", {})
                self.audit_trail.log_deduplication(
                    before=stats.get("total_retrieved", 0),
                    after=stats.get("unique_documents", 0),
                    duplicates_removed=stats.get("duplicates_removed", 0)
                )
            
            return {**state, **updates}
        except Exception as e:
            logger.error(f"Literature Retrieval failed: {e}")
            return {**state, "error_message": str(e), "workflow_status": "failed"}
    
    def _run_screening_criteria(self, state: PRISMAState) -> PRISMAState:
        """Execute Screening Criteria agent and update state."""
        try:
            logger.info("📋 Running Screening Criteria Agent...")
            updates = self.screening_criteria.run(state)
            
            if self.audit_trail:
                criteria = updates.get("screening_criteria")
                if criteria:
                    self.audit_trail.log_screening_criteria({
                        "inclusion_criteria": criteria.inclusion_criteria,
                        "exclusion_criteria": criteria.exclusion_criteria,
                        "edge_case_rules": criteria.edge_case_rules
                    })
            
            return {**state, **updates}
        except Exception as e:
            logger.error(f"Screening Criteria failed: {e}")
            return {**state, "error_message": str(e), "workflow_status": "failed"}
    
    def _run_abstract_evaluator(self, state: PRISMAState) -> PRISMAState:
        """Execute Abstract Evaluator agent and update state."""
        try:
            logger.info("✅ Running Abstract Evaluator Agent...")
            updates = self.abstract_evaluator.run(state)
            
            # Log sample screening decisions to audit trail
            if self.audit_trail:
                for decision in updates.get("screening_decisions", [])[:10]:  # Sample
                    doc = next((d for d in state.get("retrieved_documents", []) if d.id == decision.document_id), None)
                    if doc:
                        self.audit_trail.log_screening_decision(
                            document_id=decision.document_id,
                            title=doc.title,
                            decision=decision.decision,
                            confidence=decision.confidence,
                            reasoning=decision.reasoning,
                            criteria_matched=decision.criteria_matched,
                            criteria_violated=decision.criteria_violated
                        )
            
            return {**state, **updates}
        except Exception as e:
            logger.error(f"Abstract Evaluator failed: {e}")
            return {**state, "error_message": str(e), "workflow_status": "failed"}
    
    def _run_synthesis_analysis(self, state: PRISMAState) -> PRISMAState:
        """Execute Synthesis & Analysis agent and update state."""
        try:
            logger.info("🔬 Running Synthesis & Analysis Agent...")
            updates = self.synthesis_analysis.run(state)
            
            # Check if synthesis returned an error
            if updates.get("error_message"):
                logger.error(f"Synthesis returned error: {updates['error_message']}")
                return {**state, **updates, "workflow_status": "failed"}
            
            if self.audit_trail:
                synthesis = updates.get("synthesis_result")
                if synthesis:
                    self.audit_trail.log_synthesis(
                        themes_count=len(synthesis.themes),
                        gaps_count=len(synthesis.research_gaps),
                        consensus_count=len(synthesis.consensus_findings)
                    )
            
            return {**state, **updates}
        except Exception as e:
            logger.error(f"Synthesis & Analysis failed: {e}", exc_info=True)
            return {**state, "error_message": str(e), "workflow_status": "failed"}
    
    def _run_report_generator(self, state: PRISMAState) -> PRISMAState:
        """Execute Report Generator agent and update state."""
        try:
            logger.info("📄 Running Report Generator Agent...")
            updates = self.report_generator.run(state)
            
            if self.audit_trail:
                final_report = updates.get("final_report")
                if final_report:
                    self.audit_trail.log_report_generation(
                        report_path=final_report.report_path or "unknown",
                        sections=["introduction", "methods", "results", "discussion", "conclusion"]
                    )
            
            return {**state, **updates}
        except Exception as e:
            logger.error(f"Report Generator failed: {e}")
            return {**state, "error_message": str(e), "workflow_status": "failed"}
    
    # ========================================================================
    # CONDITIONAL LOGIC
    # ========================================================================
    
    def _should_run_synthesis(self, state: PRISMAState) -> str:
        """Determine if synthesis should run based on included papers."""
        included_docs = state.get("included_documents", [])
        
        if len(included_docs) == 0:
            logger.warning("⚠️  No papers included - skipping synthesis")
            return "skip"
        
        if len(included_docs) < 3:
            logger.warning(f"⚠️  Only {len(included_docs)} papers included - synthesis may be limited")
        
        return "continue"
    
    # ========================================================================
    # HUMAN-IN-THE-LOOP
    # ========================================================================
    
    def _hitl_checkpoint(self, state: PRISMAState) -> PRISMAState:
        """
        Human-in-the-loop checkpoint.
        Pauses workflow for human review if enabled.
        """
        hitl_checkpoints = state.get("hitl_checkpoints", [])
        
        if "after_screening_criteria" in hitl_checkpoints:
            logger.info("⏸️  Human-in-the-loop checkpoint: Review screening criteria")
            logger.info("Screening criteria defined. Review and approve to continue.")
            
            # In a full implementation, this would trigger UI notification
            # For now, just log
            state["workflow_status"] = "paused_for_review"
        
        return state
    
    def _should_continue_after_hitl(
        self,
        state: PRISMAState
    ) -> Literal["continue", "end"]:
        """Decide whether to continue after HITL checkpoint."""
        workflow_status = state.get("workflow_status", "running")
        
        if workflow_status == "failed":
            return "end"
        
        # Check if human approved (in real implementation, this would check actual approval)
        hitl_approvals = state.get("hitl_approvals", {})
        if state.get("workflow_status") == "paused_for_review":
            # Auto-approve for demo purposes
            logger.info("✅ Auto-approving for demo (in production, wait for human approval)")
            return "continue"
        
        return "continue"
    
    # ========================================================================
    # PUBLIC API
    # ========================================================================
    
    async def run_review(
        self,
        research_question: str,
        databases: List[str] = None,
        date_range: tuple = None,
        enable_hitl: bool = False,
        hitl_checkpoints: List[str] = None
    ) -> Dict[str, Any]:
        """
        Run complete PRISMA systematic review workflow.
        
        Args:
            research_question: The research question to investigate
            databases: List of databases to search (default: ["pubmed", "semantic_scholar", "openalex"])
            date_range: Tuple of (start_date, end_date) in "YYYY-MM-DD" format
            enable_hitl: Whether to enable human-in-the-loop checkpoints
            hitl_checkpoints: List of checkpoint names to enable
        
        Returns:
            Final state with complete review results
        """
        logger.info("="*80)
        logger.info("🚀 AUTOPRISMA SYSTEMATIC REVIEW STARTED")
        logger.info("="*80)
        logger.info(f"Research Question: {research_question}")
        
        # Initialize audit trail
        review_id = f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.audit_trail = AuditTrail(review_id)
        
        # Create initial state
        if databases is None:
            databases = ["pubmed", "semantic_scholar", "openalex"]
        
        user_preferences = {
            "databases": databases,
            "date_range": date_range,
            "enable_hitl": enable_hitl
        }
        
        if hitl_checkpoints is None and enable_hitl:
            hitl_checkpoints = ["after_screening_criteria"]
        
        initial_state = create_initial_state(
            research_question=research_question,
            user_preferences=user_preferences,
            hitl_checkpoints=hitl_checkpoints or []
        )
        
        # Run the workflow
        try:
            config = {"configurable": {"thread_id": review_id}}
            
            # Execute the graph
            final_state = None
            async for state_update in self.graph.astream(initial_state, config):
                # Log progress
                for node_name, node_state in state_update.items():
                    current_stage = node_state.get("current_stage", "unknown")
                    logger.info(f"📍 Stage: {current_stage}")
                    final_state = node_state
            
            # Save audit trail
            if self.audit_trail:
                audit_html_path = self.audit_trail.save_html_report()
                logger.info(f"📋 Audit trail saved to: {audit_html_path}")
            
            logger.info("="*80)
            logger.info("✅ AUTOPRISMA SYSTEMATIC REVIEW COMPLETED")
            logger.info("="*80)
            
            return {
                "status": "success",
                "final_state": final_state,
                "review_id": review_id,
                "audit_trail_path": audit_html_path if self.audit_trail else None,
                "final_report": final_state.get("final_report") if final_state else None
            }
            
        except Exception as e:
            logger.error(f"❌ Workflow failed: {e}", exc_info=True)
            
            if self.audit_trail:
                self.audit_trail.log(
                    agent="Orchestrator",
                    action="run_review",
                    details={"research_question": research_question},
                    status="failure",
                    error=str(e)
                )
                audit_html_path = self.audit_trail.save_html_report()
            
            return {
                "status": "failed",
                "error": str(e),
                "review_id": review_id,
                "audit_trail_path": audit_html_path if self.audit_trail else None
            }
    
    def run_review_sync(self, *args, **kwargs) -> Dict[str, Any]:
        """Synchronous wrapper for run_review."""
        return asyncio.run(self.run_review(*args, **kwargs))


# ============================================================================
# STANDALONE TESTING & CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # CLI argument parsing
    parser = argparse.ArgumentParser(description="AutoPRISMA Systematic Review System")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Research question for systematic review"
    )
    parser.add_argument(
        "--databases",
        type=str,
        nargs="+",
        default=["semantic_scholar", "arxiv"],
        help="Databases to search (pubmed, semantic_scholar, arxiv, openalex)"
    )
    parser.add_argument(
        "--date-from",
        type=str,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--date-to",
        type=str,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--enable-hitl",
        action="store_true",
        help="Enable human-in-the-loop checkpoints"
    )
    
    args = parser.parse_args()
    
    # Prepare date range
    date_range = None
    if args.date_from and args.date_to:
        date_range = (args.date_from, args.date_to)
    
    # Run orchestrator
    orchestrator = AutoPRISMAOrchestrator()
    result = orchestrator.run_review_sync(
        research_question=args.query,
        databases=args.databases,
        date_range=date_range,
        enable_hitl=args.enable_hitl
    )
    
    # Print results
    if result["status"] == "success":
        print("\n✅ Review completed successfully!")
        print(f"📋 Audit trail: {result['audit_trail_path']}")
        
        final_report = result.get("final_report")
        if final_report:
            print(f"📄 Report: {final_report.report_path}")
            print(f"\n📊 Summary:")
            print(f"  - Included papers: {len(final_report.included_papers)}")
            print(f"  - Themes identified: {len(final_report.synthesis.themes)}")
            print(f"  - Research gaps: {len(final_report.synthesis.research_gaps)}")
    else:
        print(f"\n❌ Review failed: {result['error']}")
        print(f"📋 Audit trail: {result.get('audit_trail_path')}")
