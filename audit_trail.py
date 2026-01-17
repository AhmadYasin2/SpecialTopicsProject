"""
Audit trail system for full provenance tracking.
Every decision made by agents is logged for transparency and reproducibility.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from config import settings
import logging

logger = logging.getLogger(__name__)


class AuditTrail:
    """
    Comprehensive audit trail for PRISMA workflow.
    Ensures every decision is traceable and explainable.
    """
    
    def __init__(self, review_id: str):
        self.review_id = review_id
        self.entries: List[Dict[str, Any]] = []
        self.audit_file = Path(settings.state_store_path) / f"audit_{review_id}.jsonl"
        
        # Create audit file
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)
        
    def log(
        self,
        agent: str,
        action: str,
        details: Dict[str, Any],
        status: str = "success",
        error: Optional[str] = None
    ):
        """
        Log an audit entry.
        
        Args:
            agent: Name of the agent taking the action
            action: Description of the action
            details: Detailed information about the action
            status: "success", "failure", "warning"
            error: Error message if status is "failure"
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "review_id": self.review_id,
            "agent": agent,
            "action": action,
            "status": status,
            "details": details,
            "error": error,
        }
        
        self.entries.append(entry)
        
        # Append to JSONL file for persistence
        try:
            with open(self.audit_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit entry: {e}")
    
    def log_query_generation(self, pico: Dict[str, Any], queries: List[Dict[str, Any]]):
        """Log query strategist actions."""
        self.log(
            agent="QueryStrategist",
            action="generate_search_queries",
            details={
                "pico_criteria": pico,
                "queries_generated": len(queries),
                "queries": queries,
            }
        )
    
    def log_paper_retrieval(self, database: str, count: int, query: str):
        """Log literature retrieval actions."""
        self.log(
            agent="LiteratureRetrieval",
            action=f"retrieve_papers_from_{database}",
            details={
                "database": database,
                "papers_retrieved": count,
                "query": query,
            }
        )
    
    def log_deduplication(self, before: int, after: int, duplicates_removed: int):
        """Log deduplication process."""
        self.log(
            agent="LiteratureRetrieval",
            action="deduplicate_papers",
            details={
                "papers_before": before,
                "papers_after": after,
                "duplicates_removed": duplicates_removed,
            }
        )
    
    def log_screening_criteria(self, criteria: Dict[str, Any]):
        """Log screening criteria definition."""
        self.log(
            agent="ScreeningCriteria",
            action="define_screening_criteria",
            details={
                "inclusion_criteria": criteria.get("inclusion_criteria", []),
                "exclusion_criteria": criteria.get("exclusion_criteria", []),
                "edge_case_rules": criteria.get("edge_case_rules", {}),
            }
        )
    
    def log_screening_decision(
        self,
        document_id: str,
        title: str,
        decision: str,
        confidence: float,
        reasoning: str,
        criteria_matched: List[str],
        criteria_violated: List[str]
    ):
        """Log individual screening decision."""
        self.log(
            agent="AbstractEvaluator",
            action="screen_paper",
            details={
                "document_id": document_id,
                "title": title,
                "decision": decision,
                "confidence": confidence,
                "reasoning": reasoning,
                "criteria_matched": criteria_matched,
                "criteria_violated": criteria_violated,
            }
        )
    
    def log_synthesis(self, themes_count: int, gaps_count: int, consensus_count: int):
        """Log synthesis and analysis."""
        self.log(
            agent="SynthesisAnalysis",
            action="synthesize_findings",
            details={
                "themes_identified": themes_count,
                "research_gaps_found": gaps_count,
                "consensus_findings": consensus_count,
            }
        )
    
    def log_report_generation(self, report_path: str, sections: List[str]):
        """Log report generation."""
        self.log(
            agent="ReportGenerator",
            action="generate_report",
            details={
                "report_path": report_path,
                "sections_generated": sections,
            }
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the audit trail."""
        agent_actions = {}
        total_errors = 0
        
        for entry in self.entries:
            agent = entry["agent"]
            if agent not in agent_actions:
                agent_actions[agent] = 0
            agent_actions[agent] += 1
            
            if entry["status"] == "failure":
                total_errors += 1
        
        return {
            "total_entries": len(self.entries),
            "agents_involved": list(agent_actions.keys()),
            "actions_per_agent": agent_actions,
            "total_errors": total_errors,
            "audit_file": str(self.audit_file),
        }
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export complete audit trail as dictionary."""
        return {
            "review_id": self.review_id,
            "entries": self.entries,
            "summary": self.get_summary(),
        }
    
    def load_from_file(self) -> bool:
        """Load audit trail from persisted file."""
        if not self.audit_file.exists():
            return False
        
        try:
            with open(self.audit_file, "r", encoding="utf-8") as f:
                self.entries = [json.loads(line) for line in f if line.strip()]
            return True
        except Exception as e:
            logger.error(f"Failed to load audit trail: {e}")
            return False
    
    def generate_html_report(self) -> str:
        """Generate HTML formatted audit report for human review."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Audit Trail - {self.review_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .entry {{ 
                    border: 1px solid #ddd; 
                    padding: 15px; 
                    margin: 10px 0; 
                    border-radius: 5px;
                }}
                .success {{ border-left: 4px solid #4CAF50; }}
                .failure {{ border-left: 4px solid #f44336; }}
                .warning {{ border-left: 4px solid #ff9800; }}
                .timestamp {{ color: #666; font-size: 0.9em; }}
                .agent {{ font-weight: bold; color: #1976D2; }}
                .details {{ background: #f5f5f5; padding: 10px; margin-top: 10px; }}
            </style>
        </head>
        <body>
            <h1>Audit Trail: {self.review_id}</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Total Entries:</strong> {len(self.entries)}</p>
                <p><strong>Audit File:</strong> {self.audit_file}</p>
            </div>
            <h2>Detailed Log</h2>
        """
        
        for entry in self.entries:
            status_class = entry.get("status", "success")
            html += f"""
            <div class="entry {status_class}">
                <div class="timestamp">{entry['timestamp']}</div>
                <div class="agent">Agent: {entry['agent']}</div>
                <div><strong>Action:</strong> {entry['action']}</div>
                <div><strong>Status:</strong> {entry['status']}</div>
                {f"<div class='error'><strong>Error:</strong> {entry['error']}</div>" if entry.get('error') else ""}
                <div class="details">
                    <strong>Details:</strong>
                    <pre>{json.dumps(entry['details'], indent=2)}</pre>
                </div>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def save_html_report(self, output_path: Optional[str] = None):
        """Save HTML audit report to file."""
        if output_path is None:
            output_path = Path(settings.state_store_path) / f"audit_{self.review_id}.html"
        
        html = self.generate_html_report()
        Path(output_path).write_text(html, encoding="utf-8")
        logger.info(f"Audit report saved to {output_path}")
        
        return str(output_path)
