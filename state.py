"""
Shared state and data models for the AutoPRISMA multi-agent system.
Implements TypedDict-based state for LangGraph compatibility.
"""
from typing import TypedDict, List, Dict, Optional, Annotated, Literal, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
import operator


# ============================================================================
# PYDANTIC DATA MODELS (for validation and serialization)
# ============================================================================

class Author(BaseModel):
    """Author information."""
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})
    
    name: str
    affiliation: Optional[str] = None


class Document(BaseModel):
    """Academic document/paper model."""
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})
    
    id: str = Field(description="Unique identifier")
    title: str
    authors: List[Author] = Field(default_factory=list)
    abstract: Optional[str] = None
    year: Optional[int] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    
    # Metadata
    source: str = Field(description="Source database (pubmed, semantic_scholar, etc.)")
    retrieved_at: datetime = Field(default_factory=datetime.now)
    
    # Screening information
    screening_status: Optional[Literal["included", "excluded", "borderline", "pending"]] = "pending"
    screening_confidence: Optional[float] = None
    screening_reason: Optional[str] = None
    screened_at: Optional[datetime] = None
    
    # Full text
    full_text: Optional[str] = None
    full_text_available: bool = False


class PICOCriteria(BaseModel):
    """PICO framework for search strategy."""
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})
    
    population: str = Field(description="Target population/participants")
    intervention: str = Field(description="Intervention being studied")
    comparator: Optional[str] = Field(None, description="Comparison/control")
    outcome: str = Field(description="Outcomes of interest")
    study_types: List[str] = Field(default_factory=list, description="e.g., RCT, cohort, etc.")


class SearchQuery(BaseModel):
    """Structured search query."""
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})
    
    query_id: str
    boolean_query: str = Field(description="Boolean search string")
    mesh_terms: List[str] = Field(default_factory=list, description="MeSH terms")
    keywords: List[str] = Field(default_factory=list)
    date_range: Optional[tuple[str, str]] = None
    databases: List[str] = Field(default_factory=lambda: ["pubmed", "semantic_scholar"])
    
    # Query metadata
    created_at: datetime = Field(default_factory=datetime.now)
    rationale: Optional[str] = None


class ScreeningCriteria(BaseModel):
    """Inclusion and exclusion criteria."""
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})
    
    inclusion_criteria: List[str] = Field(description="List of inclusion criteria")
    exclusion_criteria: List[str] = Field(description="List of exclusion criteria")
    
    # Edge case handling
    edge_case_rules: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of edge case scenarios to decisions"
    )
    
    rationale: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)


class ScreeningDecision(BaseModel):
    """Decision record for a single paper screening."""
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})
    
    document_id: str
    decision: Literal["include", "exclude", "borderline"]
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    reasoning: str = Field(description="Explanation for the decision")
    criteria_matched: List[str] = Field(default_factory=list)
    criteria_violated: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
    reviewer: str = "AI Agent"


class Theme(BaseModel):
    """Research theme identified in synthesis."""
    theme_id: str
    name: str
    description: str
    supporting_papers: List[str] = Field(description="Document IDs")
    strength: Literal["strong", "moderate", "weak"] = "moderate"


class SynthesisResult(BaseModel):
    """Results of synthesis and analysis."""
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})
    
    themes: List[Theme] = Field(default_factory=list)
    research_gaps: List[str] = Field(default_factory=list)
    contradictions: List[Dict[str, str]] = Field(default_factory=list)
    consensus_findings: List[str] = Field(default_factory=list)
    summary_statistics: Dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=datetime.now)


class PRISMAFlowDiagram(BaseModel):
    """PRISMA flow diagram data."""
    identification: Dict[str, int] = Field(
        description="Records identified through database searching, etc."
    )
    screening: Dict[str, int] = Field(
        description="Records screened, excluded, etc."
    )
    eligibility: Dict[str, int] = Field(
        description="Full-text articles assessed"
    )
    included: Dict[str, int] = Field(
        description="Studies included in synthesis"
    )
    diagram_path: Optional[str] = None


class ReviewReport(BaseModel):
    """Final systematic review report."""
    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})
    
    research_question: str
    pico: PICOCriteria
    search_strategy: SearchQuery
    screening_criteria: ScreeningCriteria
    
    prisma_flow: PRISMAFlowDiagram
    synthesis: SynthesisResult
    
    included_papers: List[Document]
    excluded_papers_summary: Dict[str, int]  # Exclusion reasons -> counts
    
    report_text: Optional[str] = None
    report_path: Optional[str] = None
    
    generated_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# LANGRAPH STATE (TypedDict for graph state management)
# ============================================================================

class PRISMAState(TypedDict, total=False):
    """
    Central state for the PRISMA workflow.
    Uses TypedDict for LangGraph compatibility.
    Annotated with operator.add for list accumulation.
    """
    
    # Input
    research_question: str
    user_preferences: Dict[str, any]
    
    # Query Strategist outputs
    pico_criteria: Optional[PICOCriteria]
    search_queries: Annotated[List[SearchQuery], operator.add]
    
    # Literature Retrieval outputs
    retrieved_documents: Annotated[List[Document], operator.add]
    retrieval_stats: Dict[str, any]
    
    # Screening Criteria outputs
    screening_criteria: Optional[ScreeningCriteria]
    
    # Abstract Evaluator outputs
    screening_decisions: Annotated[List[ScreeningDecision], operator.add]
    included_documents: Annotated[List[Document], operator.add]
    excluded_documents: Annotated[List[Document], operator.add]
    borderline_documents: Annotated[List[Document], operator.add]
    
    # Synthesis & Analysis outputs
    synthesis_result: Optional[SynthesisResult]
    
    # Report Generator outputs
    prisma_flow: Optional[PRISMAFlowDiagram]
    final_report: Optional[ReviewReport]
    
    # Workflow metadata
    current_stage: str
    workflow_status: str  # "running", "paused", "completed", "failed"
    error_message: Optional[str]
    
    # Human-in-the-loop
    hitl_checkpoints: List[str]
    hitl_approvals: Dict[str, bool]
    
    # Audit trail
    audit_log: Annotated[List[Dict[str, any]], operator.add]
    
    # Timestamps (stored as ISO format strings for JSON serialization)
    started_at: str
    completed_at: Optional[str]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_initial_state(research_question: str, **kwargs) -> PRISMAState:
    """Create initial state for a new review."""
    return PRISMAState(
        research_question=research_question,
        user_preferences=kwargs.get("user_preferences", {}),
        search_queries=[],
        retrieved_documents=[],
        retrieval_stats={},
        screening_decisions=[],
        included_documents=[],
        excluded_documents=[],
        borderline_documents=[],
        audit_log=[],
        current_stage="initialized",
        workflow_status="running",
        hitl_checkpoints=kwargs.get("hitl_checkpoints", []),
        hitl_approvals={},
        started_at=datetime.now().isoformat(),
    )


def add_audit_entry(
    state: PRISMAState,
    agent: str,
    action: str,
    details: Dict[str, any]
) -> Dict[str, any]:
    """Add an entry to the audit log."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "agent": agent,
        "action": action,
        "details": details,
    }
    return {"audit_log": [entry]}
