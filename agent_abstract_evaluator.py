"""
Agent 4: Abstract Evaluator
Responsible for: Batch screening of abstracts, confidence scoring, bias detection,
and comprehensive audit trail of decisions.
"""
from typing import Dict, Any, List, Literal
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import logging
from datetime import datetime
import asyncio
from config import settings
from state import (
    Document, ScreeningCriteria, ScreeningDecision,
    PRISMAState, add_audit_entry
)

logger = logging.getLogger(__name__)


# ============================================================================
# OUTPUT MODELS
# ============================================================================

class SingleScreeningResult(BaseModel):
    """Result of screening a single paper."""
    document_id: str
    decision: Literal["include", "exclude", "borderline"] = Field(
        description="Screening decision: include, exclude, or borderline"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in decision (0-1)"
    )
    reasoning: str = Field(
        description="Clear explanation for the decision with specific criteria references"
    )
    criteria_matched: List[str] = Field(
        description="List of inclusion criteria that were met"
    )
    criteria_violated: List[str] = Field(
        description="List of exclusion criteria that were violated"
    )
    potential_biases: List[str] = Field(
        default_factory=list,
        description="Any potential biases detected (publication bias, selection bias, etc.)"
    )


class BatchScreeningResult(BaseModel):
    """Result of screening multiple papers."""
    results: List[SingleScreeningResult]


# ============================================================================
# ABSTRACT EVALUATOR AGENT
# ============================================================================

class AbstractEvaluatorAgent:
    """
    Agent 4: Abstract Evaluator
    
    Responsibilities:
    1. Screen papers in batches for efficiency
    2. Apply screening criteria consistently
    3. Assign confidence scores to each decision
    4. Flag borderline cases for human review
    5. Detect potential biases
    6. Maintain complete audit trail
    """
    
    def __init__(self):
        self.llm = ChatGroq(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.groq_api_key
        )
        
        self.parser = PydanticOutputParser(pydantic_object=SingleScreeningResult)
    
    def screen_paper(
        self,
        document: Document,
        criteria: ScreeningCriteria,
        research_question: str
    ) -> ScreeningDecision:
        """
        Screen a single paper against criteria.
        
        Args:
            document: Paper to screen
            criteria: Screening criteria to apply
            research_question: Original research question for context
        
        Returns:
            ScreeningDecision with full audit information
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert systematic review screener trained in PRISMA 2020 methodology.

Your task is to screen papers by applying inclusion/exclusion criteria STRICTLY and CONSISTENTLY.

Critical Instructions:
1. Read the title and abstract carefully
2. Check EACH inclusion criterion - ALL must be met to include
3. Check EACH exclusion criterion - ANY violation means exclude
4. If unclear or borderline, mark as "borderline" for full-text review
5. Provide specific, evidence-based reasoning referencing the criteria
6. Assign honest confidence scores (low confidence = borderline)
7. Flag any potential biases you detect

Decision Rules:
- **INCLUDE**: All inclusion criteria met AND no exclusion criteria violated AND confidence >= 0.7
- **EXCLUDE**: Any exclusion criterion violated OR clearly irrelevant
- **BORDERLINE**: Uncertain, insufficient information, or confidence < 0.7

Research Question: {research_question}

Inclusion Criteria (ALL must be met):
{inclusion_criteria}

Exclusion Criteria (ANY met = exclude):
{exclusion_criteria}

Edge Case Rules:
{edge_case_rules}

Paper to Screen:
Title: {title}
Abstract: {abstract}
Year: {year}
Journal: {journal}

{format_instructions}"""),
            ("user", "Screen this paper and provide a clear decision with reasoning.")
        ])
        
        chain = prompt | self.llm | self.parser
        
        try:
            # Prepare criteria strings
            inclusion_str = "\n".join([f"- {c}" for c in criteria.inclusion_criteria])
            exclusion_str = "\n".join([f"- {c}" for c in criteria.exclusion_criteria])
            edge_cases_str = "\n".join([f"- {k}: {v}" for k, v in criteria.edge_case_rules.items()])
            
            result = chain.invoke({
                "research_question": research_question,
                "inclusion_criteria": inclusion_str,
                "exclusion_criteria": exclusion_str,
                "edge_case_rules": edge_cases_str,
                "title": document.title,
                "abstract": document.abstract or "No abstract available",
                "year": document.year or "Unknown",
                "journal": document.journal or "Unknown",
                "format_instructions": self.parser.get_format_instructions()
            })
            
            # Convert to ScreeningDecision
            decision = ScreeningDecision(
                document_id=document.id,
                decision=result.decision,
                confidence=result.confidence,
                reasoning=result.reasoning,
                criteria_matched=result.criteria_matched,
                criteria_violated=result.criteria_violated,
                timestamp=datetime.now()
            )
            
            logger.debug(f"Screened '{document.title[:50]}...' -> {result.decision} ({result.confidence:.2f})")
            
            return decision
            
        except Exception as e:
            logger.error(f"Screening failed for document {document.id}: {e}")
            # Fallback: mark as borderline with low confidence
            return ScreeningDecision(
                document_id=document.id,
                decision="borderline",
                confidence=0.0,
                reasoning=f"Screening failed due to error: {str(e)}",
                criteria_matched=[],
                criteria_violated=[],
                timestamp=datetime.now()
            )
    
    async def screen_batch_async(
        self,
        documents: List[Document],
        criteria: ScreeningCriteria,
        research_question: str,
        batch_size: int = None
    ) -> List[ScreeningDecision]:
        """
        Screen multiple papers asynchronously for efficiency.
        
        Args:
            documents: List of papers to screen
            criteria: Screening criteria
            research_question: Research question
            batch_size: Number of papers to screen in parallel
        
        Returns:
            List of ScreeningDecision objects
        """
        if batch_size is None:
            batch_size = settings.screening_batch_size
        
        all_decisions = []
        
        # Process in batches to avoid rate limits
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            logger.info(f"Screening batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
            
            # Create tasks for parallel processing
            tasks = [
                asyncio.to_thread(self.screen_paper, doc, criteria, research_question)
                for doc in batch
            ]
            
            # Execute in parallel
            decisions = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            valid_decisions = [
                d for d in decisions
                if isinstance(d, ScreeningDecision)
            ]
            
            all_decisions.extend(valid_decisions)
            
            # Small delay between batches to avoid rate limits
            if i + batch_size < len(documents):
                await asyncio.sleep(1)
        
        return all_decisions
    
    def screen_batch(
        self,
        documents: List[Document],
        criteria: ScreeningCriteria,
        research_question: str
    ) -> List[ScreeningDecision]:
        """Synchronous wrapper for batch screening."""
        return asyncio.run(
            self.screen_batch_async(documents, criteria, research_question)
        )
    
    def analyze_screening_results(
        self,
        decisions: List[ScreeningDecision]
    ) -> Dict[str, Any]:
        """
        Analyze screening results for statistics and potential issues.
        
        Returns:
            Dictionary with screening statistics and insights
        """
        total = len(decisions)
        included = sum(1 for d in decisions if d.decision == "include")
        excluded = sum(1 for d in decisions if d.decision == "exclude")
        borderline = sum(1 for d in decisions if d.decision == "borderline")
        
        # Confidence statistics
        confidences = [d.confidence for d in decisions]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        low_confidence_count = sum(1 for c in confidences if c < settings.confidence_threshold)
        
        # Exclusion reasons analysis
        exclusion_reasons = {}
        for decision in decisions:
            if decision.decision == "exclude":
                for criterion in decision.criteria_violated:
                    exclusion_reasons[criterion] = exclusion_reasons.get(criterion, 0) + 1
        
        # Sort exclusion reasons by frequency
        top_exclusion_reasons = sorted(
            exclusion_reasons.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            "total_screened": total,
            "included": included,
            "excluded": excluded,
            "borderline": borderline,
            "inclusion_rate": included / total if total > 0 else 0,
            "average_confidence": avg_confidence,
            "low_confidence_count": low_confidence_count,
            "top_exclusion_reasons": top_exclusion_reasons,
        }
    
    def flag_high_priority_borderline(
        self,
        decisions: List[ScreeningDecision],
        top_n: int = 10
    ) -> List[ScreeningDecision]:
        """
        Identify borderline cases that most urgently need human review.
        Prioritizes by confidence (lower = higher priority).
        """
        borderline = [d for d in decisions if d.decision == "borderline"]
        # Sort by confidence (ascending)
        borderline.sorted_by_priority = sorted(borderline, key=lambda x: x.confidence)
        return borderline.sorted_by_priority[:top_n]
    
    def run(self, state: PRISMAState) -> Dict[str, Any]:
        """
        Execute the Abstract Evaluator agent.
        
        Args:
            state: Current PRISMA workflow state
        
        Returns:
            State updates with screening decisions
        """
        logger.info("=== Abstract Evaluator Agent Started ===")
        
        documents = state.get("retrieved_documents", [])
        criteria = state.get("screening_criteria")
        research_question = state["research_question"]
        
        if not documents:
            logger.error("No documents found for screening")
            return {"error_message": "No documents available for screening"}
        
        if not criteria:
            logger.error("No screening criteria found")
            return {"error_message": "Screening criteria required"}
        
        logger.info(f"Screening {len(documents)} papers...")
        
        # Screen all papers
        decisions = self.screen_batch(documents, criteria, research_question)
        
        # Categorize papers
        included_docs = []
        excluded_docs = []
        borderline_docs = []
        
        for decision in decisions:
            # Find the document
            doc = next((d for d in documents if d.id == decision.document_id), None)
            if not doc:
                continue
            
            # Update document with screening info
            doc.screening_status = decision.decision
            doc.screening_confidence = decision.confidence
            doc.screening_reason = decision.reasoning
            doc.screened_at = decision.timestamp
            
            # Categorize
            if decision.decision == "include":
                included_docs.append(doc)
            elif decision.decision == "exclude":
                excluded_docs.append(doc)
            else:
                borderline_docs.append(doc)
        
        # Analyze results
        analysis = self.analyze_screening_results(decisions)
        
        logger.info(f"Screening complete: {analysis['included']} included, "
                   f"{analysis['excluded']} excluded, {analysis['borderline']} borderline")
        
        # Prepare state updates
        updates = {
            "screening_decisions": decisions,
            "included_documents": included_docs,
            "excluded_documents": excluded_docs,
            "borderline_documents": borderline_docs,
            "current_stage": "abstract_evaluation_complete"
        }
        
        # Add audit entry
        audit_entry = add_audit_entry(
            state,
            agent="AbstractEvaluator",
            action="screen_abstracts",
            details={
                "total_screened": len(decisions),
                "screening_results": analysis,
                "sample_decisions": [
                    {
                        "document_id": d.document_id,
                        "decision": d.decision,
                        "confidence": d.confidence,
                        "reasoning": d.reasoning
                    }
                    for d in decisions[:5]  # Sample of first 5
                ]
            }
        )
        updates.update(audit_entry)
        
        logger.info("=== Abstract Evaluator Agent Completed ===")
        return updates


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == "__main__":
    from state import create_initial_state, Author
    
    # Create test documents
    test_docs = [
        Document(
            id="test1",
            title="Cognitive Behavioral Therapy for Anxiety Disorders: A Randomized Controlled Trial",
            authors=[Author(name="John Doe")],
            abstract="This RCT examined the effectiveness of CBT in treating generalized anxiety disorder in adults. 120 participants were randomized to CBT or waitlist control. Results showed significant reduction in anxiety symptoms.",
            year=2020,
            journal="Journal of Anxiety Disorders",
            source="test"
        ),
        Document(
            id="test2",
            title="The Role of Serotonin in Depression: A Review",
            authors=[Author(name="Jane Smith")],
            abstract="This review examines the neurobiological mechanisms of depression, focusing on serotonin pathways.",
            year=2019,
            journal="Neuroscience Reviews",
            source="test"
        )
    ]
    
    # Create test criteria
    from state import ScreeningCriteria
    criteria = ScreeningCriteria(
        inclusion_criteria=[
            "Studies involving adults with anxiety disorders",
            "Studies investigating cognitive behavioral therapy",
            "Studies measuring anxiety symptoms as outcomes",
            "Randomized controlled trials or controlled studies"
        ],
        exclusion_criteria=[
            "Studies focusing on depression only",
            "Animal studies",
            "Studies with children or adolescents only",
            "Non-interventional studies"
        ],
        edge_case_rules={}
    )
    
    # Create state
    state = create_initial_state(
        research_question="Effectiveness of CBT for anxiety disorders in adults"
    )
    state["retrieved_documents"] = test_docs
    state["screening_criteria"] = criteria
    
    # Run agent
    agent = AbstractEvaluatorAgent()
    result = agent.run(state)
    
    print("\n=== Screening Results ===")
    print(f"Included: {len(result['included_documents'])}")
    print(f"Excluded: {len(result['excluded_documents'])}")
    print(f"Borderline: {len(result['borderline_documents'])}")
    
    print("\n=== Sample Decisions ===")
    for decision in result["screening_decisions"]:
        print(f"\nDocument: {decision.document_id}")
        print(f"Decision: {decision.decision} (confidence: {decision.confidence:.2f})")
        print(f"Reasoning: {decision.reasoning}")
