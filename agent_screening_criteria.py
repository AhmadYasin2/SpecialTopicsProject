"""  
Agent 3: Screening Criteria
Responsible for: Defining inclusion/exclusion criteria, creating reproducible
screening protocols, and handling edge cases.
"""
from typing import Dict, Any, List
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import logging
from config import settings
from state import PICOCriteria, ScreeningCriteria, PRISMAState, add_audit_entry

logger = logging.getLogger(__name__)


# ============================================================================
# OUTPUT MODELS
# ============================================================================

class ScreeningCriteriaOutput(BaseModel):
    """Structured screening criteria output."""
    inclusion_criteria: List[str] = Field(
        description="List of specific, measurable inclusion criteria"
    )
    exclusion_criteria: List[str] = Field(
        description="List of specific, measurable exclusion criteria"
    )
    edge_case_rules: Dict[str, str] = Field(
        description="Mapping of edge case scenarios to resolution strategies"
    )
    rationale: str = Field(
        description="Explanation of how criteria were derived from PICO and research question"
    )
    screening_workflow: List[str] = Field(
        description="Ordered steps for applying screening criteria"
    )


# ============================================================================
# SCREENING CRITERIA AGENT
# ============================================================================

class ScreeningCriteriaAgent:
    """
    Agent 3: Screening Criteria
    
    Responsibilities:
    1. Define clear, specific inclusion criteria based on PICO
    2. Define clear, specific exclusion criteria
    3. Identify potential edge cases and create resolution rules
    4. Ensure criteria are reproducible and unambiguous
    5. Document rationale for all criteria decisions
    """
    
    def __init__(self):
        self.llm = ChatOllama(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            base_url=settings.ollama_base_url
        )
        
        self.parser = PydanticOutputParser(pydantic_object=ScreeningCriteriaOutput)
    
    def define_criteria(
        self,
        research_question: str,
        pico: PICOCriteria,
        user_preferences: Dict[str, Any]
    ) -> ScreeningCriteria:
        """
        Define comprehensive screening criteria.
        
        Args:
            research_question: The systematic review research question
            pico: Extracted PICO criteria
            user_preferences: User-specified preferences (language, date range, etc.)
        
        Returns:
            ScreeningCriteria object with inclusion/exclusion rules
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert systematic review methodologist specializing in PRISMA 2020 guidelines.

Your task is to define CLEAR, SPECIFIC, REPRODUCIBLE screening criteria for abstract/title screening.

Key Principles:
1. **Specificity**: Criteria must be unambiguous (avoid "relevant", use measurable terms)
2. **Reproducibility**: Two reviewers should reach same decision using these criteria
3. **PICO-aligned**: Criteria must directly reflect the PICO framework
4. **Practical**: Criteria must be applicable at abstract-level screening
5. **Comprehensive**: Cover all relevant dimensions (population, intervention, outcomes, study design)

Inclusion Criteria Guidelines:
- Start with must-have requirements (population, intervention, outcome)
- Specify acceptable study designs
- Define minimum reporting standards
- Specify language requirements if applicable
- Define date range if applicable

Exclusion Criteria Guidelines:
- Be explicit about what to exclude
- Cover common irrelevant study types (editorials, letters, case reports if not needed)
- Specify wrong population, intervention, or outcomes
- Include animal studies exclusion if human-only
- Specify language exclusions if applicable

Edge Case Rules:
- What to do with mixed populations?
- What to do with secondary outcomes only?
- What to do with conference abstracts vs full papers?
- What to do with unclear study designs?
- Default rule for borderline cases?

Research Question: {research_question}

PICO Framework:
- Population: {population}
- Intervention: {intervention}
- Comparator: {comparator}
- Outcome: {outcome}
- Study Types: {study_types}

User Preferences:
{user_preferences}

{format_instructions}"""),
            ("user", "Generate comprehensive, reproducible screening criteria for this systematic review.")
        ])
        
        chain = prompt | self.llm | self.parser
        
        try:
            result = chain.invoke({
                "research_question": research_question,
                "population": pico.population,
                "intervention": pico.intervention,
                "comparator": pico.comparator,
                "outcome": pico.outcome,
                "study_types": ", ".join(pico.study_types),
                "user_preferences": str(user_preferences),
                "format_instructions": self.parser.get_format_instructions()
            })
            
            criteria = ScreeningCriteria(
                inclusion_criteria=result.inclusion_criteria,
                exclusion_criteria=result.exclusion_criteria,
                edge_case_rules=result.edge_case_rules,
                rationale=result.rationale
            )
            
            logger.info(f"Generated {len(criteria.inclusion_criteria)} inclusion criteria")
            logger.info(f"Generated {len(criteria.exclusion_criteria)} exclusion criteria")
            logger.info(f"Generated {len(criteria.edge_case_rules)} edge case rules")
            
            return criteria
            
        except Exception as e:
            logger.error(f"Criteria generation failed: {e}")
            # Fallback: basic criteria
            return self._generate_fallback_criteria(pico)
    
    def _generate_fallback_criteria(self, pico: PICOCriteria) -> ScreeningCriteria:
        """Generate basic fallback criteria if LLM generation fails."""
        return ScreeningCriteria(
            inclusion_criteria=[
                f"Studies involving {pico.population}",
                f"Studies investigating {pico.intervention}",
                f"Studies measuring {pico.outcome}",
                "Studies published in peer-reviewed journals",
                "Studies with available abstract or full text"
            ],
            exclusion_criteria=[
                "Non-English language studies",
                "Animal studies",
                "Case reports with n < 5",
                "Editorials, letters, or commentaries",
                "Studies not relevant to the research question"
            ],
            edge_case_rules={
                "mixed_population": "Include if target population comprises >50% of study sample",
                "unclear_design": "Mark as borderline for full-text review",
                "conference_abstract": "Include if sufficient detail provided",
                "secondary_outcomes_only": "Exclude if primary outcome not addressed"
            },
            rationale="Fallback criteria generated due to LLM error"
        )
    
    def validate_criteria(self, criteria: ScreeningCriteria) -> tuple[bool, List[str]]:
        """
        Validate screening criteria for completeness and clarity.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for minimum number of criteria
        if len(criteria.inclusion_criteria) < 3:
            issues.append("Too few inclusion criteria (minimum 3 recommended)")
        
        if len(criteria.exclusion_criteria) < 3:
            issues.append("Too few exclusion criteria (minimum 3 recommended)")
        
        # Check for vague terms
        vague_terms = ["relevant", "appropriate", "suitable", "good quality"]
        for criterion in criteria.inclusion_criteria + criteria.exclusion_criteria:
            for term in vague_terms:
                if term in criterion.lower():
                    issues.append(f"Vague term '{term}' found in criterion: {criterion}")
        
        # Check for overlaps (simplified check)
        inclusion_words = set()
        for criterion in criteria.inclusion_criteria:
            inclusion_words.update(criterion.lower().split())
        
        for criterion in criteria.exclusion_criteria:
            exclusion_words = set(criterion.lower().split())
            overlap = inclusion_words.intersection(exclusion_words)
            if overlap:
                issues.append(f"Potential contradiction in criteria (shared terms: {overlap})")
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info("Criteria validation passed")
        else:
            logger.warning(f"Criteria validation found {len(issues)} issues")
        
        return is_valid, issues
    
    def generate_screening_checklist(self, criteria: ScreeningCriteria) -> str:
        """Generate human-readable screening checklist."""
        checklist = "# Screening Checklist\n\n"
        checklist += "## Inclusion Criteria (ALL must be met)\n\n"
        
        for i, criterion in enumerate(criteria.inclusion_criteria, 1):
            checklist += f"{i}. ☐ {criterion}\n"
        
        checklist += "\n## Exclusion Criteria (ANY met = exclude)\n\n"
        
        for i, criterion in enumerate(criteria.exclusion_criteria, 1):
            checklist += f"{i}. ☐ {criterion}\n"
        
        checklist += "\n## Edge Case Rules\n\n"
        
        for scenario, resolution in criteria.edge_case_rules.items():
            checklist += f"- **{scenario.replace('_', ' ').title()}**: {resolution}\n"
        
        checklist += f"\n## Decision Rule\n\n"
        checklist += "- If ALL inclusion criteria met AND NO exclusion criteria met → **INCLUDE**\n"
        checklist += "- If ANY exclusion criteria met → **EXCLUDE**\n"
        checklist += "- If unclear or borderline → **BORDERLINE** (full-text review)\n"
        
        return checklist
    
    def run(self, state: PRISMAState) -> Dict[str, Any]:
        """
        Execute the Screening Criteria agent.
        
        Args:
            state: Current PRISMA workflow state
        
        Returns:
            State updates with screening criteria
        """
        logger.info("=== Screening Criteria Agent Started ===")
        
        research_question = state["research_question"]
        pico = state.get("pico_criteria")
        user_preferences = state.get("user_preferences", {})
        
        if not pico:
            logger.error("No PICO criteria found in state")
            return {"error_message": "PICO criteria required for screening criteria"}
        
        # Step 1: Define criteria
        logger.info("Step 1: Defining screening criteria")
        criteria = self.define_criteria(research_question, pico, user_preferences)
        
        # Step 2: Validate criteria
        logger.info("Step 2: Validating criteria")
        is_valid, issues = self.validate_criteria(criteria)
        
        if not is_valid:
            logger.warning(f"Criteria validation issues: {issues}")
            # Could trigger human-in-the-loop here
        
        # Step 3: Generate checklist
        logger.info("Step 3: Generating screening checklist")
        checklist = self.generate_screening_checklist(criteria)
        
        # Prepare state updates
        updates = {
            "screening_criteria": criteria,
            "current_stage": "screening_criteria_complete"
        }
        
        # Add audit entry
        audit_entry = add_audit_entry(
            state,
            agent="ScreeningCriteria",
            action="define_screening_criteria",
            details={
                "inclusion_criteria": criteria.inclusion_criteria,
                "exclusion_criteria": criteria.exclusion_criteria,
                "edge_case_rules": criteria.edge_case_rules,
                "validation_passed": is_valid,
                "validation_issues": issues,
                "checklist": checklist
            }
        )
        updates.update(audit_entry)
        
        logger.info("=== Screening Criteria Agent Completed ===")
        return updates


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == "__main__":
    from state import create_initial_state
    
    # Create test state with PICO
    pico = PICOCriteria(
        population="Adults with anxiety disorders",
        intervention="Cognitive behavioral therapy (CBT)",
        comparator="Standard care or pharmacotherapy",
        outcome="Reduction in anxiety symptoms",
        study_types=["RCT", "Controlled Trial", "Systematic Review"]
    )
    
    state = create_initial_state(
        research_question="What is the effectiveness of cognitive behavioral therapy for treating anxiety disorders in adults?",
        user_preferences={
            "language": "English",
            "date_range": ("2015-01-01", "2024-12-31")
        }
    )
    state["pico_criteria"] = pico
    
    # Run agent
    agent = ScreeningCriteriaAgent()
    result = agent.run(state)
    
    criteria = result["screening_criteria"]
    
    print("\n=== INCLUSION CRITERIA ===")
    for i, criterion in enumerate(criteria.inclusion_criteria, 1):
        print(f"{i}. {criterion}")
    
    print("\n=== EXCLUSION CRITERIA ===")
    for i, criterion in enumerate(criteria.exclusion_criteria, 1):
        print(f"{i}. {criterion}")
    
    print("\n=== EDGE CASE RULES ===")
    for scenario, resolution in criteria.edge_case_rules.items():
        print(f"- {scenario}: {resolution}")
