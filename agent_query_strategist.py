"""
Agent 1: Query Strategist
Responsible for: PICO extraction, Boolean query construction, MeSH term identification,
and database-specific query syntax generation.
"""
from typing import Dict, Any, List
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import logging
from config import settings
from state import PICOCriteria, SearchQuery, PRISMAState, add_audit_entry
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


# ============================================================================
# OUTPUT MODELS
# ============================================================================

class PICOOutput(BaseModel):
    """Structured PICO extraction output."""
    population: str = Field(description="Target population or participants")
    intervention: str = Field(description="Intervention, exposure, or treatment being studied")
    comparator: str = Field(description="Comparison or control (if applicable)")
    outcome: str = Field(description="Primary outcomes of interest")
    study_types: List[str] = Field(description="Appropriate study types (e.g., RCT, cohort, meta-analysis)")
    reasoning: str = Field(description="Explanation of how PICO was derived from the research question")


class QueryOutput(BaseModel):
    """Structured search query output."""
    boolean_query: str = Field(description="Boolean search string with AND, OR, NOT operators")
    mesh_terms: List[str] = Field(description="Medical Subject Headings (MeSH) terms")
    keywords: List[str] = Field(description="Additional keywords and synonyms")
    database_specific_queries: Dict[str, str] = Field(description="Queries adapted for specific databases")
    rationale: str = Field(description="Explanation of query construction decisions")


# ============================================================================
# QUERY STRATEGIST AGENT
# ============================================================================

class QueryStrategistAgent:
    """
    Agent 1: Query Strategist
    
    Responsibilities:
    1. Extract PICO components from research question
    2. Identify relevant MeSH terms
    3. Construct Boolean search queries
    4. Adapt queries for different databases (PubMed, Semantic Scholar, etc.)
    5. Document search strategy with full transparency
    """
    
    def __init__(self):
        # Initialize LLM with low temperature for reproducibility
        self.llm = ChatOllama(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            base_url=settings.ollama_base_url
        )
        
        # Set up parsers
        self.pico_parser = PydanticOutputParser(pydantic_object=PICOOutput)
        self.query_parser = PydanticOutputParser(pydantic_object=QueryOutput)
    
    def extract_pico(self, research_question: str) -> PICOCriteria:
        """
        Extract PICO components from research question.
        
        Args:
            research_question: The systematic review research question
        
        Returns:
            PICOCriteria object with structured components
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert systematic review methodologist specializing in PICO framework extraction.
Your task is to analyze research questions and extract structured PICO components that will guide the literature search.

PICO Framework:
- P (Population): Who are the participants/patients? Include age, condition, setting if relevant.
- I (Intervention): What is the intervention, exposure, or treatment being studied?
- C (Comparator): What is it being compared to? (May be "none" for some studies)
- O (Outcome): What outcomes are we measuring? Include primary and key secondary outcomes.

Also identify appropriate study types (RCT, cohort, case-control, meta-analysis, etc.).

Be specific and comprehensive. This PICO will drive the entire search strategy.

{format_instructions}"""),
            ("user", "Research Question: {research_question}")
        ])
        
        chain = prompt | self.llm | self.pico_parser
        
        try:
            result = chain.invoke({
                "research_question": research_question,
                "format_instructions": self.pico_parser.get_format_instructions()
            })
            
            pico = PICOCriteria(
                population=result.population,
                intervention=result.intervention,
                comparator=result.comparator,
                outcome=result.outcome,
                study_types=result.study_types
            )
            
            logger.info(f"Extracted PICO: {pico.model_dump(mode='json')}")
            return pico
            
        except Exception as e:
            logger.error(f"PICO extraction failed: {e}")
            # Fallback: basic extraction
            return PICOCriteria(
                population="General population",
                intervention=research_question,
                comparator="Standard care or placebo",
                outcome="Primary outcomes related to the research question",
                study_types=["RCT", "Cohort Study", "Systematic Review"]
            )
    
    def generate_queries(
        self,
        research_question: str,
        pico: PICOCriteria,
        target_databases: List[str]
    ) -> List[SearchQuery]:
        """
        Generate comprehensive search queries with MeSH terms and Boolean logic.
        
        Args:
            research_question: Original research question
            pico: Extracted PICO criteria
            target_databases: List of databases to target (e.g., ['pubmed', 'semantic_scholar'])
        
        Returns:
            List of SearchQuery objects, one per database
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert medical librarian specializing in systematic review search strategies.

Your task is to create comprehensive, sensitive search queries that will retrieve all relevant literature while maintaining reasonable precision.

Guidelines:
1. Use Boolean operators (AND, OR, NOT) correctly
2. Include synonyms and related terms (use OR within concepts, AND between concepts)
3. Identify appropriate MeSH terms (Medical Subject Headings) for PubMed
4. Consider spelling variations, acronyms, and related terminology
5. Balance sensitivity (finding all relevant papers) with precision (avoiding too much noise)
6. Adapt syntax for different databases:
   - PubMed: Use [MeSH] tags, field tags [Title/Abstract], quotation marks for phrases
   - Semantic Scholar: Simpler boolean queries, natural language friendly
   - arXiv: Category-based + keyword queries
   - OpenAlex: Concept-based search

PICO Components:
Population: {population}
Intervention: {intervention}
Comparator: {comparator}
Outcome: {outcome}
Study Types: {study_types}

Target Databases: {databases}

{format_instructions}"""),
            ("user", "Research Question: {research_question}\n\nGenerate optimized search queries for each target database.")
        ])
        
        chain = prompt | self.llm | self.query_parser
        
        try:
            result = chain.invoke({
                "research_question": research_question,
                "population": pico.population,
                "intervention": pico.intervention,
                "comparator": pico.comparator,
                "outcome": pico.outcome,
                "study_types": ", ".join(pico.study_types),
                "databases": ", ".join(target_databases),
                "format_instructions": self.query_parser.get_format_instructions()
            })
            
            # Create SearchQuery objects
            queries = []
            
            # Main boolean query
            main_query = SearchQuery(
                query_id=str(uuid.uuid4()),
                boolean_query=result.boolean_query,
                mesh_terms=result.mesh_terms,
                keywords=result.keywords,
                databases=target_databases,
                rationale=result.rationale
            )
            queries.append(main_query)
            
            # Database-specific queries
            for db, db_query in result.database_specific_queries.items():
                if db in target_databases:
                    queries.append(SearchQuery(
                        query_id=str(uuid.uuid4()),
                        boolean_query=db_query,
                        mesh_terms=result.mesh_terms if db == "pubmed" else [],
                        keywords=result.keywords,
                        databases=[db],
                        rationale=f"Database-specific optimization for {db}"
                    ))
            
            logger.info(f"Generated {len(queries)} search queries")
            return queries
            
        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            # Fallback: simple query
            fallback_query = SearchQuery(
                query_id=str(uuid.uuid4()),
                boolean_query=research_question,
                mesh_terms=[],
                keywords=research_question.split(),
                databases=target_databases,
                rationale="Fallback simple query due to generation error"
            )
            return [fallback_query]
    
    def run(self, state: PRISMAState) -> Dict[str, Any]:
        """
        Execute the Query Strategist agent.
        
        Args:
            state: Current PRISMA workflow state
        
        Returns:
            State updates with PICO and search queries
        """
        logger.info("=== Query Strategist Agent Started ===")
        
        research_question = state["research_question"]
        target_databases = state.get("user_preferences", {}).get(
            "databases",
            ["pubmed", "semantic_scholar", "openalex"]
        )
        
        # Step 1: Extract PICO
        logger.info("Step 1: Extracting PICO components")
        pico = self.extract_pico(research_question)
        
        # Step 2: Generate search queries
        logger.info("Step 2: Generating search queries")
        queries = self.generate_queries(research_question, pico, target_databases)
        
        # Prepare state updates
        updates = {
            "pico_criteria": pico,
            "search_queries": queries,
            "current_stage": "query_strategy_complete"
        }
        
        # Add audit entry
        audit_entry = add_audit_entry(
            state,
            agent="QueryStrategist",
            action="generate_search_strategy",
            details={
                "pico": pico.model_dump(mode='json'),
                "queries_generated": len(queries),
                "queries": [q.model_dump(mode='json') for q in queries],
                "target_databases": target_databases
            }
        )
        updates.update(audit_entry)
        
        logger.info("=== Query Strategist Agent Completed ===")
        return updates


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == "__main__":
    # Test the agent
    from state import create_initial_state
    
    test_question = "What is the effectiveness of cognitive behavioral therapy for treating anxiety disorders in adults?"
    
    state = create_initial_state(
        research_question=test_question,
        user_preferences={"databases": ["pubmed", "semantic_scholar"]}
    )
    
    agent = QueryStrategistAgent()
    result = agent.run(state)
    
    print("\n=== PICO Extraction ===")
    print(result["pico_criteria"])
    
    print("\n=== Generated Queries ===")
    for i, query in enumerate(result["search_queries"], 1):
        print(f"\nQuery {i}:")
        print(f"  Boolean: {query.boolean_query}")
        print(f"  MeSH Terms: {query.mesh_terms}")
        print(f"  Databases: {query.databases}")
