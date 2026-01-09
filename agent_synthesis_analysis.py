"""
Agent 5: Synthesis & Analysis
Responsible for: Theme extraction, identifying contradictions vs consensus,
finding research gaps, and generating summary statistics.
"""
from typing import Dict, Any, List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import logging
from collections import Counter
from config import settings
from state import (
    Document, Theme, SynthesisResult,
    PRISMAState, add_audit_entry
)
from vector_store import VectorStore

logger = logging.getLogger(__name__)


# ============================================================================
# OUTPUT MODELS
# ============================================================================

class ThemeOutput(BaseModel):
    """Output model for theme extraction."""
    themes: List[Dict[str, Any]] = Field(
        description="List of identified themes with name, description, and strength"
    )


class GapsOutput(BaseModel):
    """Output model for research gaps."""
    research_gaps: List[str] = Field(
        description="List of identified research gaps"
    )


class ConsensusOutput(BaseModel):
    """Output model for consensus/contradiction analysis."""
    consensus_findings: List[str] = Field(
        description="Areas where studies agree"
    )
    contradictions: List[Dict[str, str]] = Field(
        description="Areas where studies contradict, with explanation"
    )


# ============================================================================
# SYNTHESIS & ANALYSIS AGENT
# ============================================================================

class SynthesisAnalysisAgent:
    """
    Agent 5: Synthesis & Analysis
    
    Responsibilities:
    1. Extract major themes from included papers
    2. Identify areas of consensus
    3. Identify contradictions and explain them
    4. Find research gaps
    5. Generate summary statistics
    6. Prepare synthesis for reporting
    """
    
    def __init__(self):
        self.llm = ChatGroq(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.groq_api_key
        )
        
        self.theme_parser = PydanticOutputParser(pydantic_object=ThemeOutput)
        self.gaps_parser = PydanticOutputParser(pydantic_object=GapsOutput)
        self.consensus_parser = PydanticOutputParser(pydantic_object=ConsensusOutput)
        
        # Initialize vector store for clustering
        self.vector_store = VectorStore(collection_name="synthesis_papers")
    
    def extract_themes(
        self,
        documents: List[Document],
        research_question: str
    ) -> List[Theme]:
        """
        Extract major research themes from included papers.
        Uses both LLM analysis and vector clustering.
        """
        logger.info(f"Extracting themes from {len(documents)} papers...")
        
        # Step 1: Use vector clustering to identify potential themes
        doc_texts = []
        for doc in documents:
            text = f"{doc.title}. {doc.abstract or ''}"
            doc_texts.append({
                "id": doc.id,
                "text": text,
                "metadata": {"title": doc.title, "year": doc.year}
            })
        
        # Add to vector store
        self.vector_store.add_documents(doc_texts)
        
        # Cluster papers
        n_clusters = min(5, len(documents) // 3) if len(documents) > 10 else 3
        clusters = self.vector_store.cluster_documents(
            [doc.id for doc in documents],
            n_clusters=n_clusters
        )
        
        logger.info(f"Identified {len(clusters)} potential theme clusters")
        
        # Step 2: Use LLM to analyze and name themes
        # Prepare paper summaries
        paper_summaries = []
        for i, doc in enumerate(documents[:20], 1):  # Limit to prevent context overflow
            summary = f"{i}. **{doc.title}** ({doc.year})\n"
            if doc.abstract:
                summary += f"   Abstract: {doc.abstract[:300]}...\n"
            paper_summaries.append(summary)
        
        papers_text = "\n".join(paper_summaries)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert systematic review synthesizer.

Your task is to identify the MAJOR THEMES in the included literature.

A good theme:
- Captures a meaningful pattern across multiple papers
- Is specific enough to be useful
- Is broad enough to encompass related findings
- Represents a distinct aspect of the research area

For each theme, provide:
- Descriptive name (3-7 words)
- Clear description (2-3 sentences)
- Strength assessment (strong/moderate/weak based on evidence)

Research Question: {research_question}

Number of papers: {n_papers}

Sample of included papers:
{papers}

{format_instructions}"""),
            ("user", "Identify the major research themes in this literature.")
        ])
        
        chain = prompt | self.llm | self.theme_parser
        
        try:
            result = chain.invoke({
                "research_question": research_question,
                "n_papers": len(documents),
                "papers": papers_text,
                "format_instructions": self.theme_parser.get_format_instructions()
            })
            
            # Convert to Theme objects
            themes = []
            for i, theme_data in enumerate(result.themes, 1):
                # Map cluster papers to theme
                cluster_id = i % len(clusters) if clusters else 0
                supporting_papers = clusters.get(cluster_id, [doc.id for doc in documents[:5]])
                
                theme = Theme(
                    theme_id=f"theme_{i}",
                    name=theme_data.get("name", f"Theme {i}"),
                    description=theme_data.get("description", ""),
                    supporting_papers=supporting_papers[:10],  # Limit
                    strength=theme_data.get("strength", "moderate")
                )
                themes.append(theme)
            
            logger.info(f"Extracted {len(themes)} themes")
            return themes
            
        except Exception as e:
            logger.error(f"Theme extraction failed: {e}")
            return []
    
    def identify_consensus_and_contradictions(
        self,
        documents: List[Document],
        research_question: str
    ) -> tuple[List[str], List[Dict[str, str]]]:
        """
        Identify areas of consensus and contradiction in the literature.
        """
        logger.info("Analyzing consensus and contradictions...")
        
        # Prepare paper summaries
        paper_summaries = []
        for i, doc in enumerate(documents[:15], 1):
            summary = f"{i}. {doc.title} ({doc.year}): {doc.abstract[:200] if doc.abstract else 'No abstract'}..."
            paper_summaries.append(summary)
        
        papers_text = "\n".join(paper_summaries)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at synthesizing systematic review findings.

Your task is to identify:
1. **CONSENSUS**: Where multiple studies agree on findings
2. **CONTRADICTIONS**: Where studies reach different conclusions

For consensus findings:
- State the agreed-upon finding clearly
- Note how many/which studies support it

For contradictions:
- Clearly describe what studies disagree about
- Propose potential explanations (methodology, population, etc.)

Research Question: {research_question}

Included papers:
{papers}

{format_instructions}"""),
            ("user", "Analyze these papers for consensus and contradictions.")
        ])
        
        chain = prompt | self.llm | self.consensus_parser
        
        try:
            result = chain.invoke({
                "research_question": research_question,
                "papers": papers_text,
                "format_instructions": self.consensus_parser.get_format_instructions()
            })
            
            return result.consensus_findings, result.contradictions
            
        except Exception as e:
            logger.error(f"Consensus analysis failed: {e}")
            return [], []
    
    def identify_research_gaps(
        self,
        documents: List[Document],
        research_question: str,
        themes: List[Theme]
    ) -> List[str]:
        """
        Identify gaps in the current research literature.
        """
        logger.info("Identifying research gaps...")
        
        themes_text = "\n".join([f"- {t.name}: {t.description}" for t in themes])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at identifying research gaps in systematic reviews.

Your task is to identify what is MISSING or UNDEREXPLORED in the current literature.

Good research gaps:
- Are specific and actionable
- Represent genuine knowledge needs
- Could guide future research
- Are derived from what IS present in the literature

Consider:
- Populations not studied
- Interventions not compared
- Outcomes not measured
- Methodological limitations
- Contexts not explored

Research Question: {research_question}

Number of studies: {n_papers}

Identified themes:
{themes}

{format_instructions}"""),
            ("user", "Identify key research gaps that future studies should address.")
        ])
        
        chain = prompt | self.llm | self.gaps_parser
        
        try:
            result = chain.invoke({
                "research_question": research_question,
                "n_papers": len(documents),
                "themes": themes_text,
                "format_instructions": self.gaps_parser.get_format_instructions()
            })
            
            return result.research_gaps
            
        except Exception as e:
            logger.error(f"Gap identification failed: {e}")
            return []
    
    def generate_summary_statistics(
        self,
        documents: List[Document]
    ) -> Dict[str, Any]:
        """Generate descriptive statistics about included papers."""
        stats = {
            "total_papers": len(documents),
            "year_range": None,
            "year_distribution": {},
            "top_journals": [],
            "top_authors": [],
            "papers_with_abstracts": 0,
            "average_authors_per_paper": 0,
        }
        
        if not documents:
            return stats
        
        # Year distribution
        years = [doc.year for doc in documents if doc.year]
        if years:
            stats["year_range"] = (min(years), max(years))
            year_counts = Counter(years)
            stats["year_distribution"] = dict(year_counts.most_common())
        
        # Journal distribution
        journals = [doc.journal for doc in documents if doc.journal]
        journal_counts = Counter(journals)
        stats["top_journals"] = journal_counts.most_common(10)
        
        # Author statistics
        all_authors = []
        for doc in documents:
            all_authors.extend([author.name for author in doc.authors])
        
        author_counts = Counter(all_authors)
        stats["top_authors"] = author_counts.most_common(10)
        stats["average_authors_per_paper"] = (
            sum(len(doc.authors) for doc in documents) / len(documents)
            if documents else 0
        )
        
        # Abstract availability
        stats["papers_with_abstracts"] = sum(
            1 for doc in documents if doc.abstract
        )
        
        return stats
    
    def run(self, state: PRISMAState) -> Dict[str, Any]:
        """
        Execute the Synthesis & Analysis agent.
        
        Args:
            state: Current PRISMA workflow state
        
        Returns:
            State updates with synthesis results
        """
        logger.info("=== Synthesis & Analysis Agent Started ===")
        
        included_docs = state.get("included_documents", [])
        research_question = state["research_question"]
        
        if not included_docs:
            logger.error("No included documents for synthesis")
            return {"error_message": "No included papers available for synthesis"}
        
        logger.info(f"Synthesizing {len(included_docs)} included papers...")
        
        # Step 1: Extract themes
        logger.info("Step 1: Extracting themes")
        themes = self.extract_themes(included_docs, research_question)
        
        # Step 2: Consensus and contradictions
        logger.info("Step 2: Analyzing consensus and contradictions")
        consensus, contradictions = self.identify_consensus_and_contradictions(
            included_docs, research_question
        )
        
        # Step 3: Research gaps
        logger.info("Step 3: Identifying research gaps")
        gaps = self.identify_research_gaps(included_docs, research_question, themes)
        
        # Step 4: Summary statistics
        logger.info("Step 4: Generating summary statistics")
        stats = self.generate_summary_statistics(included_docs)
        
        # Create synthesis result
        synthesis = SynthesisResult(
            themes=themes,
            research_gaps=gaps,
            contradictions=contradictions,
            consensus_findings=consensus,
            summary_statistics=stats
        )
        
        # Prepare state updates
        updates = {
            "synthesis_result": synthesis,
            "current_stage": "synthesis_complete"
        }
        
        # Add audit entry
        audit_entry = add_audit_entry(
            state,
            agent="SynthesisAnalysis",
            action="synthesize_literature",
            details={
                "papers_analyzed": len(included_docs),
                "themes_identified": len(themes),
                "consensus_findings": len(consensus),
                "contradictions": len(contradictions),
                "research_gaps": len(gaps),
                "summary_statistics": stats
            }
        )
        updates.update(audit_entry)
        
        logger.info("=== Synthesis & Analysis Agent Completed ===")
        return updates


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == "__main__":
    from state import create_initial_state, Author
    
    # Create test documents
    test_docs = [
        Document(
            id=f"test{i}",
            title=f"CBT Study {i}: {['Effectiveness', 'Mechanisms', 'Long-term outcomes'][i % 3]}",
            authors=[Author(name=f"Author {i}")],
            abstract=f"This study examined cognitive behavioral therapy for anxiety disorders. Sample size: {50 + i * 10}.",
            year=2018 + i % 5,
            journal=["Journal A", "Journal B", "Journal C"][i % 3],
            source="test"
        )
        for i in range(10)
    ]
    
    state = create_initial_state(
        research_question="Effectiveness of CBT for anxiety"
    )
    state["included_documents"] = test_docs
    
    agent = SynthesisAnalysisAgent()
    result = agent.run(state)
    
    synthesis = result["synthesis_result"]
    
    print("\n=== THEMES ===")
    for theme in synthesis.themes:
        print(f"\n{theme.name} ({theme.strength})")
        print(f"  {theme.description}")
        print(f"  Supporting papers: {len(theme.supporting_papers)}")
    
    print("\n=== CONSENSUS FINDINGS ===")
    for finding in synthesis.consensus_findings:
        print(f"- {finding}")
    
    print("\n=== RESEARCH GAPS ===")
    for gap in synthesis.research_gaps:
        print(f"- {gap}")
