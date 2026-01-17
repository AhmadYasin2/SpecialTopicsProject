# AutoPRISMA: Automated PRISMA 2020 Systematic Literature Review System

A sophisticated multi-agent system that automates the PRISMA 2020 systematic literature review workflow with full traceability and auditability.

## 🎯 Overview

AutoPRISMA implements a complete PRISMA 2020-compliant workflow using a multi-agent architecture built with LangGraph. The system provides transparent, reproducible, and auditable systematic reviews.

## 🏗️ Architecture

### Multi-Agent System (6 Core + 1 Orchestrator)

1. **Query Strategist Agent** - PICO extraction, Boolean queries, MeSH terms, database-specific syntax
2. **Literature Retrieval Agent** - API integrations (Semantic Scholar, OpenAlex, PubMed, arXiv), deduplication
3. **Screening Criteria Agent** - Inclusion/exclusion rules, reproducible protocol
4. **Abstract Evaluator Agent** - Batch screening, confidence scores, bias detection, audit trail
5. **Synthesis & Analysis Agent** - Theme extraction, contradictions vs consensus, research gaps
6. **Report Generator Agent** - PRISMA flow diagram, tables, figures, multi-format export
7. **Orchestrator Agent** - State management, dependency control, error recovery

### Shared Knowledge Base

- Vector Store (ChromaDB/FAISS)
- Document Store (structured metadata)
- State Store (workflow checkpoints)
- Audit Trail (full provenance)

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Groq API key

### Installation

```bash
# Clone repository
cd SpecialTopicsProjecty

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env
# Edit .env with your Groq API key
```

### Running the System

#### Option 1: FastAPI Backend

```bash
# Start the API server
python main.py
# Access at http://localhost:8000
# API docs at http://localhost:8000/docs
```

#### Option 2: Streamlit UI

```bash
# Start the interactive UI
streamlit run app.py
# Access at http://localhost:8501
```

#### Option 3: Command Line

```bash
# Run the orchestrator directly
python orchestrator.py --query "Your research question here"
```

## 📁 Project Structure

```
SpecialTopicsProjecty/
├── config.py                      # Configuration management
├── state.py                       # Shared state and data models
├── audit_trail.py                 # Audit logging system
├── vector_store.py                # Vector store implementation
│
├── agents/
│   ├── agent_query_strategist.py      # Agent 1: Query formulation
│   ├── agent_literature_retrieval.py  # Agent 2: Paper retrieval
│   ├── agent_screening_criteria.py    # Agent 3: Protocol definition
│   ├── agent_abstract_evaluator.py    # Agent 4: Screening
│   ├── agent_synthesis_analysis.py    # Agent 5: Analysis
│   ├── agent_report_generator.py      # Agent 6: Report generation
│   └── orchestrator.py                # Agent 7: Coordination
│
├── main.py                        # FastAPI backend (uvicorn)
├── app.py                         # Streamlit UI
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## 🔧 Configuration

Edit `.env` file:

```env
# Required: Groq API Key
GROQ_API_KEY=gsk_...

# Optional: Academic API Keys (improves retrieval)
SEMANTIC_SCHOLAR_API_KEY=your_key
NCBI_API_KEY=your_key
OPENALEX_EMAIL=your@email.com
```

## 📊 Features

### ✅ PRISMA 2020 Compliance

- Structured search strategy with full documentation
- Transparent inclusion/exclusion criteria
- Reproducible screening with confidence scores
- Complete audit trail for every decision
- PRISMA flow diagram generation

### ✅ Multi-Agent Architecture

- Clean separation of concerns
- Stateful execution with checkpoints
- Human-in-the-loop intervention points
- Error recovery and retry logic

### ✅ Academic Data Integration

- Semantic Scholar API
- OpenAlex API
- PubMed/NCBI API
- arXiv API
- Automatic deduplication

### ✅ Explainability & Auditability

- Every decision is logged with reasoning
- Reproducible across runs (controlled temperature)
- Confidence scores for borderline cases
- Full provenance chain

### ✅ Output Formats

- PRISMA flow diagram (PNG/PDF)
- Structured report (DOCX/PDF)
- Data tables (XLSX/CSV)
- Citation exports (BibTeX/RIS)

## 🎓 Usage Examples

### Example 1: Basic Systematic Review

```python
from orchestrator import AutoPRISMAOrchestrator

orchestrator = AutoPRISMAOrchestrator()
result = await orchestrator.run_review(
    research_question="What is the effectiveness of cognitive behavioral therapy for anxiety disorders?",
    databases=["pubmed", "semantic_scholar"],
    date_range=("2015-01-01", "2024-12-31")
)

print(result.summary)
result.save_report("output/review_report.pdf")
```

### Example 2: With Human-in-the-Loop

```python
result = await orchestrator.run_review(
    research_question="...",
    enable_hitl=True,  # Pause for human review at key points
    hitl_checkpoints=["after_screening_criteria", "after_abstract_evaluation"]
)
```

## ⚠️ Important Notes

### This is a Research Prototype

- **NOT FOR PUBLICATION**: Results require expert validation
- **DEMONSTRATION ONLY**: Use for learning and prototyping
- **LLMs are fallible**: Always verify critical decisions
- **Requires expert oversight**: Medical/scientific reviews need domain experts

### Limitations

- LLM hallucinations possible
- API rate limits may slow retrieval
- Requires quality input data
- Not a replacement for human expertise

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# Test individual agents
pytest tests/test_query_strategist.py
pytest tests/test_literature_retrieval.py
```

## 📚 Technical Details

### State Management

The system uses LangGraph's StateGraph for deterministic workflow execution with:

- Checkpointing at every agent transition
- Rollback capability for error recovery
- Parallel execution where possible
- Human approval gates

### Prompt Engineering

Each agent uses carefully crafted prompts with:

- Clear role definition
- Structured output formats
- Few-shot examples
- Chain-of-thought reasoning
- Explicit error handling

### Vector Store

Embeddings are used for:

- Semantic similarity search
- Duplicate detection
- Theme clustering
- Citation network analysis

## 🤝 Contributing

This is a course project. For educational purposes only.

## 📄 License

Educational use only. Not for commercial deployment.

## 🙏 Acknowledgments

- PRISMA 2020 Guidelines
- LangGraph/LangChain frameworks
- Academic API providers

---

**Disclaimer**: This system is a demonstration prototype for educational purposes. All results must be validated by qualified domain experts before any real-world use. The system does not replace human expertise in systematic reviews.
