"""
Agent 2: Literature Retrieval
Responsible for: API integrations with academic databases, paper retrieval,
deduplication, and metadata extraction.
"""
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import hashlib
import asyncio
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential
from config import settings
from state import Document, Author, SearchQuery, PRISMAState, add_audit_entry

logger = logging.getLogger(__name__)


# ============================================================================
# DATABASE CLIENTS
# ============================================================================

class SemanticScholarClient:
    """Client for Semantic Scholar API."""
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.semantic_scholar_api_key
        self.session = None
    
    async def __aenter__(self):
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        self.session = aiohttp.ClientSession(headers=headers)
        return self
    
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def search_papers(
        self,
        query: str,
        limit: int = 100,
        year_range: Optional[tuple] = None
    ) -> List[Dict[str, Any]]:
        """Search papers using Semantic Scholar API."""
        params = {
            "query": query,
            "limit": min(limit, 100),  # API limit per request
            "fields": "paperId,title,abstract,authors,year,venue,citationCount,url,externalIds"
        }
        
        if year_range:
            params["year"] = f"{year_range[0]}-{year_range[1]}"
        
        try:
            async with self.session.get(f"{self.BASE_URL}/paper/search", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [])
                else:
                    logger.error(f"Semantic Scholar API error: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {e}")
            return []


class PubMedClient:
    """Client for PubMed/NCBI E-utilities API."""
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.ncbi_api_key
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def search_papers(
        self,
        query: str,
        limit: int = 100,
        year_range: Optional[tuple] = None
    ) -> List[Dict[str, Any]]:
        """Search papers using PubMed API."""
        # Step 1: Search for PMIDs
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": limit,
            "retmode": "json"
        }
        
        if self.api_key:
            search_params["api_key"] = self.api_key
        
        if year_range:
            search_params["term"] += f" AND {year_range[0]}:{year_range[1]}[pdat]"
        
        try:
            async with self.session.get(f"{self.BASE_URL}/esearch.fcgi", params=search_params) as response:
                if response.status != 200:
                    logger.error(f"PubMed search error: {response.status}")
                    return []
                
                search_result = await response.json()
                pmids = search_result.get("esearchresult", {}).get("idlist", [])
                
                if not pmids:
                    return []
                
                # Step 2: Fetch details for PMIDs
                return await self._fetch_details(pmids)
                
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return []
    
    async def _fetch_details(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """Fetch paper details for given PMIDs."""
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml"
        }
        
        if self.api_key:
            fetch_params["api_key"] = self.api_key
        
        try:
            async with self.session.get(f"{self.BASE_URL}/efetch.fcgi", params=fetch_params) as response:
                if response.status == 200:
                    xml_data = await response.text()
                    return self._parse_pubmed_xml(xml_data, pmids)
                return []
        except Exception as e:
            logger.error(f"PubMed fetch failed: {e}")
            return []
    
    def _parse_pubmed_xml(self, xml_data: str, pmids: List[str]) -> List[Dict[str, Any]]:
        """Parse PubMed XML response (simplified)."""
        import xml.etree.ElementTree as ET
        
        papers = []
        try:
            root = ET.fromstring(xml_data)
            for article in root.findall(".//PubmedArticle"):
                try:
                    medline = article.find(".//MedlineCitation")
                    pmid = medline.find(".//PMID").text if medline.find(".//PMID") is not None else None
                    
                    article_elem = medline.find(".//Article")
                    if article_elem is None:
                        continue
                    
                    title_elem = article_elem.find(".//ArticleTitle")
                    title = title_elem.text if title_elem is not None else "No title"
                    
                    abstract_elem = article_elem.find(".//Abstract/AbstractText")
                    abstract = abstract_elem.text if abstract_elem is not None else None
                    
                    # Extract authors
                    authors = []
                    for author in article_elem.findall(".//Author"):
                        lastname = author.find("LastName")
                        forename = author.find("ForeName")
                        if lastname is not None and forename is not None:
                            authors.append({
                                "name": f"{forename.text} {lastname.text}",
                                "affiliation": None
                            })
                    
                    # Extract year
                    year_elem = article_elem.find(".//PubDate/Year")
                    year = int(year_elem.text) if year_elem is not None else None
                    
                    # Extract journal
                    journal_elem = article_elem.find(".//Journal/Title")
                    journal = journal_elem.text if journal_elem is not None else None
                    
                    papers.append({
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract,
                        "authors": authors,
                        "year": year,
                        "journal": journal,
                        "doi": None,  # Would need to extract from ArticleId
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to parse article: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"XML parsing failed: {e}")
        
        return papers


class ArXivClient:
    """Client for arXiv API."""
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    async def search_papers(
        self,
        query: str,
        limit: int = 100,
        year_range: Optional[tuple] = None
    ) -> List[Dict[str, Any]]:
        """Search papers using arXiv API."""
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.BASE_URL, params=params) as response:
                    if response.status == 200:
                        xml_data = await response.text()
                        return self._parse_arxiv_xml(xml_data)
                    return []
        except Exception as e:
            logger.error(f"arXiv search failed: {e}")
            return []
    
    def _parse_arxiv_xml(self, xml_data: str) -> List[Dict[str, Any]]:
        """Parse arXiv XML response."""
        import xml.etree.ElementTree as ET
        
        papers = []
        try:
            root = ET.fromstring(xml_data)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            
            for entry in root.findall("atom:entry", ns):
                try:
                    title = entry.find("atom:title", ns).text.strip()
                    abstract = entry.find("atom:summary", ns).text.strip()
                    
                    authors = []
                    for author in entry.findall("atom:author", ns):
                        name = author.find("atom:name", ns).text
                        authors.append({"name": name, "affiliation": None})
                    
                    published = entry.find("atom:published", ns).text
                    year = int(published.split("-")[0]) if published else None
                    
                    url = entry.find("atom:id", ns).text
                    arxiv_id = url.split("/")[-1]
                    
                    papers.append({
                        "arxiv_id": arxiv_id,
                        "title": title,
                        "abstract": abstract,
                        "authors": authors,
                        "year": year,
                        "journal": "arXiv",
                        "doi": None,
                        "url": url
                    })
                except Exception as e:
                    logger.warning(f"Failed to parse arXiv entry: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"arXiv XML parsing failed: {e}")
        
        return papers


class OpenAlexClient:
    """Client for OpenAlex API."""
    
    BASE_URL = "https://api.openalex.org"
    
    def __init__(self, email: Optional[str] = None):
        self.email = email or settings.openalex_email
    
    async def search_papers(
        self,
        query: str,
        limit: int = 100,
        year_range: Optional[tuple] = None
    ) -> List[Dict[str, Any]]:
        """Search papers using OpenAlex API."""
        params = {
            "search": query,
            "per_page": min(limit, 200),
            "mailto": self.email
        }
        
        if year_range:
            params["filter"] = f"publication_year:{year_range[0]}-{year_range[1]}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.BASE_URL}/works", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_openalex_response(data)
                    return []
        except Exception as e:
            logger.error(f"OpenAlex search failed: {e}")
            return []
    
    def _parse_openalex_response(self, data: Dict) -> List[Dict[str, Any]]:
        """Parse OpenAlex API response."""
        papers = []
        
        for work in data.get("results", []):
            try:
                authors = []
                for authorship in work.get("authorships", []):
                    author_info = authorship.get("author", {})
                    authors.append({
                        "name": author_info.get("display_name", "Unknown"),
                        "affiliation": None
                    })
                
                papers.append({
                    "openalex_id": work.get("id"),
                    "title": work.get("title", "No title"),
                    "abstract": work.get("abstract"),
                    "authors": authors,
                    "year": work.get("publication_year"),
                    "journal": work.get("primary_location", {}).get("source", {}).get("display_name"),
                    "doi": work.get("doi"),
                    "url": work.get("id")
                })
            except Exception as e:
                logger.warning(f"Failed to parse OpenAlex work: {e}")
                continue
        
        return papers


# ============================================================================
# LITERATURE RETRIEVAL AGENT
# ============================================================================

class LiteratureRetrievalAgent:
    """
    Agent 2: Literature Retrieval
    
    Responsibilities:
    1. Execute searches across multiple academic databases
    2. Handle API rate limits and errors gracefully
    3. Deduplicate papers using multiple strategies
    4. Extract and normalize metadata
    5. Track retrieval statistics
    """
    
    def __init__(self):
        self.clients = {
            "semantic_scholar": SemanticScholarClient(),
            "pubmed": PubMedClient(),
            "arxiv": ArXivClient(),
            "openalex": OpenAlexClient()
        }
    
    async def retrieve_from_database(
        self,
        database: str,
        query: str,
        limit: int = 100,
        year_range: Optional[tuple] = None
    ) -> List[Document]:
        """Retrieve papers from a specific database."""
        logger.info(f"Retrieving from {database} with query: {query[:100]}...")
        
        client = self.clients.get(database)
        if not client:
            logger.error(f"Unknown database: {database}")
            return []
        
        # Simplify query for databases that don't support complex Boolean syntax
        simplified_query = self._simplify_query_for_database(query, database)
        logger.info(f"Simplified query for {database}: {simplified_query[:100]}...")
        
        # Use context manager for clients that support it
        if hasattr(client, "__aenter__"):
            async with client as c:
                raw_papers = await c.search_papers(simplified_query, limit, year_range)
        else:
            raw_papers = await client.search_papers(simplified_query, limit, year_range)
        
        # If no results and database supports simple queries, try with just key terms
        if len(raw_papers) == 0 and database in ["semantic_scholar", "arxiv"]:
            fallback_query = self._extract_key_terms(simplified_query)
            if fallback_query != simplified_query and len(fallback_query) > 0:
                logger.info(f"No results with simplified query, trying key terms: {fallback_query}")
                if hasattr(client, "__aenter__"):
                    async with client as c:
                        raw_papers = await c.search_papers(fallback_query, limit, year_range)
                else:
                    raw_papers = await client.search_papers(fallback_query, limit, year_range)
        
        # Convert to Document objects
        documents = []
        for paper in raw_papers:
            doc = self._convert_to_document(paper, database)
            if doc:
                documents.append(doc)
        
        logger.info(f"Retrieved {len(documents)} papers from {database}")
        return documents
    
    def _simplify_query_for_database(self, query: str, database: str) -> str:
        """Simplify complex Boolean queries for databases with limited search syntax."""
        if database in ["semantic_scholar", "arxiv"]:
            # These APIs don't support Boolean operators - they do simple keyword search
            # Just extract the meaningful keywords from the complex query
            
            # Remove all Boolean operators and special syntax
            simplified = query
            simplified = simplified.replace(" AND ", " ")
            simplified = simplified.replace(" OR ", " ")
            simplified = simplified.replace(" NOT ", " ")
            
            # Remove all parentheses, quotes, brackets
            for char in ['(', ')', '"', "'", '[', ']', '{', '}']:
                simplified = simplified.replace(char, " ")
            
            # Remove wildcards and special characters
            for char in ['*', '?', ':', ';']:
                simplified = simplified.replace(char, " ")
            
            # Remove category prefixes for arXiv (cat:cs.SE, etc.)
            words = simplified.split()
            cleaned_words = [w for w in words if not w.startswith('cat')]
            simplified = " ".join(cleaned_words)
            
            # Clean up extra spaces and convert to lowercase for consistency
            simplified = " ".join(simplified.split()).strip()
            
            # For very long queries, extract just the key terms
            if len(simplified.split()) > 10:
                simplified = self._extract_key_terms(simplified)
            
            logger.info(f"Query simplification: '{query[:60]}...' → '{simplified[:60]}...'")
            return simplified
        
        return query
    
    def _extract_key_terms(self, query: str) -> str:
        """Extract the most important keywords from a query for fallback search."""
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                      'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                      'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'should', 'could', 'may', 'might', 'must', 'can'}
        
        # Split into words and filter
        words = query.lower().split()
        key_terms = [w for w in words if len(w) > 2 and w not in stop_words]
        
        # Take most significant terms (first 5-6 unique terms)
        unique_terms = []
        for term in key_terms:
            if term not in unique_terms:
                unique_terms.append(term)
            if len(unique_terms) >= 6:
                break
        
        return " ".join(unique_terms)
    
    def _convert_to_document(self, paper: Dict[str, Any], source: str) -> Optional[Document]:
        """Convert raw API response to Document object."""
        try:
            # Generate unique ID
            id_fields = [
                paper.get("pmid"),
                paper.get("paperId"),
                paper.get("arxiv_id"),
                paper.get("openalex_id"),
                paper.get("doi"),
                paper.get("title")
            ]
            id_string = "_".join([str(f) for f in id_fields if f])
            doc_id = hashlib.md5(id_string.encode()).hexdigest()
            
            # Parse authors
            authors = []
            for author_data in paper.get("authors", []):
                if isinstance(author_data, dict):
                    authors.append(Author(
                        name=author_data.get("name", "Unknown"),
                        affiliation=author_data.get("affiliation")
                    ))
            
            return Document(
                id=doc_id,
                title=paper.get("title", "No title"),
                authors=authors,
                abstract=paper.get("abstract"),
                year=paper.get("year"),
                journal=paper.get("journal"),
                doi=paper.get("doi"),
                url=paper.get("url"),
                keywords=[],
                source=source,
                retrieved_at=datetime.now()
            )
        
        except Exception as e:
            logger.error(f"Failed to convert paper to Document: {e}")
            return None
    
    def deduplicate_documents(self, documents: List[Document]) -> tuple[List[Document], int]:
        """
        Deduplicate papers using multiple strategies.
        
        Returns:
            Tuple of (unique_documents, num_duplicates_removed)
        """
        logger.info(f"Deduplicating {len(documents)} documents...")
        
        seen_titles = set()
        seen_dois = set()
        unique_docs = []
        
        for doc in documents:
            # Strategy 1: DOI-based deduplication
            if doc.doi:
                doi_normalized = doc.doi.lower().strip()
                if doi_normalized in seen_dois:
                    continue
                seen_dois.add(doi_normalized)
            
            # Strategy 2: Title-based deduplication (normalized)
            title_normalized = doc.title.lower().strip()
            title_normalized = "".join(c for c in title_normalized if c.isalnum() or c.isspace())
            
            if title_normalized in seen_titles:
                continue
            seen_titles.add(title_normalized)
            
            unique_docs.append(doc)
        
        num_duplicates = len(documents) - len(unique_docs)
        logger.info(f"Removed {num_duplicates} duplicates, {len(unique_docs)} unique papers remain")
        
        return unique_docs, num_duplicates
    
    async def run(self, state: PRISMAState) -> Dict[str, Any]:
        """
        Execute the Literature Retrieval agent.
        
        Args:
            state: Current PRISMA workflow state
        
        Returns:
            State updates with retrieved documents
        """
        logger.info("=== Literature Retrieval Agent Started ===")
        
        search_queries = state.get("search_queries", [])
        if not search_queries:
            logger.error("No search queries found in state")
            return {"error_message": "No search queries available"}
        
        # Extract date range from user preferences
        date_range = state.get("user_preferences", {}).get("date_range")
        year_range = None
        if date_range:
            year_range = (
                int(date_range[0].split("-")[0]),
                int(date_range[1].split("-")[0])
            )
        
        # Retrieve from all databases
        all_documents = []
        retrieval_stats = {}
        
        for query in search_queries:
            for database in query.databases:
                try:
                    docs = await self.retrieve_from_database(
                        database=database,
                        query=query.boolean_query,
                        limit=settings.max_papers_per_query,
                        year_range=year_range
                    )
                    
                    all_documents.extend(docs)
                    retrieval_stats[database] = retrieval_stats.get(database, 0) + len(docs)
                    
                except Exception as e:
                    logger.error(f"Retrieval from {database} failed: {e}")
                    retrieval_stats[f"{database}_error"] = str(e)
        
        # Deduplicate
        unique_documents, num_duplicates = self.deduplicate_documents(all_documents)
        retrieval_stats["total_retrieved"] = len(all_documents)
        retrieval_stats["duplicates_removed"] = num_duplicates
        retrieval_stats["unique_documents"] = len(unique_documents)
        
        # Prepare state updates
        updates = {
            "retrieved_documents": unique_documents,
            "retrieval_stats": retrieval_stats,
            "current_stage": "literature_retrieval_complete"
        }
        
        # Add audit entry
        audit_entry = add_audit_entry(
            state,
            agent="LiteratureRetrieval",
            action="retrieve_papers",
            details=retrieval_stats
        )
        updates.update(audit_entry)
        
        logger.info("=== Literature Retrieval Agent Completed ===")
        return updates


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == "__main__":
    import asyncio
    from state import create_initial_state, SearchQuery
    import uuid
    
    # Create test state
    test_query = SearchQuery(
        query_id=str(uuid.uuid4()),
        boolean_query="cognitive behavioral therapy AND anxiety",
        mesh_terms=["Cognitive Behavioral Therapy", "Anxiety Disorders"],
        keywords=["CBT", "anxiety", "treatment"],
        databases=["semantic_scholar", "arxiv"]
    )
    
    state = create_initial_state(
        research_question="CBT for anxiety",
        user_preferences={"databases": ["semantic_scholar", "arxiv"]}
    )
    state["search_queries"] = [test_query]
    
    # Run agent
    agent = LiteratureRetrievalAgent()
    result = asyncio.run(agent.run(state))
    
    print(f"\n=== Retrieval Stats ===")
    print(result["retrieval_stats"])
    print(f"\n=== Sample Papers ===")
    for doc in result["retrieved_documents"][:3]:
        print(f"\nTitle: {doc.title}")
        print(f"Source: {doc.source}")
        print(f"Year: {doc.year}")
