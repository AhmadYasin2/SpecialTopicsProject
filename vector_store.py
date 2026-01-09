"""
Vector store implementation for semantic search and duplicate detection.
Supports ChromaDB and FAISS backends.
"""
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from config import settings
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector store for semantic search and similarity detection.
    Used for:
    - Duplicate detection (semantic similarity)
    - Theme clustering
    - Related paper discovery
    - Citation network analysis
    """
    
    def __init__(self, collection_name: str = "prisma_papers"):
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        
        # Try ChromaDB first, fall back to FAISS
        self.backend = self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize vector store backend."""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            
            client = chromadb.Client(ChromaSettings(
                persist_directory=settings.vector_store_path,
                anonymized_telemetry=False,
            ))
            
            # Get or create collection
            self.collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PRISMA document embeddings"}
            )
            
            logger.info("Using ChromaDB as vector store backend")
            return "chromadb"
            
        except ImportError:
            logger.warning("ChromaDB not available, falling back to FAISS")
            return self._initialize_faiss()
    
    def _initialize_faiss(self):
        """Initialize FAISS backend as fallback."""
        try:
            import faiss
            
            # Create index
            self.index = faiss.IndexFlatL2(self.dimension)
            self.id_to_doc = {}  # Mapping from FAISS ID to document ID
            self.doc_to_id = {}  # Mapping from document ID to FAISS ID
            self.faiss_counter = 0
            
            # Try to load existing index
            index_path = Path(settings.vector_store_path) / f"{self.collection_name}.index"
            if index_path.exists():
                self.index = faiss.read_index(str(index_path))
                logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            
            logger.info("Using FAISS as vector store backend")
            return "faiss"
            
        except ImportError:
            logger.error("Neither ChromaDB nor FAISS is available!")
            return "none"
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: Optional[List[np.ndarray]] = None
    ):
        """
        Add documents to vector store.
        
        Args:
            documents: List of document dictionaries with 'id', 'text', and optional 'metadata'
            embeddings: Pre-computed embeddings (optional)
        """
        if not documents:
            return
        
        # Generate embeddings if not provided
        if embeddings is None:
            texts = [doc.get("text", "") for doc in documents]
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        
        if self.backend == "chromadb":
            self._add_to_chromadb(documents, embeddings)
        elif self.backend == "faiss":
            self._add_to_faiss(documents, embeddings)
        else:
            logger.warning("No vector store backend available")
    
    def _add_to_chromadb(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        """Add documents to ChromaDB."""
        ids = [doc["id"] for doc in documents]
        texts = [doc.get("text", "") for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(documents)} documents to ChromaDB")
    
    def _add_to_faiss(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        """Add documents to FAISS."""
        # Ensure embeddings are float32
        embeddings = embeddings.astype('float32')
        
        # Add to index
        start_id = self.faiss_counter
        self.index.add(embeddings)
        
        # Update mappings
        for i, doc in enumerate(documents):
            faiss_id = start_id + i
            doc_id = doc["id"]
            self.id_to_doc[faiss_id] = doc_id
            self.doc_to_id[doc_id] = faiss_id
        
        self.faiss_counter += len(documents)
        
        # Persist index
        self._save_faiss_index()
        
        logger.info(f"Added {len(documents)} documents to FAISS")
    
    def _save_faiss_index(self):
        """Persist FAISS index to disk."""
        try:
            import faiss
            index_path = Path(settings.vector_store_path) / f"{self.collection_name}.index"
            faiss.write_index(self.index, str(index_path))
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search for similar documents.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            threshold: Similarity threshold (optional)
        
        Returns:
            List of matching documents with scores
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        if self.backend == "chromadb":
            return self._search_chromadb(query_embedding, top_k)
        elif self.backend == "faiss":
            return self._search_faiss(query_embedding, top_k, threshold)
        else:
            return []
    
    def _search_chromadb(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Search in ChromaDB."""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        matches = []
        if results['ids'] and len(results['ids']) > 0:
            for i, doc_id in enumerate(results['ids'][0]):
                matches.append({
                    "id": doc_id,
                    "text": results['documents'][0][i] if results['documents'] else "",
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else 0.0,
                })
        
        return matches
    
    def _search_faiss(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Search in FAISS."""
        if self.index.ntotal == 0:
            return []
        
        # Ensure query is float32 and 2D
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        matches = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            if threshold is not None and distance > threshold:
                continue
            
            doc_id = self.id_to_doc.get(int(idx))
            if doc_id:
                matches.append({
                    "id": doc_id,
                    "distance": float(distance),
                })
        
        return matches
    
    def find_duplicates(
        self,
        documents: List[Dict[str, Any]],
        threshold: float = 0.95
    ) -> List[Tuple[str, str, float]]:
        """
        Find duplicate or near-duplicate documents.
        
        Args:
            documents: List of documents to check
            threshold: Similarity threshold (0-1, higher = more similar)
        
        Returns:
            List of (doc_id_1, doc_id_2, similarity) tuples
        """
        duplicates = []
        
        # Generate embeddings for all documents
        texts = [doc.get("text", "") for doc in documents]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        
        # Compute pairwise similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embeddings)
        
        # Find pairs above threshold (excluding self-similarity)
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                similarity = similarities[i, j]
                if similarity >= threshold:
                    duplicates.append((
                        documents[i]["id"],
                        documents[j]["id"],
                        float(similarity)
                    ))
        
        return duplicates
    
    def cluster_documents(
        self,
        document_ids: List[str],
        n_clusters: int = 5
    ) -> Dict[int, List[str]]:
        """
        Cluster documents for theme identification.
        
        Args:
            document_ids: List of document IDs to cluster
            n_clusters: Number of clusters
        
        Returns:
            Dictionary mapping cluster ID to list of document IDs
        """
        try:
            from sklearn.cluster import KMeans
            
            # Get embeddings for documents
            if self.backend == "chromadb":
                results = self.collection.get(ids=document_ids, include=["embeddings"])
                embeddings = np.array(results['embeddings'])
            elif self.backend == "faiss":
                # Retrieve from FAISS (simplified - in production, store embeddings separately)
                logger.warning("Clustering with FAISS requires separate embedding storage")
                return {0: document_ids}  # Fallback: all in one cluster
            else:
                return {0: document_ids}
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            
            # Group by cluster
            clusters = {}
            for doc_id, label in zip(document_ids, labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(doc_id)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return {0: document_ids}  # Fallback: all in one cluster
    
    def clear(self):
        """Clear all vectors from the store."""
        if self.backend == "chromadb":
            # Delete and recreate collection
            try:
                self.collection.delete()
            except:
                pass
        elif self.backend == "faiss":
            self.index.reset()
            self.id_to_doc.clear()
            self.doc_to_id.clear()
            self.faiss_counter = 0
            self._save_faiss_index()
