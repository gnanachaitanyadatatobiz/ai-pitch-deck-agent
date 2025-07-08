"""
Vector Database Module for Pinecone Integration
Handles vector storage, retrieval, and similarity search for pitch deck documents.
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from pinecone import Pinecone, ServerlessSpec
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import hashlib
import re
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDatabase:
    """Handles Pinecone vector database operations for pitch deck analysis."""
    
    def __init__(self):
        """Initialize the vector database with Pinecone and OpenAI embeddings."""
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "pitchdeck")
        
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        
        # Initialize OpenAI embeddings
        self.embedding_function = OpenAIEmbeddings(
            model="text-embedding-3-small",  # More cost-effective option
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.index = None

        # Initialize the index
        self._setup_index()
        
        # Initialize text splitter for extracted files
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", r"--- PAGE \d+ ---", " ", ""]
        )
    
    def _setup_index(self):
        """Set up the Pinecone index with proper configuration."""
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]

            if self.index_name not in existing_indexes:
                logger.info(f"Creating new index: {self.index_name}")

                # Create index with the new spec format
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,  # OpenAI text-embedding-3-small dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )

                # Wait for index to be ready
                logger.info("Waiting for index to be ready...")
                time.sleep(10)
            else:
                logger.info(f"Using existing index: {self.index_name}")

            # Connect to the index
            self.index = self.pc.Index(self.index_name)

            logger.info("Vector database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup Pinecone index: {e}")
            raise
    
    def _extract_company_name_from_filename(self, filename: str) -> str:
        """
        Extracts a clean company name from the filename.
        Example: 'uberpitchdeck-170823132244_extracted_20250703_115629.txt' -> 'uber'
        """
        match = re.match(r'([a-zA-Z0-9\-]+)', filename)
        if match:
            return match.group(1).split('-')[0].lower() # Always store as lowercase
        return "unknown"

    def process_extracted_text_files(self, text_folder: str = "extracted_texts") -> List[Document]:
        """
        Process all extracted .txt files from a folder.
        
        Args:
            text_folder: Path to folder containing extracted .txt files.

        Returns:
            List of LangChain Document objects ready for ingestion.
        """
        source_path = Path(text_folder)
        if not source_path.exists():
            logger.error(f"Source folder {source_path} does not exist.")
            return []
            
        txt_files = list(source_path.glob("*.txt"))
        all_documents = []

        logger.info(f"Found {len(txt_files)} to process from {source_path}.")

        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Split content into chunks
                chunks = self.text_splitter.split_text(content)
                
                # Extract metadata
                company_name = self._extract_company_name_from_filename(file_path.name)
                doc_id = hashlib.md5(str(file_path).encode()).hexdigest()

                for i, chunk in enumerate(chunks):
                    chunk_metadata = {
                        "company_name": company_name,
                        "file_name": file_path.name,
                        "file_type": "extracted_txt",
                        "document_id": doc_id,
                        "chunk_id": f"{doc_id}_{i}",
                        "chunk_index": i,
                    }
                    doc = Document(page_content=chunk, metadata=chunk_metadata)
                    all_documents.append(doc)
                
                logger.info(f"Processed {file_path.name}: created {len(chunks)} chunks.")

            except Exception as e:
                logger.error(f"Error processing file {file_path.name}: {e}")

        logger.info(f"Total documents created from text files: {len(all_documents)}")
        return all_documents

    def add_documents(self, documents: List[Document], batch_size: int = 100) -> bool:
        """
        Add documents to the vector database in batches.

        Args:
            documents: List of Document objects to add
            batch_size: Number of documents to process in each batch

        Returns:
            True if successful, False otherwise
        """
        try:
            if not documents:
                logger.warning("No documents to add")
                return False

            logger.info(f"Adding {len(documents)} documents to vector database...")

            # Process documents in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                vectors_to_upsert = []

                try:
                    # Prepare vectors for this batch
                    for j, doc in enumerate(batch):
                        # Generate embedding for the document
                        embedding = self.embedding_function.embed_query(doc.page_content)

                        # Create unique ID from metadata
                        doc_id = doc.metadata.get('chunk_id', f"chunk_{i}_{j}")

                        # Prepare metadata (Pinecone has limits on metadata size)
                        metadata = {
                            "text": doc.page_content[:2000],  # Limit text size
                            "company_name": doc.metadata.get('company_name', ''),
                            "file_name": doc.metadata.get('file_name', ''),
                            "file_type": doc.metadata.get('file_type', ''),
                            "chunk_index": doc.metadata.get('chunk_index', 0)
                        }

                        vectors_to_upsert.append({
                            "id": doc_id,
                            "values": embedding,
                            "metadata": metadata
                        })

                    # Upsert to Pinecone
                    self.index.upsert(vectors=vectors_to_upsert)
                    logger.info(f"Added batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")

                    # Small delay to avoid rate limits
                    time.sleep(2)

                except Exception as e:
                    logger.error(f"Error adding batch {i//batch_size + 1}: {e}")
                    continue

            logger.info("Successfully added all documents to vector database")
            return True

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    def search_similar_documents(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None) -> List[Document]:
        """
        Search for similar documents using semantic similarity.

        Args:
            query: Search query text
            k: Number of similar documents to return
            filter_dict: Optional metadata filters

        Returns:
            List of similar Document objects
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_function.embed_query(query)

            # Perform similarity search
            search_results = self.index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True,
                filter=filter_dict
            )

            # Convert results to Document objects
            documents = []
            for match in search_results.matches:
                metadata = match.metadata
                doc = Document(
                    page_content=metadata.get('text', ''),
                    metadata={
                        'company_name': metadata.get('company_name', ''),
                        'file_name': metadata.get('file_name', ''),
                        'file_type': metadata.get('file_type', ''),
                        'chunk_index': metadata.get('chunk_index', 0),
                        'score': match.score
                    }
                )
                documents.append(doc)

            logger.info(f"Found {len(documents)} similar documents for query: {query[:50]}...")
            return documents

        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []
    
    def search_by_query(self, query: str, k: int = 5) -> str:
        """
        Searches for documents by a general query string.
        """
        if self.index is None:
            logger.error("Vector database index is not initialized.")
            return "Error: Vector database not initialized."
        try:
            logger.info(f"Searching for query: {query}")
            query_embedding = self.embedding_function.embed_query(query)
            results = self.index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True
            )
            
            if not results['matches']:
                return "No relevant documents found for the query."

            context_str = "\n---\n".join([res['metadata']['text'] for res in results['matches']])
            return context_str
        except Exception as e:
            logger.error(f"Error during search_by_query: {e}")
            return f"An error occurred during search: {e}"

    def search_by_company(self, company_name: str, k: int = 5) -> str:
        """
        Searches for documents filtered by a specific company name.
        DEPRECATED: Use search_by_query for more general searches.
        """
        try:
            logger.info(f"Searching for company: {company_name}")
            # This is a simple implementation, assuming a generic query for the company
            query_text = f"pitch deck for {company_name}"
            query_embedding = self.embedding_function.embed_query(query_text)
            results = self.index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True
            )
            
            if not results['matches']:
                return "No relevant documents found for the company."

            context_str = "\n---\n".join([res['metadata']['text'] for res in results['matches']])
            return context_str
        except Exception as e:
            logger.error(f"Error during search_by_company: {e}")
            return f"An error occurred during search: {e}"
    
    def get_companies_list(self) -> List[str]:
        """
        Get a list of all companies in the database.

        Returns:
            List of company names
        """
        try:
            # For now, return empty list as Pinecone doesn't provide easy metadata enumeration
            # This would need to be maintained separately or queried differently
            logger.info("Company list retrieval not implemented yet")
            return []

        except Exception as e:
            logger.error(f"Error getting companies list: {e}")
            return []
    
    def check_company_exists(self, company_name: str) -> bool:
        """
        Check if a company exists in the database.
        
        Args:
            company_name: Name of the company to check
            
        Returns:
            True if company exists, False otherwise
        """
        try:
            results = self.search_by_company(company_name, k=1)
            return len(results) > 0
            
        except Exception as e:
            logger.error(f"Error checking if company exists: {e}")
            return False
    
    def get_company_document_count(self, company_name: str) -> int:
        """
        Gets the count of all document chunks for a specific company.
        
        Args:
            company_name: The name of the company to count documents for.
            
        Returns:
            The total number of document chunks found for the company.
        """
        try:
            # We query with a high 'k' to get all chunks for the company.
            documents = self.search_by_company(company_name, k=1000)
            return len(documents)
        except Exception as e:
            logger.error(f"Error getting document count for {company_name}: {e}")
            return 0

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            stats = self.index.describe_index_stats()
            
            return {
                "total_vectors": stats.get("total_vector_count", 0),
                "index_fullness": stats.get("index_fullness", 0),
                "dimension": stats.get("dimension", 0),
                "namespaces": stats.get("namespaces", {})
            }
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    def delete_all_vectors(self) -> bool:
        """
        Delete all vectors from the index (use with caution).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.warning(f"Deleting all vectors from index: {self.index_name}")
            self.index.delete(delete_all=True)
            logger.info("Successfully deleted all vectors.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete all vectors: {e}")
            return False
