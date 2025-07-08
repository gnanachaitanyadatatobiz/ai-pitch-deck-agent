"""
Ingestion Script for Extracted Text
Processes text files from the 'extracted_texts' folder and ingests them into the Pinecone vector database.
"""

import logging
from vector_database import VectorDatabase
from pinecone import Pinecone, ServerlessSpec
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def recreate_pinecone_index(index_name: str, dimension: int):
    """Deletes and recreates a Pinecone index to ensure correct dimensions."""
    logger.info(f"Attempting to recreate index '{index_name}' with dimension {dimension}...")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # Check if the index exists
    if index_name in [index.name for index in pc.list_indexes()]:
        logger.warning(f"Index '{index_name}' exists. Deleting it now to re-create with correct dimensions.")
        pc.delete_index(index_name)
        logger.info(f"Index '{index_name}' deleted successfully.")
        import time
        time.sleep(20) # Wait for deletion to propagate

    logger.info(f"Creating new index '{index_name}' with dimension {dimension}...")
    pc.create_index(
        name=index_name, 
        dimension=dimension, 
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    logger.info(f"Index '{index_name}' created successfully.")


def main():
    """Main function to run the ingestion process."""
    logger.info("🚀 Starting ingestion process for extracted text files...")
    
    try:
        # This is the correct dimension for 'text-embedding-3-small'
        required_dimension = 1536
        index_name = os.getenv("PINECONE_INDEX_NAME", "pitchdeck")

        # 1. Recreate the index to ensure the dimension is correct
        recreate_pinecone_index(index_name, required_dimension)

        # 2. Initialize VectorDatabase
        logger.info("Initializing vector database...")
        vector_db = VectorDatabase()
        
        # 3. Process extracted text files
        logger.info("Processing extracted text files from 'extracted_texts' folder...")
        documents_to_add = vector_db.process_extracted_text_files()
        
        if not documents_to_add:
            logger.warning("No new documents found to process. Exiting.")
            return

        # 4. Add documents to Pinecone
        logger.info(f"Adding {len(documents_to_add)} document chunks to Pinecone...")
        success = vector_db.add_documents(documents_to_add)
        
        if success:
            logger.info("✅ Ingestion process completed successfully!")
        else:
            logger.error("❌ Ingestion process failed.")
            
        # 5. Get database stats
        logger.info("Waiting for 30 seconds for Pinecone index to update stats...")
        import time
        time.sleep(30)
        
        logger.info("Fetching final database stats...")
        stats = vector_db.get_database_stats()
        logger.info(f"📊 Current database stats:")
        logger.info(f"   - Dimension: {stats.get('dimension')}")
        logger.info(f"   - Index Fullness: {stats.get('index_fullness')}")
        logger.info(f"   - Total Vectors: {stats.get('total_vectors')}")
        logger.info(f"   - Namespaces: {stats.get('namespaces')}")

    except Exception as e:
        logger.error(f"An error occurred during the ingestion process: {e}", exc_info=True)

if __name__ == "__main__":
    main() 