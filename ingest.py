"""
Data ingestion module for RAG chatbot application.
This module handles loading, processing, and storing documents for retrieval.
"""

import os
from typing import List
from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma


class DocumentIngestor:
    """Handles document ingestion and vector store creation."""
    
    def __init__(self, data_path: str = "./data", persist_directory: str = "./chroma_db"):
        """
        Initialize the document ingestor.
        
        Args:
            data_path: Path to the directory containing documents
            persist_directory: Path to store the vector database
        """
        self.data_path = data_path
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
    def load_documents(self) -> List:
        """
        Load documents from the data directory.
        
        Returns:
            List of loaded documents
        """
        print(f"Loading documents from {self.data_path}...")
        
        # Load text files
        text_loader = DirectoryLoader(
            self.data_path,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        
        # Load PDF files
        pdf_loader = DirectoryLoader(
            self.data_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        
        documents = []
        try:
            documents.extend(text_loader.load())
        except Exception as e:
            print(f"Error loading text files: {e}")
            
        try:
            documents.extend(pdf_loader.load())
        except Exception as e:
            print(f"Error loading PDF files: {e}")
        
        print(f"Loaded {len(documents)} documents")
        return documents
    
    def split_documents(self, documents: List, chunk_size: int = 1000, chunk_overlap: int = 200) -> List:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of documents to split
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of document chunks
        """
        print("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        return chunks
    
    def create_vector_store(self, chunks: List):
        """
        Create and persist a vector store from document chunks.
        
        Args:
            chunks: List of document chunks to embed
        """
        print("Creating vector store...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        vectorstore.persist()
        print(f"Vector store created and persisted at {self.persist_directory}")
        return vectorstore
    
    def ingest(self):
        """
        Execute the full ingestion pipeline.
        """
        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            print(f"Created data directory at {self.data_path}")
            print("Please add your documents to this directory and run again.")
            return
        
        # Load documents
        documents = self.load_documents()
        
        if not documents:
            print("No documents found. Please add documents to the data directory.")
            return
        
        # Split documents
        chunks = self.split_documents(documents)
        
        # Create vector store
        self.create_vector_store(chunks)
        
        print("Ingestion complete!")


def main():
    """Main function to run document ingestion."""
    ingestor = DocumentIngestor()
    ingestor.ingest()


if __name__ == "__main__":
    main()
