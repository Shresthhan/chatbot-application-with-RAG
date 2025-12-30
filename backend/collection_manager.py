"""
collection_manager.py - Manages Qdrant collections and their metadata
Handles collection registry, metadata storage, and discovery
"""

import json
import os
from typing import Dict, List, Optional
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from pathlib import Path

# Paths
QDRANT_PATH = "./Qdrant_DB"
COLLECTIONS_CONFIG_PATH = "./collections_config.json"

class CollectionManager:
    """Manages Qdrant collections and their metadata"""
    
    def __init__(self, qdrant_path: str = QDRANT_PATH):
        self.qdrant_path = qdrant_path
        self.config_path = COLLECTIONS_CONFIG_PATH
        self.client = None
        self._initialize_client()
        self._ensure_config_file()
    
    def _initialize_client(self):
        """Initialize Qdrant client"""
        from backend.ingest import get_qdrant_client
        self.client = get_qdrant_client()
    
    def _ensure_config_file(self):
        """Create collections config file if it doesn't exist"""
        if not os.path.exists(self.config_path):
            self._save_config({})
            print(f"✓ Created collections config at {self.config_path}")
    
    def _load_config(self) -> Dict:
        """Load collections configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_config(self, config: Dict):
        """Save collections configuration to JSON file"""
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def list_collections(self) -> List[str]:
        """Get list of all collection names from Qdrant"""
        try:
            collections = self.client.get_collections()
            return [col.name for col in collections.collections]
        except Exception as e:
            print(f"Error listing collections: {e}")
            return []
    
    def get_collection_metadata(self, collection_name: str) -> Optional[Dict]:
        """Get metadata for a specific collection"""
        config = self._load_config()
        return config.get(collection_name)
    
    def get_all_collections_metadata(self) -> Dict[str, Dict]:
        """Get metadata for all collections"""
        config = self._load_config()
        # Sync with actual Qdrant collections
        actual_collections = self.list_collections()
        
        # Add missing collections with default metadata
        for col_name in actual_collections:
            if col_name not in config:
                config[col_name] = {
                    "description": f"Collection: {col_name}",
                    "created_at": datetime.now().isoformat(),
                    "document_count": 0,
                    "last_updated": datetime.now().isoformat()
                }
        
        # Remove deleted collections
        for col_name in list(config.keys()):
            if col_name not in actual_collections:
                del config[col_name]
        
        self._save_config(config)
        return config
    
    def create_collection(
        self, 
        collection_name: str, 
        description: str,
        vector_size: int = 768,  # Default for all-mpnet-base-v2
        distance: Distance = Distance.COSINE
    ) -> bool:
        """
        Create a new collection in Qdrant with metadata
        
        Args:
            collection_name: Name of the collection
            description: Human-readable description (used for tool description)
            vector_size: Dimension of embedding vectors
            distance: Distance metric (COSINE, EUCLID, DOT)
        
        Returns:
            True if created successfully, False otherwise
        """
        try:
            # Check if collection already exists
            existing = self.list_collections()
            if collection_name in existing:
                print(f"⚠ Collection '{collection_name}' already exists")
                return False
            
            # Create collection in Qdrant
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                )
            )
            
            # Save metadata
            config = self._load_config()
            config[collection_name] = {
                "description": description,
                "created_at": datetime.now().isoformat(),
                "document_count": 0,
                "last_updated": datetime.now().isoformat(),
                "vector_size": vector_size,
                "distance": distance.name
            }
            self._save_config(config)
            
            print(f"✓ Created collection '{collection_name}'")
            return True
            
        except Exception as e:
            print(f"Error creating collection: {e}")
            return False
    
    def update_collection_metadata(
        self, 
        collection_name: str, 
        description: Optional[str] = None,
        document_count: Optional[int] = None
    ):
        """Update metadata for an existing collection"""
        config = self._load_config()
        
        if collection_name not in config:
            print(f"⚠ Collection '{collection_name}' not found in config")
            return False
        
        if description:
            config[collection_name]["description"] = description
        
        if document_count is not None:
            config[collection_name]["document_count"] = document_count
        
        config[collection_name]["last_updated"] = datetime.now().isoformat()
        
        self._save_config(config)
        print(f"✓ Updated metadata for '{collection_name}'")
        return True
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection from Qdrant and remove its metadata"""
        try:
            # Delete from Qdrant
            self.client.delete_collection(collection_name=collection_name)
            
            # Remove from config
            config = self._load_config()
            if collection_name in config:
                del config[collection_name]
                self._save_config(config)
            
            print(f"✓ Deleted collection '{collection_name}'")
            return True
            
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False
    
    def get_collection_info(self, collection_name: str) -> Optional[Dict]:
        """Get detailed information about a collection from Qdrant"""
        try:
            collection_info = self.client.get_collection(collection_name)
            metadata = self.get_collection_metadata(collection_name)
            
            return {
                "name": collection_name,
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "metadata": metadata
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return None


# Global instance
_collection_manager = None

def get_collection_manager() -> CollectionManager:
    """Get or create the shared CollectionManager instance"""
    global _collection_manager
    if _collection_manager is None:
        _collection_manager = CollectionManager()
    return _collection_manager
