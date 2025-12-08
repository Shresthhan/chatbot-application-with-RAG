# database.py - Database setup and operations

from sqlalchemy import create_engine, Column, String, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import enum

# ========== DATABASE CONNECTION ==========

# This creates a file called "ingestions.db" in your project folder
DATABASE_URL = "sqlite:///./ingestions.db"

# Create the database engine (like opening the notebook)
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}  # SQLite setting
)

# SessionLocal is how we'll read/write to the database
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all our database tables
Base = declarative_base()

# ========== STATUS OPTIONS ==========

class IngestionStatus(str, enum.Enum):
    """
    These are the only 4 possible statuses an ingestion can have.
    Like a traffic light: only red, yellow, or green - nothing else.
    """
    PENDING = "pending"        # Waiting to start
    PROCESSING = "processing"  # Currently working
    COMPLETED = "completed"    # Successfully finished
    FAILED = "failed"         # Something went wrong

# ========== DATABASE TABLE STRUCTURE ==========

class IngestionJob(Base):
    """
    This defines the structure of our 'ingestion_jobs' table.
    Think of it like a spreadsheet with columns for each piece of info.
    
    Each row = one ingestion task
    """
    __tablename__ = "ingestion_jobs"
    
    # Primary key (unique ID for each job)
    ingestion_id = Column(String, primary_key=True, index=True)
    
    # Status tracking
    status = Column(String, nullable=False, default=IngestionStatus.PENDING)
    message = Column(String, nullable=False, default="Ingestion queued")
    progress = Column(Integer, default=0)  # 0 to 100
    
    # Document details
    collection_name = Column(String, nullable=False)
    chunking_strategy = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    
    # Results (filled in when completed)
    num_chunks = Column(Integer, nullable=True)
    error = Column(String, nullable=True)
    
    # Timestamps
    started_at = Column(DateTime, default=datetime.now)
    completed_at = Column(DateTime, nullable=True)
    
    def to_dict(self):
        """
        Converts database row to Python dictionary.
        Needed because FastAPI can't directly send database objects as JSON.
        """
        return {
            "ingestion_id": self.ingestion_id,
            "status": self.status,
            "message": self.message,
            "progress": self.progress,
            "collection_name": self.collection_name,
            "chunking_strategy": self.chunking_strategy,
            "original_filename": self.original_filename,
            "num_chunks": self.num_chunks,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

# ========== HELPER FUNCTIONS ==========

def get_db():
    """
    Opens a database session (connection).
    Use this in FastAPI endpoints with 'Depends(get_db)'.
    Automatically closes the connection when done.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """
    Creates the database tables if they don't exist yet.
    Call this once when your app starts.
    """
    Base.metadata.create_all(bind=engine)
    print("✓ Database initialized (ingestions.db created)")

# ========== CRUD OPERATIONS ==========
# CRUD = Create, Read, Update, Delete

def create_ingestion_job(db, **kwargs):
    """
    Create a new ingestion job in the database.
    
    Example usage:
    create_ingestion_job(
        db,
        ingestion_id="abc-123",
        collection_name="my_docs",
        original_filename="paper.pdf"
    )
    """
    job = IngestionJob(**kwargs)
    db.add(job)
    db.commit()
    db.refresh(job)  # Get the saved version with auto-filled fields
    return job

def get_ingestion_job(db, ingestion_id: str):
    """
    Retrieve one ingestion job by its ID.
    Returns None if not found.
    
    Example:
    job = get_ingestion_job(db, "abc-123")
    """
    return db.query(IngestionJob).filter(
        IngestionJob.ingestion_id == ingestion_id
    ).first()

def update_ingestion_job(db, ingestion_id: str, **updates):
    """
    Update specific fields of an ingestion job.
    
    Example:
    update_ingestion_job(
        db,
        ingestion_id="abc-123",
        status="completed",
        progress=100
    )
    """
    job = get_ingestion_job(db, ingestion_id)
    if job:
        for key, value in updates.items():
            setattr(job, key, value)  # Updates the field
        db.commit()
        db.refresh(job)
    return job

def list_ingestion_jobs(db, limit: int = 100):
    """
    Get a list of all ingestion jobs, newest first.
    Limit controls how many to return.
    """
    return db.query(IngestionJob).order_by(
        IngestionJob.started_at.desc()
    ).limit(limit).all()
