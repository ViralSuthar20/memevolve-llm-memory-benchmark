from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

class MemoryRequest(BaseModel):
    query: str
    context: str
    status: dict

class MemoryItem(BaseModel):
    content: Any
    metadata: dict

class MemoryResponse(BaseModel):
    items: List[MemoryItem]

class TrajectoryData(BaseModel):
    query: str
    trajectory: List[dict]
    metadata: dict

class BaseMemoryProvider(ABC):
    """Abstract base class for memory providers."""
    
    def __init__(self, memory_type: str, config: Optional[dict] = None):
        self.memory_type = memory_type
        self.config = config or {}

    @abstractmethod
    def provide_memory(self, request: MemoryRequest) -> MemoryResponse:
        """
        Retrieve relevant memories based on query, context, and status.
        Maps to the Retrieve (R) module.
        """
        pass

    @abstractmethod
    def take_in_memory(self, trajectory_data: TrajectoryData) -> tuple[bool, str]:
        """
        Store/ingest new memory from trajectory data.
        Maps to the Encode (E) and Store (U) modules.
        """
        pass

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the memory provider (load existing data, setup indices, etc.).
        """
        pass
        
    def get_memory_type(self) -> str:
        """Get the type of this memory provider."""
        return self.memory_type

    def get_config(self) -> dict:
        """Get the configuration of this memory provider."""
        return self.config.copy()