from typing import Optional
from evolvelab.base import BaseMemoryProvider, MemoryRequest, MemoryResponse, TrajectoryData, MemoryItem

class MemEvolveProvider(BaseMemoryProvider):
    """
    Concrete implementation of a self-evolving memory system.
    Designed for ablation studies on Encode, Store, Retrieve, and Manage.
    """
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__(memory_type="MemEvolve_Base", config=config)
        self.memory_state = []  # Placeholder for vector DB or graph [cite: 242]
        
    def initialize(self) -> bool:
        # Load any existing FAISS indices or JSON storage
        print("Initialized MemEvolve Provider Database.")
        return True

    # ---------------------------------------------------------
    # MODULE 1: ENCODE (E) & MODULE 2: STORE (U)
    # ---------------------------------------------------------
    def take_in_memory(self, trajectory_data: TrajectoryData) -> tuple[bool, str]:
        """Orchestrates ingestion: Extract -> Encode -> Store[cite: 816, 817]."""
        try:
            # Encode Module (E)
            encoded_experiences = self._encode(trajectory_data)
            
            # Store Module (U)
            self._store(encoded_experiences)
            
            return True, f"Successfully absorbed {len(encoded_experiences)} units."
        except Exception as e:
            return False, str(e)

    def _encode(self, data: TrajectoryData) -> list:
        """Transforms raw experiences into structured representations[cite: 239]."""
        # Ablation Target: Change this from raw text to LLM-generated insights.
        extracted_units = [{"content": step, "type": "raw_step"} for step in data.trajectory]
        return extracted_units

    def _store(self, encoded_units: list) -> None:
        """Integrates encoded experiences into persistent memory[cite: 241]."""
        # Ablation Target: Change from flat list to Vector DB.
        self.memory_state.extend(encoded_units)

    # ---------------------------------------------------------
    # MODULE 3: RETRIEVE (R)
    # ---------------------------------------------------------
    def provide_memory(self, request: MemoryRequest) -> MemoryResponse:
        """Provides task-relevant memory content[cite: 243]."""
        retrieved_content = self._retrieve(request)
        
        items = [MemoryItem(content=c["content"], metadata={"source": c["type"]}) for c in retrieved_content]
        return MemoryResponse(items=items)

    def _retrieve(self, request: MemoryRequest) -> list:
        """Context-aware recall logic."""
        # Ablation Target: Change from generic string match to semantic similarity.
        relevant_memories = [m for m in self.memory_state if request.query in str(m.get("content", ""))]
        return relevant_memories[:3]  # Return top 3

    # ---------------------------------------------------------
    # MODULE 4: MANAGE (G)
    # ---------------------------------------------------------
    def manage_memory(self) -> None:
        """Performs offline consolidation and selective forgetting[cite: 245]."""
        # Ablation Target: Implement Dynamic Cheatsheet style pruning here.
        initial_size = len(self.memory_state)
        # Placeholder: Remove duplicates
        self.memory_state = [dict(t) for t in {tuple(d.items()) for d in self.memory_state}]
        print(f"Consolidated memory from {initial_size} to {len(self.memory_state)} items.")

    def test_complex_tasks(self) -> None:
        # Testing with more complex tasks or edge cases
        complex_request = MemoryRequest(query="complex_task")
        retrieved_content = self._retrieve(complex_request)
        print(f"Retrieved {len(retrieved_content)} memories for complex task")

    def test_edge_cases(self) -> None:
        # Testing with edge cases
        edge_request = MemoryRequest(query="")
        retrieved_content = self._retrieve(edge_request)
        print(f"Retrieved {len(retrieved_content)} memories for edge case")

# Example usage
if __name__ == "__main__":
    provider = MemEvolveProvider()
    provider.initialize()
    trajectory_data = TrajectoryData(trajectory=["step1", "step2", "step3"])
    provider.take_in_memory(trajectory_data)
    request = MemoryRequest(query="step1")
    response = provider.provide_memory(request)
    print(f"Retrieved {len(response.items)} memories")
    provider.manage_memory()
    provider.test_complex_tasks()
    provider.test_edge_cases()