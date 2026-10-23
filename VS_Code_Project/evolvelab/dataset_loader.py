import os

from datasets import load_dataset
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

# The exact schema defined in Section 4.3 of our Progress Report
class TrajectoryData(BaseModel):
    task_id: str
    trajectory: List[Dict[str, Any]]
    timestamp: datetime
    metadata: Optional[Dict[str, Any]]

class BenchmarkLoader:
    def __init__(self):
        print("Initializing Benchmark Loader...")

    def load_gaia_validation(self, max_samples: int = 5) -> List[TrajectoryData]:
        """
        Loads the 2023 GAIA validation dataset from HuggingFace.
        We strictly limit samples during dev to prevent massive API costs.
        """
        print(f"Pulling {max_samples} samples from GAIA benchmark...")
        
        # Load GAIA validation set
        dataset = load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation", token=os.environ.get("HF_TOKEN"))
        
        formatted_tasks = []
        for i, row in enumerate(dataset):
            if i >= max_samples:
                break
                
            # Map GAIA schema to our MemEvolve schema
            task_data = TrajectoryData(
                task_id=f"GAIA_val_{row['task_id']}",
                trajectory=[{
                    "role": "user",
                    "content": row["Question"],
                    "ground_truth": row["Final answer"]
                }],
                timestamp=datetime.now(),
                metadata={
                    "level": row["Level"],
                    "file_name": row["file_name"],
                    "annotator_metadata": row["Annotator Metadata"]
                }
            )
            formatted_tasks.append(task_data)
            
        print(f"Successfully loaded {len(formatted_tasks)} GAIA tasks.")
        return formatted_tasks

# --- Execution Test ---
if __name__ == "__main__":
    loader = BenchmarkLoader()
    gaia_tasks = loader.load_gaia_validation(max_samples=2)
    
    for task in gaia_tasks:
        print(f"\nTask ID: {task.task_id}")
        print(f"Level: {task.metadata['level']}")
        print(f"Question: {task.trajectory[0]['content']}")
        print(f"Expected Answer: {task.trajectory[0]['ground_truth']}")