import os
from evolvelab.memevolve_provider import MemEvolveProvider
from evolvelab.smol_inner_loop import SmolAgentEvaluator
from evolvelab.meta_evolver_v1_archive import MetaEvolver

def run_memevolve_pipeline():
    print("=== STAGE 1: INNER LOOP (Evaluation) ===")
    
    # 1. Setup the initial memory architecture
    provider = MemEvolveProvider()
    evaluator = SmolAgentEvaluator(provider)
    
    # 2. Define the benchmark dataset
    test_dataset = [
        {"query": "What is the capital of France? Multiply the number of letters in the capital's name by 5."},
        {"query": "If I have 3 apples and eat 1, how many are left?"}
    ]
    
    # 3. Execute the agent and capture the metrics profile
    candidate_profile = evaluator.run_benchmark(test_dataset)
    
    print("\n=== STAGE 2: OUTER LOOP (Meta-Evolution) ===")
    
    # 4. Initialize the Meta-Evolver
    evolver = MetaEvolver()
    
    # 5. Define file paths for the rewrite process
    # Using relative paths from the root directory to locate the target files
    input_file = os.path.join("evolvelab", "memevolve_provider.py")
    output_file = os.path.join("evolvelab", "evolved_provider_v2.py")
    
    # 6. Execute the evolution cycle (Diagnose & Design)
    # The evolver expects a list of candidates for Pareto-ranking, so we wrap our single profile in a list
    evolver.run_evolution_cycle(
        candidate_profiles=[candidate_profile],
        provider_filepath=input_file,
        output_filepath=output_file
    )
    
    print(f"\n=== PIPELINE COMPLETE ===")
    print(f"Check the evolvelab directory. The new architecture should be saved as: {output_file}")

if __name__ == "__main__":
    # Ensure the Hugging Face token is present before executing the pipeline
    if not os.environ.get("HF_TOKEN"):
        print("CRITICAL WARNING: HF_TOKEN environment variable not set. The evaluation will fail.")
    else:
        run_memevolve_pipeline()