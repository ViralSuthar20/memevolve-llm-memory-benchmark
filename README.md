# MemEvolve: Empirical Benchmarking & Failure Analysis of LLM Agent Memory Architectures

**Author:** [Viral Suthar](https://github.com/ViralSuthar20)  
**Models Evaluated:** `Qwen2.5` & `Llama-3.3`  
**Core Focus:** Agentic Memory Systems, Context Window Token Bloat, Empirical Ablation & Failure-Mode Analysis

---

## 1. Overview & Problem Statement
Self-improving and memory-augmented Large Language Model (LLM) agents rely on external memory architectures to store, retrieve, and synthesize past interaction trajectories across multi-turn tasks. While adding retrieval mechanisms is often assumed to monotonically improve agent performance, uncurated memory injection frequently introduces severe **context window token bloat** and distractor noise.

This repository contains an empirical benchmarking and failure-analysis study built around the `evolvelab` pipeline, evaluating how baseline vs. evolved memory retrieval providers affect inner-loop task execution, token efficiency, and meta-evolutionary fitness across open-weight instruction-tuned models (`Qwen2.5` and `Llama-3.3`).

---

## 2. System Architecture & Module Breakdown

```text
memevolve-llm-memory-benchmark/
├── evolvelab/
│   ├── __init__.py
│   ├── base.py                     # Core abstract interfaces and data structures
│   ├── dataset_loader.py           # Benchmark dataset ingestion and preprocessing
│   ├── memevolve_provider.py       # Baseline MemEvolve memory provider implementation
│   ├── evolved_memory.py           # Evolved memory state representation and storage
│   ├── evolved_provider_v2.py      # Modified retrieval-augmented memory provider (v2)
│   ├── inner_loop.py               # Unified inner-loop agent task execution harness
│   ├── inner_loop_baseline.py      # Baseline inner-loop evaluation runner
│   ├── inner_loop_evolved.py       # Evolved memory inner-loop evaluation runner
│   ├── smol_inner_loop.py          # Lightweight agent inner-loop execution wrapper
│   ├── meta_evolver.py             # Outer-loop meta-evolution and fitness scoring engine
│   └── meta_evolver_v1_archive.py  # Archived v1 meta-evolver implementation for comparison
├── main.py                         # Primary entry point for running benchmark pipelines
├── req.txt                         # Python package dependencies
└── README.md                       # Project documentation and ablation findings
