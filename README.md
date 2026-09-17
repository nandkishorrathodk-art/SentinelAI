# 🛡️ SentinelAI — Living Neural ASI Framework

**SentinelAI** is an autonomous, self-evolving Artificial Superintelligence (ASI) foundation built from scratch for cybersecurity, offensive/defensive hacking, code synthesis, multimodal visual perception, and recursive self-improvement.

---

## 🌟 Core Architecture & Innovations

- **Sentinel Adaptive MoE (SA-MoE)**: Mixture-of-Experts with an always-on Shared Expert, Learned Neural Router, Cross-Expert Residuals, and dynamic expert birth/death lifecycle.
- **Synaptic Plasticity (Living Weights)**: Micro-Hebbian updates during inference, allowing the model to adapt and learn from every interaction without full retraining.
- **3-Tier Hierarchical Memory**:
  - `L1`: Working Context Ring-Buffer
  - `L2`: Episodic Memory with vector indexing
  - `L3`: Permanent Semantic Crystals & Knowledge Graphs
  - `Dream Consolidator`: Offline sleep mode that crystallizes lessons and prunes noise.
- **Cognitive Swarm (10 Sub-Agents)**:
  - 5 Knowledge Harvesters (Reasoning, Code/AST, Vision, Cyber/Exploit, Multimodal Synthesizer)
  - 5 Meta-Cognitive Agents (Red Team Self-Critic, Curriculum Designer, Memory Architect, Adversarial Quality Gate, MoE Topology Evolver)
- **ASI Substrate**:
  - `RecursiveCodeEvolver`: Seed AI engine that inspects, benchmarks, and mutates its own source code and architecture.
  - `AdversarialSelfPlayArena`: Dual-mind competitive arena (Red Mind vs Blue Mind) for superhuman knowledge synthesis.
  - `NeuralWorldModel`: Latent dynamic simulator with Monte Carlo Tree Search (MCTS) for forward mental planning.
- **Hardware Agnostic & NPU Ready**:
  - Trained on Kaggle Dual NVIDIA T4 GPUs via `torch.nn.DataParallel` and mixed-precision FP16.
  - Compiles to OpenVINO INT8 Intermediate Representation for sub-10ms inference on Intel AI Boost NPUs (11 TOPS).

---

## 📁 Repository Structure

```
sentinel-ai/
├── pyproject.toml                         # Project manifest & dependencies
├── build_kaggle_notebook.py               # Builder for self-contained Kaggle notebook
├── notebooks/
│   ├── train_on_kaggle.ipynb              # Dual-T4 Kaggle GPU training notebook
│   └── kernel-metadata.json               # Kaggle kernel push configuration
└── sentinel/
    ├── tokenizer/                         # Pure Python Byte-Pair Encoding (from scratch)
    ├── model/                             # SA-MoE Transformer, GQA, RoPE, Synaptic Plasticity
    ├── vision/                            # ViT PatchEmbedding & Multimodal Projector
    ├── memory/                            # 3-Tier Memory (L1, L2, L3) & Dream Consolidator
    ├── sentinel_loop/                     # Always-on loop (Awake, Patrol, Dream)
    ├── agents/                            # 10 Cognitive Sub-Agents & Central Orchestrator
    ├── asi/                               # Seed AI Self-Evolution, Self-Play Arena & World Model
    ├── training/                          # Dataset loaders & Multi-GPU SentinelTrainer
    ├── inference/                         # Streaming Generator & Intel NPU OpenVINO Exporter
    └── tests/                             # Core unit & integration smoke test suite
```

---

## 🚀 Quickstart

### 1. Training on Kaggle (Free Dual-T4 GPU)
```bash
# Push directly to Kaggle using the Kaggle API
export KAGGLE_API_TOKEN="YOUR_KAGGLE_API_TOKEN"
python -m kaggle kernels push -p notebooks/
```

### 2. Exporting to Intel NPU
```bash
python -m sentinel.inference.export_openvino --checkpoint checkpoints/sentinel_final.pt --device NPU
```

---

## 📜 License
MIT License. Authored by Nandkishor Rathod.
