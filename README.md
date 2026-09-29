# 🧠 SYNAPSE-AI STUDIO: Autonomous Agents, LLM Fine-Tuning & Data Science Workbench

[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](LICENSE)
[![Agents: ReAct DAG](https://img.shields.io/badge/Agents-ReAct%20Cognitive%20Swarm-indigo.svg)]()
[![LLM: LoRA & DPO](https://img.shields.io/badge/LLM-LoRA%20PEFT%20%2B%20DPO-purple.svg)]()
[![Attention: Multi-Head QKV](https://img.shields.io/badge/Attention-Multi--Head%20QKV-cyan.svg)]()
[![AutoML: XGBoost & SHAP](https://img.shields.io/badge/AutoML-XGBoost%20%2B%20SHAP-amber.svg)]()

> **Synapse-AI Studio** is an all-in-one flagship platform bridging **Autonomous Agent Architecture**, **Large Language Model (LLM) Fine-Tuning**, and **Automated Data Science**. Built to provide an interactive, visual, and executable workbench for cutting-edge Artificial Intelligence engineering.

---

## 🏛️ System Architecture

```
                                  +-----------------------------------------------+
                                  |         Synapse-AI Interactive Studio         |
                                  +-----------------------+-----------------------+
                                                          |
             +--------------------------------------------+--------------------------------------------+
             |                                            |                                            |
             v                                            v                                            v
+--------------------------+                 +--------------------------+                 +--------------------------+
|  Autonomous Agent Swarm  |                 |   LLM Training & LoRA    |                 |   Data Science & AutoML  |
|  [ReAct DAG Orchestrator]|                 | [QKV Attention & SFT/DPO]|                 | [EDA, ROC-AUC & SHAP XAI]|
+------------+-------------+                 +------------+-------------+                 +------------+-------------+
             |                                            |                                            |
     +-------+-------+                            +-------+-------+                            +-------+-------+
     |               |                            |               |                            |               |
     v               v                            v               v                            v               v
[Architect]     [Researcher]                [QKV Heatmaps]  [LoRA Decomp]                [Correlation]   [Model Arena]
[Coder]         [Critic]                    [SFT Simulator] [DPO Align]                  [Confusion Mtx] [SHAP Plots]
```

---

## ⚡ Core Technical Pillars

### 1. 🤖 Autonomous Agent Design & Multi-Agent Swarms (`agent-swarm.js`)
- **4 Specialized Cognitive Agents:**
  - **Lead Architect:** Decomposes complex user goals into formal Directed Acyclic Graph (DAG) execution steps.
  - **RAG Researcher:** Queries vector stores, extracts documentation, and retrieves academic literature.
  - **Systems Coder:** Writes clean, modular, type-safe Python and TypeScript implementations with sandboxed unit testing.
  - **Critic & Alignment Evaluator:** Constitutional AI safety auditor checking code soundness and hallucination scores.
- **ReAct Cognitive Architecture:**
  $$\text{Thought} \longrightarrow \text{Action (Tool Call)} \longrightarrow \text{Observation} \longrightarrow \text{Reflection}$$
- **Dual-Tier Memory Hierarchy:**
  - Short-Term sliding conversation token buffer.
  - Long-Term episodic vector memory with cosine similarity ranking.

### 2. 🔍 Transformer Multi-Head Self-Attention (`llm-trainer.js`)
- **Scaled Dot-Product Attention:**
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
- **Multi-Head Specialization Matrix:**
  - Head 0: Syntactic Dependencies & Verb-Object Binding
  - Head 1: Long-Range Coreference Resolution
  - Head 2: Semantic Similarity & Lexical Associations
  - Head 3: Relative Positional & Sequence Ordering
- Interactive token hover probe detailing query-key vector interactions and linguistic interpretations.

### 3. ⚡ LoRA (Low-Rank Adaptation) PEFT Studio (`llm-trainer.js`)
- **Weight Decomposition:**
  $$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$
- **Parameter Efficiency:** Freezes $99.2\%$ of base model parameters while adapting full conversational capacity using low-rank matrices ($r=16, \alpha=32$).
- **Live Training Simulators:**
  - **SFT (Supervised Fine-Tuning):** Task instruction tuning.
  - **DPO (Direct Preference Optimization):** Closed-form implicit reward alignment without reinforcement learning stability issues.
  - **Causal Pre-Training:** Next-token loss reduction curves.

### 4. 📊 Data Science, AutoML & SHAP Explainability (`data-science.js`)
- **Exploratory Data Analysis (EDA):** $6 \times 6$ Pearson Correlation Matrix with color-coded intensity heatmap.
- **AutoML Model Arena:** Automated benchmarking of **XGBoost** (Champion: $0.984\text{ AUC}$), LightGBM, Random Forest, and Deep MLP.
- **Evaluation & Explainability (XAI):**
  - Interactive Canvas ROC-AUC curve.
  - $2 \times 2$ Confusion Matrix ($TP, FP, TN, FN$).
  - **SHAP Waterfall Attribution:** Visualizing how each individual feature pushes an inference prediction above or below the baseline expectation $E[f(x)]$.

---

## 🌐 Live Deployment & Interactive Studio

- **Live URL:** [https://tareq0001.github.io/Synapse-Agentic-Studio/](https://tareq0001.github.io/Synapse-Agentic-Studio/)
- **Repository:** [https://github.com/Tareq0001/Synapse-Agentic-Studio](https://github.com/Tareq0001/Synapse-Agentic-Studio)

---

## 🛠️ Local Development & Quick Start

```bash
# Clone the repository
git clone https://github.com/Tareq0001/Synapse-Agentic-Studio.git
cd Synapse-Agentic-Studio

# Serve locally
npx serve .
# Or via Python
python -m http.server 8080
```

Open `http://localhost:8080` in any modern web browser.

---

## 📜 License
Distributed under the **MIT License**. Created by [Tareq Abuashi](https://github.com/Tareq0001).
