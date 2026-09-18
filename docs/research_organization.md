# PaperTrail AI Research Organization: Laboratories & Inter-Lab Architecture

This document describes the laboratory architecture of the **PaperTrail AI Research Organization**, its 10 specialized research laboratories, the structural distinction between research domains and cross-cutting methodologies, and the communication protocols that enable cross-lab matrix collaboration.

---

## 1. Executive Organization & Taxonomy

Rather than treating every topic as an isolated vertical category, PaperTrail structures its research organization into three distinct architectural tiers:

```
                    PaperTrail AI Research Organization
                                     │
       ┌─────────────────────────────┼─────────────────────────────┐
       │                             │                             │
  Research Domains        Cross-Cutting Methodologies         Applications &
 (Core Disciplines)          (Horizontal Enablers)              Translation
       │                             │                             │
  ┌────┴─────────────────┐      ┌────┴─────────────────┐      ┌────┴─────────────────┐
  │ LMI  · Language &    │      │ FMPT · Foundation    │      │ AISL · AI for        │
  │        Multimodal    │      │        Model Post-   │      │        Science &     │
  │ AI   · Agentic Intel │      │        Training      │      │        Learning      │
  │ VI   · Visual Intel  │      │ ISAI · Intelligent   │      └──────────────────────┘
  │ RCI  · Reasoning &   │      │        Systems & AI  │
  │        Cognition     │      │        Infrastructure│
  │ SIKG · Semantic & KG │      └──────────────────────┘
  │ GI   · Generative    │
  │ RLDI · RL & Decision │
  └──────────────────────┘
```

### Why Cross-Cutting Methodologies are Distinct
Methodologies like **Post-Training (FMPT)** and **AI Systems & Infrastructure (ISAI)** are not isolated topical silos. Instead, they are **horizontal capabilities** that cut across all core research domains and downstream applications:

- **Agentic Intelligence × Foundation Model Post-Training:** Research on reinforcement learning from environment feedback (GRPO, DPO) to train models for reliable autonomous tool invocation and self-correction.
- **Language & Multimodal Intelligence × Intelligent Systems & AI Infrastructure:** Designing optimized distributed serving kernels (vLLM, FP8 quantization, KV cache compression) for 1M-token multimodal context windows.
- **AI for Science & Learning × Agentic Intelligence × Foundation Model Post-Training:** Developing autonomous tutoring and mathematical visualization agents (e.g. generating Manim animations) post-trained on verified mathematical derivation traces.

---

## 2. Directory of Research Laboratories

### A. Core Research Domains (7 Labs)

| Code | Lab Name | Short Name | Core Scientific Mission | Key Research Topics |
|---|---|---|---|---|
| **`LMI`** | **Language & Multimodal Intelligence Lab** | Language & Multimodal Intelligence (LMI) | Foundation LLMs, vision-language models, multimodal reasoning, and cross-modal alignment. | LLMs, VLMs, cross-modal learning, in-context learning, multimodal RAG |
| **`AI`** | **Agentic Intelligence Lab** | Agentic Intelligence (AI) Lab | Autonomous agents, planning, tool execution, memory architectures, multi-agent systems, workflows. | Autonomous agents, hierarchical planning, agent memory, self-correction, agent evaluation |
| **`VI`** | **Visual Intelligence Lab** | Visual Intelligence (VI) Lab | Image and video understanding, visual perception, open-vocabulary segmentation, spatiotemporal reasoning. | Computer vision, video modeling, object detection, semantic segmentation, visual reasoning |
| **`RCI`** | **Reasoning & Cognitive Intelligence Lab** | Reasoning & Cognitive Intelligence (RCI) Lab | Logical deduction, mathematical problem-solving, neuro-symbolic reasoning, formal verification. | Mathematical reasoning, chain-of-thought verification, neuro-symbolic AI, theorem proving |
| **`SIKG`** | **Semantic Intelligence & Knowledge Graph Lab** | Semantic Intelligence & Knowledge Graphs (SIKG) Lab | Knowledge graphs, formal ontologies, CIDOC-CRM semantics, entity linking, graph neural networks. | Knowledge graphs, ontologies, CIDOC-CRM, entity linking, graph reasoning, linked open data |
| **`GI`** | **Generative Intelligence Lab** | Generative Intelligence (GI) Lab | Generative synthesis across text, image, video, audio, 3D, code, and scientific artifacts. | Diffusion models, flow matching, video generation, 3D assets, neural rendering, code synthesis |
| **`RLDI`** | **Reinforcement Learning & Decision Intelligence Lab** | Reinforcement Learning & Decision Intelligence (RLDI) Lab | Reinforcement learning, reward modeling, preference optimization, sequential decision making under uncertainty. | Policy optimization, offline RL, reward modeling, exploration, decision transformers |

---

### B. Cross-Cutting Methodologies (2 Labs)

| Code | Lab Name | Short Name | Core Scientific Mission | Key Research Topics |
|---|---|---|---|---|
| **`FMPT`** | **Foundation Model Post-Training Lab** | Foundation Model Post-Training (FMPT) Lab | Advanced post-training: SFT, LoRA/QLoRA, DPO, GRPO, RLHF, reward model engineering, safety alignment, reasoning elicitation. | SFT, LoRA/QLoRA, DPO, GRPO, RLHF, reward models, model adaptation, alignment |
| **`ISAI`** | **Intelligent Systems & AI Infrastructure Lab** | Intelligent Systems & AI Infrastructure (ISAI) Lab | Training and inference systems, model serving, distributed AI, FP8/INT4 quantization, KV cache optimization, GPU kernels. | Distributed training, vLLM serving, quantization, KV cache compression, custom GPU kernels |

---

### C. Applications & Translation (1 Lab)

| Code | Lab Name | Short Name | Core Scientific Mission | Key Research Topics |
|---|---|---|---|---|
| **`AISL`** | **AI for Science & Learning Lab** | AI for Science & Learning (AISL) Lab | Real-world scientific discovery and education: mathematical modeling, physics simulation, tutoring, and interactive visualization (Manim). | AI for science, physics simulation, pedagogical AI, intelligent tutoring, scientific visualization |

---

## 3. Communication Protocols: How Labs Communicate

Laboratories in PaperTrail do not operate in isolation. They communicate using **typed Pydantic message contracts** across three communication channels:

```
                            ┌────────────────────────────────────┐
                            │ Senior Research Lead (Orchestrator)│
                            └─────────────────┬──────────────────┘
                                              │ (Vertical Dispatch & Synthesis)
                                              ▼
                    ┌──────────────────────────────────────────────────┐
                    │               Research Lab (e.g. AI)             │
                    └─────────┬──────────────────────────────▲─────────┘
                              │                              │
         InterLabRequest      │                              │ InterLabResponse
                              ▼                              │
                    ┌────────────────────────────────────────┴─────────┐
                    │     Cross-Cutting Lab (e.g. FMPT / ISAI)         │
                    └──────────────────────────────────────────────────┘
```

### 1. Vertical Communication: Senior Lead ↔ Laboratories
The user speaks to the **Senior Research Lead**, who analyzes the research objective and dispatches work:
- The Lead can consult any individual lab directly: `lead.consult_lab(lab_code, task)`.
- The Lead coordinates multi-lab matrix projects: `lead.orchestrate_matrix_collaboration(project)`.

### 2. Horizontal Communication: Lab ↔ Lab
When a research domain lab needs methodology or infrastructure assistance, it issues an `InterLabRequest` to peer labs:

```python
# Agentic Intelligence Lab requests post-training assistance from FMPT:
request = InterLabRequest(
    from_lab="AI",
    to_lab="FMPT",
    task_description="We need an SFT + GRPO recipe to align agents for multi-step tool recovery.",
    context={"base_model": "Llama-3-8B", "environment": "code_interpreter"}
)
```

The responding lab executes its domain reasoning and returns a structured `InterLabResponse`:

```python
# Foundation Model Post-Training Lab returns concrete recipes:
response = InterLabResponse(
    from_lab="FMPT",
    to_lab="AI",
    analysis="Standard SFT leads to compounding errors in multi-step recovery. GRPO with verifier rewards is required.",
    methodology_recommendations=[
        "Curate negative tool traces where recovery succeeds.",
        "Implement Group Relative Policy Optimization (GRPO) with outcome-based unit test rewards.",
        "Apply QLoRA (r=64, alpha=128) targeting attention and MLP projections.",
    ],
    proposed_experiments=[
        "Benchmark tool-use recovery rate on HotpotQA with 3-step simulated errors.",
        "Compare DPO vs GRPO sample efficiency on trajectory correction.",
    ]
)
```

### 3. Multi-Lab Matrix Projects (`MatrixProjectCoordinator`)
Complex scientific initiatives combine a primary domain lab, one or more methodology labs, and an application lab into a `MatrixProject`:

```python
from papertrail.schemas.schema import MatrixProject
from papertrail.labs.collaboration import MatrixProjectCoordinator

project = MatrixProject(
    project_name="Autonomous Educational Physics Animator",
    primary_domain="AI",
    collaborating_methodologies=["FMPT", "ISAI"],
    collaborating_applications=["AISL"],
    mission_statement="Develop and optimize an autonomous agent that generates verified Manim physics animations."
)

coordinator = MatrixProjectCoordinator()
synthesis = coordinator.execute_matrix_project(project)
```

The coordinator:
1. Queries **`AI`** for agent planning, memory, and code self-correction mechanisms.
2. Queries **`FMPT`** for SFT data synthesis and DPO preference alignment on Python/Manim syntax.
3. Queries **`ISAI`** for low-latency speculative decoding and serving containerization.
4. Queries **`AISL`** for physics syllabus standards, pedagogical clarity benchmarks, and visualization guidelines.
5. Emits a strongly typed **`CrossLabSynthesis`** report aggregating all dimensions.

---

## 4. CLI Guide: Working with Research Laboratories

### 1. View the Lab Directory
List all 10 laboratories grouped by Domain, Methodology, and Application:
```bash
papertrail labs
```

### 2. Direct Inquiry to a Specific Lab
Interact directly with any lab using its uppercase code:
```bash
# Language & Multimodal Intelligence
papertrail lab lmi "How do vision-language models handle spatial grounding in high-resolution images?"

# Agentic Intelligence
papertrail lab ai "What memory architecture prevents catastrophic forgetting in multi-day agent runs?"

# Foundation Model Post-Training
papertrail lab fmpt "Compare DPO vs GRPO for mathematical reasoning alignment."

# Intelligent Systems & AI Infrastructure
papertrail lab isai "What are the trade-offs between vLLM PagedAttention and Chunked Prefill?"

# AI for Science & Learning
papertrail lab aisl "How can LLMs be effectively constrained to produce verifiable Manim animations?"
```

### 3. Launch a Multi-Lab Matrix Project
Launch cross-cutting initiatives that bridge domains, methodologies, and applications:
```bash
# Agentic Intelligence × Post-Training
papertrail matrix "Self-Correcting Tool Agents" \
  --domain AI \
  --methodology FMPT \
  --mission "Develop RL-aligned agents with robust error recovery."

# Domain × Multiple Methodologies × Application
papertrail matrix "Automated STEM Physics Tutoring" \
  --domain AI \
  --methodology FMPT,ISAI \
  --application AISL \
  --mission "High-throughput autonomous agent generating interactive Manim physics simulations."

# Output structured synthesis as JSON
papertrail matrix "Neurosymbolic Knowledge Distillation" \
  --domain RCI \
  --methodology FMPT \
  --application AISL \
  --json
```

---

## 5. Summary of Architecture Benefits

1. **No Category Silos:** Prevents the common pitfall of treating "LLMs" and "Generative AI" as disjoint buckets.
2. **First-Class Methodologies:** Treats post-training and infrastructure as cross-cutting horizontal enablers that accelerate every research domain.
3. **Structured Pydantic Schemas:** All inter-lab requests, responses, and matrix syntheses are strongly typed Pydantic models.
4. **Single Conversational Lead:** Users can conduct entire research programs through the Senior Lead (`papertrail lead`), which automatically consults and orchestrates the specialized labs behind the scenes.
