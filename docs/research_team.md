# PaperTrail AI Research Team: 4-Tier Multi-Agent Hierarchy

PaperTrail organizes autonomous scientific agents into a **4-tier academic research hierarchy** powered by **PydanticAI**.

The system is designed so that **you primarily interact with the Senior Researcher / Lead**, who strategically coordinates, plans, and delegates tasks down the research chain. You can also occasionally communicate directly with any of the specialized agents whenever targeted assistance is required.

---

## 1. Research Team Hierarchy & Division of Labor

```
                                  ┌───────────────────┐
                                  │       USER        │
                                  └─────────┬─────────┘
                                            │
               ┌────────────────────────────┼────────────────────────────┐
               │ (Primary Interface)        │ (Occasional Direct)        │
               ▼                            ▼                            ▼
┌──────────────────────────────┐ ┌────────────────────┐ ┌────────────────────┐
│ Tier 4: Senior Researcher    │ │ Tier 3: Researcher │ │ Tier 2: Assistant  │
│         / Lead (PI)          │ │                    │ │                    │
└──────────────┬───────────────┘ └────────────────────┘ └────────────────────┘
               │
               ├─► delegates hypotheses & experiments ──► Tier 3: Researcher
               │
               ├─► delegates benchmarks & synthesis   ──► Tier 2: Research Assistant
               │
               └─► delegates search & data collection ──► Tier 1: Research Intern
```

| Tier | Role | Primary Focus | CLI Command |
|---|---|---|---|
| **Tier 4** | **Senior Researcher / Lead** | Strategic direction, methodology soundness, peer-review quality control, cross-domain coordination, team orchestration | `papertrail lead` |
| **Tier 3** | **Researcher** | Formulate testable scientific hypotheses, design controlled empirical experiments, interpret results against theory | `papertrail researcher` / `plan` / `hypothesis` / `experiment` |
| **Tier 2** | **Research Assistant** | Comparative literature matrices, empirical benchmarks, engineering implementation blueprints, gap analyses | `papertrail assistant` |
| **Tier 1** | **Research Intern** | Broad literature search, arXiv harvesting, PDF download & FAISS indexing, reproduction checks, simple baseline experiments | `papertrail intern` |

---

## 2. Interaction Model: How to Work with the Team

### 🌟 Primary Workflow: Talking to the Senior Researcher / Lead

In typical research, you only need to speak to the **Senior Researcher / Lead**. The Lead acts as your Principal Investigator (PI) and chief orchestrator. It uses autonomous delegation tools to command the Researcher, Assistant, and Intern behind the scenes, returning cohesive, rigorous, executive-level scientific briefs.

```bash
# General inquiry — the Lead synthesizes theory and delegates tasks as needed:
papertrail lead "Should we pivot our architecture from pure Transformers to hybrid Mamba-Transformer models for 1M context?"

# End-to-end multi-tier research campaign:
papertrail lead "State space models vs attention for genomic sequence modeling" --orchestrate

# Peer-review quality assessment on a paper or proposal:
papertrail lead "Attention Is All You Need" --review-quality
papertrail lead "Attention Is All You Need" --review-quality --json

# Multi-phase strategic research roadmap:
papertrail lead "Sub-quadratic foundation models for long-horizon planning" --roadmap
```

---

### 🔬 Occasional Direct Interactions with Individual Tiers

When you need focused work without full orchestration, you can interact directly with each tier:

#### Direct Interaction: Tier 3 — Researcher
Use when you already have literature and want to brainstorm hypotheses, design controlled experiments, or analyze empirical anomalies:
```bash
# Formulate a testable, falsifiable hypothesis:
papertrail researcher "associative recall in recurrent linear models" --hypothesis
papertrail researcher "associative recall in recurrent linear models" --hypothesis --json

# Design a controlled experiment protocol with baselines and metrics:
papertrail researcher "Mamba outperforms Transformers on synthetic induction heads at 128k tokens" --experiment

# Interpret empirical results against theory:
papertrail researcher "Model achieved 94.2% on Needle-in-a-Haystack at 32k, but dropped to 61.5% at 64k" --interpret
```

#### Direct Interaction: Tier 2 — Research Assistant
Use when you need implementation blueprints, benchmark evaluations, or side-by-side paper comparisons:
```bash
# Benchmark evaluation across models:
papertrail assistant "Transformer vs Mamba vs RWKV" --benchmark
papertrail assistant "Transformer vs Mamba vs RWKV" --benchmark --json

# Detailed implementation blueprint:
papertrail assistant "Bidirectional selective state space block in PyTorch" --implement

# Side-by-side comparative literature matrix:
papertrail assistant "Sparse Attention vs Linear Attention" --compare

# Detailed gap analysis & open questions:
papertrail assistant "Sub-quadratic attention mechanisms" --gaps

# Structured Research Dossier:
papertrail assistant "Long-context language modeling" --dossier
```

#### Direct Interaction: Tier 1 — Research Intern
Use for ground-level literature retrieval, harvesting arXiv papers, or reproduction checks:
```bash
# Quick literature inquiry:
papertrail intern "What papers in our repository discuss flash attention?"

# Autonomous multi-step literature review:
papertrail intern "Diffusion models for text generation" --investigate

# Bulk download & index papers from arXiv:
papertrail intern "multi-modal state space models" --collect

# Reproduction feasibility assessment:
papertrail intern "2312.00752" --reproduce
```

---

## 3. Strongly Typed Pydantic Schemas

All structured communications across tiers are validated by Pydantic models in `src/papertrail/schemas/schema.py`:

- **`HypothesisSpec`**: Clear testable claim, theoretical rationale, independent & dependent variables, falsification criteria, confidence level.
- **`ExperimentSpec`**: Title, hypothesis spec, datasets, baseline models, ablation conditions, quantitative metrics, expected outcomes, implementation notes.
- **`BenchmarkResult`**: Benchmark name, task type, primary metric, model score dictionary, baseline gain comparison, compute notes.
- **`PaperReviewScorecard`**: Soundness (1-5), novelty (1-5), empirical rigor (1-5), clarity (1-5), overall verdict, strengths, weaknesses, recommendations.
- **`ResearchRoadmap`**: Initiative name, strategic objective, phased execution milestones, cross-domain synergies, technical risks, team delegation plan.
- **`ResearchDossier`**: Executive summary, taxonomy dictionary, list of `PaperSummary`, comparative dimensions, and gap analysis.
- **`GapAnalysis`**: Deficiencies, untested assumptions, open scientific questions, and promising directions.

---

## 4. Python API Usage

You can also orchestrate or query the research team directly in Python:

```python
from papertrail.agents import (
    LeadResearcher,
    ResearcherAgent,
    ResearchAssistant,
    ResearchIntern,
)

# 1. Primary interface: Talk to the Lead
lead = LeadResearcher()
report = lead.orchestrate_campaign("State space models for long-horizon planning")
scorecard = lead.review_paper_quality("2312.00752")
print(f"Paper Verdict: {scorecard.overall_verdict} (Soundness: {scorecard.soundness_score}/5)")

# 2. Direct interaction with Researcher
researcher = ResearcherAgent()
hypothesis = researcher.formulate_hypothesis_spec("linear attention scalability")
print(f"Hypothesis: {hypothesis.hypothesis}")
print(f"Falsification: {hypothesis.falsification_criteria}")

# 3. Direct interaction with Assistant
assistant = ResearchAssistant()
benchmarks = assistant.run_benchmarks("Mamba vs Transformer")
print(f"Benchmark: {benchmarks.benchmark_name} -> {benchmarks.model_scores}")

# 4. Direct interaction with Intern
intern = ResearchIntern()
papers = intern.collect_data("reinforcement learning from AI feedback", limit=3)
```
