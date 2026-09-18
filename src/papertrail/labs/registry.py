"""
PaperTrail Research Laboratories Registry.

Catalog of the 10 specialized research laboratories organized into:
- Core Research Domains (LMI, AI, VI, RCI, SIKG, GI, RLDI)
- Cross-Cutting Methodologies (FMPT, ISAI)
- Applications & Translation (AISL)
"""
from typing import Dict, List, Optional
from papertrail.schemas.schema import LabInfo, LabCategory


LAB_CATALOG: Dict[str, LabInfo] = {
    # ── Core Research Domains ────────────────────────────────────────────────
    "LMI": LabInfo(
        code="LMI",
        full_name="Language & Multimodal Intelligence Lab",
        short_name="Language & Multimodal Intelligence (LMI)",
        category="domain",
        focus=(
            "Large language models, vision-language models, multimodal reasoning, "
            "language understanding, foundation models, and cross-modal representation learning."
        ),
        key_topics=[
            "large language models",
            "vision-language models",
            "multimodal reasoning",
            "cross-modal alignment",
            "in-context learning",
            "retrieval-augmented generation",
        ],
    ),
    "AI": LabInfo(
        code="AI",
        full_name="Agentic Intelligence Lab",
        short_name="Agentic Intelligence (AI) Lab",
        category="domain",
        focus=(
            "Autonomous agents, tool use, hierarchical planning, episodic and semantic memory, "
            "multi-agent orchestration, human-agent workflows, and agent evaluation."
        ),
        key_topics=[
            "autonomous agents",
            "tool use",
            "multi-agent systems",
            "hierarchical planning",
            "agent memory architectures",
            "self-correction",
            "agent evaluation",
        ],
    ),
    "VI": LabInfo(
        code="VI",
        full_name="Visual Intelligence Lab",
        short_name="Visual Intelligence (VI) Lab",
        category="domain",
        focus=(
            "Image and video understanding, visual perception, open-vocabulary segmentation, "
            "object detection, spatiotemporal visual reasoning, and dense vision representations."
        ),
        key_topics=[
            "computer vision",
            "video understanding",
            "visual perception",
            "object detection",
            "semantic segmentation",
            "spatiotemporal modeling",
        ],
    ),
    "RCI": LabInfo(
        code="RCI",
        full_name="Reasoning & Cognitive Intelligence Lab",
        short_name="Reasoning & Cognitive Intelligence (RCI) Lab",
        category="domain",
        focus=(
            "Logical reasoning, mathematical problem-solving, structured multi-step inference, "
            "neuro-symbolic synthesis, formal verification, and cognitive architectures."
        ),
        key_topics=[
            "mathematical reasoning",
            "logical deduction",
            "neuro-symbolic AI",
            "chain-of-thought verification",
            "formal theorem proving",
            "cognitive architectures",
        ],
    ),
    "SIKG": LabInfo(
        code="SIKG",
        full_name="Semantic Intelligence & Knowledge Graph Lab",
        short_name="Semantic Intelligence & Knowledge Graphs (SIKG) Lab",
        category="domain",
        focus=(
            "Knowledge graphs, formal ontologies, CIDOC-CRM cultural semantics, entity linking, "
            "graph neural networks, graph reasoning, and linked open data interoperability."
        ),
        key_topics=[
            "knowledge graphs",
            "ontologies",
            "CIDOC-CRM",
            "entity linking",
            "graph neural networks",
            "semantic web",
            "linked data",
        ],
    ),
    "GI": LabInfo(
        code="GI",
        full_name="Generative Intelligence Lab",
        short_name="Generative Intelligence (GI) Lab",
        category="domain",
        focus=(
            "Generative modeling across modalities: diffusion models, autoregressive synthesis, "
            "video generation, audio & music synthesis, 3D assets, neural rendering, and code generation."
        ),
        key_topics=[
            "diffusion models",
            "flow matching",
            "video generation",
            "3D generation",
            "audio synthesis",
            "neural rendering",
            "code synthesis",
        ],
    ),
    "RLDI": LabInfo(
        code="RLDI",
        full_name="Reinforcement Learning & Decision Intelligence Lab",
        short_name="Reinforcement Learning & Decision Intelligence (RLDI) Lab",
        category="domain",
        focus=(
            "Reinforcement learning, reward modeling, direct preference optimization, sequential "
            "decision-making under uncertainty, model-based RL, exploration, and game-theoretic agents."
        ),
        key_topics=[
            "reinforcement learning",
            "reward modeling",
            "policy optimization",
            "offline RL",
            "exploration strategies",
            "sequential decision making",
        ],
    ),

    # ── Cross-Cutting Methodologies ──────────────────────────────────────────
    "FMPT": LabInfo(
        code="FMPT",
        full_name="Foundation Model Post-Training Lab",
        short_name="Foundation Model Post-Training (FMPT) Lab",
        category="methodology",
        focus=(
            "Cross-cutting post-training methodologies: Supervised Fine-Tuning (SFT), LoRA/QLoRA, "
            "Direct Preference Optimization (DPO), Group Relative Policy Optimization (GRPO), "
            "RLHF, reward model training, safety alignment, reasoning elicitation, and efficient adaptation."
        ),
        key_topics=[
            "supervised fine-tuning (SFT)",
            "LoRA / QLoRA",
            "DPO",
            "GRPO",
            "RLHF",
            "reward models",
            "model alignment",
            "reasoning post-training",
        ],
    ),
    "ISAI": LabInfo(
        code="ISAI",
        full_name="Intelligent Systems & AI Infrastructure Lab",
        short_name="Intelligent Systems & AI Infrastructure (ISAI) Lab",
        category="methodology",
        focus=(
            "High-performance AI systems: distributed training, low-latency inference serving, "
            "KV cache optimization, quantization (FP8, INT4, AWQ), custom GPU kernels, "
            "evaluation harnesses, and scalable AI infrastructure."
        ),
        key_topics=[
            "distributed training",
            "inference serving (vLLM, TensorRT-LLM)",
            "quantization (FP8, INT4)",
            "KV cache compression",
            "GPU kernel optimization",
            "evaluation infrastructure",
        ],
    ),

    # ── Applications & Translation ───────────────────────────────────────────
    "AISL": LabInfo(
        code="AISL",
        full_name="AI for Science & Learning Lab",
        short_name="AI for Science & Learning (AISL) Lab",
        category="application",
        focus=(
            "Real-world scientific and educational translation: AI for mathematics and physics, "
            "molecular simulation, pedagogical AI, automated tutoring, educational assessment, "
            "personalized learning, and interactive visual explanation (e.g. Manim animations)."
        ),
        key_topics=[
            "AI for scientific discovery",
            "physics simulation",
            "pedagogical AI",
            "intelligent tutoring",
            "educational assessment",
            "scientific visualization (Manim)",
        ],
    ),
}


def get_all_labs() -> List[LabInfo]:
    """Return all registered research laboratories."""
    return list(LAB_CATALOG.values())


def get_lab(code: str) -> Optional[LabInfo]:
    """Retrieve lab metadata by its code (case-insensitive)."""
    return LAB_CATALOG.get(code.upper())


def get_labs_by_category(category: LabCategory) -> List[LabInfo]:
    """Retrieve all laboratories belonging to a specific category."""
    return [lab for lab in LAB_CATALOG.values() if lab.category == category]


def list_lab_codes() -> List[str]:
    """Return all valid lab codes."""
    return list(LAB_CATALOG.keys())
