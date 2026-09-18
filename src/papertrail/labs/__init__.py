"""
PaperTrail Research Laboratories Package.

Contains the 10 specialized research laboratories, inter-lab communication protocols,
and cross-lab matrix project collaboration coordination.
"""
from papertrail.labs.registry import (
    LAB_CATALOG,
    get_all_labs,
    get_lab,
    get_labs_by_category,
    list_lab_codes,
)
from papertrail.labs.lab_agent import (
    ResearchLabAgent,
    create_lab_pydantic_agent,
    create_interlab_response_agent,
    build_lab_system_prompt,
)
from papertrail.labs.collaboration import (
    MatrixProjectCoordinator,
    create_matrix_synthesis_agent,
)

__all__ = [
    "LAB_CATALOG",
    "get_all_labs",
    "get_lab",
    "get_labs_by_category",
    "list_lab_codes",
    "ResearchLabAgent",
    "create_lab_pydantic_agent",
    "create_interlab_response_agent",
    "build_lab_system_prompt",
    "MatrixProjectCoordinator",
    "create_matrix_synthesis_agent",
]
