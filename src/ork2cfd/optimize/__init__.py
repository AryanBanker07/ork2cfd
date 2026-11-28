"""Reusable constrained Bayesian fin shape optimization workflow for ork2cfd."""

from .constraints import compute_barrowman_cp, compute_fin_flutter_speed, estimate_cg_from_ork
from .models import (
    ACTIVE_CONTINUOUS_PARAMS,
    FinCandidate,
    FinDesignBounds,
    FinOptimizationConfig,
    FinOptimizationResult,
    LockedMeshCharacteristics,
    SimEvaluation,
    StabilityReport,
)
from .parameterizer import (
    apply_candidate_to_rocket,
    candidate_from_fin,
    default_bounds_for_fin,
    resolve_target_fin,
    write_modified_ork,
)
from .plots import render_fluent_hdf5_contours, render_optimization_cd_plot
from .runner import load_locked_mesh_from_convergence, resolve_domain_mode, run_fin_optimization
from .surrogate import (
    GaussianProcessSurrogate,
    estimate_analytic_drag,
    propose_next_bo_candidate,
    sample_feasible_lhs,
)

__all__ = [
    "ACTIVE_CONTINUOUS_PARAMS",
    "FinCandidate",
    "FinDesignBounds",
    "FinOptimizationConfig",
    "FinOptimizationResult",
    "GaussianProcessSurrogate",
    "LockedMeshCharacteristics",
    "SimEvaluation",
    "StabilityReport",
    "apply_candidate_to_rocket",
    "candidate_from_fin",
    "compute_barrowman_cp",
    "compute_fin_flutter_speed",
    "default_bounds_for_fin",
    "estimate_analytic_drag",
    "estimate_cg_from_ork",
    "load_locked_mesh_from_convergence",
    "propose_next_bo_candidate",
    "render_fluent_hdf5_contours",
    "render_optimization_cd_plot",
    "resolve_domain_mode",
    "resolve_target_fin",
    "run_fin_optimization",
    "sample_feasible_lhs",
    "write_modified_ork",
]
