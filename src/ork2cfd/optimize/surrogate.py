"""Constrained Gaussian Process Bayesian Optimization (Matérn-5/2 ARD + EI) & dry-run proxy."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats.qmc import LatinHypercube

from ..exceptions import Ork2CfdError
from ..models.rocket import FinCrossSection, FinSet, RocketModel
from ..physics.boundary_layer import compute_boundary_layer
from .models import (
    FinCandidate,
    FinDesignBounds,
    FinOptimizationConfig,
    StabilityReport,
)

_CROSS_SECTION_FORM_FACTOR: dict[FinCrossSection, float] = {
    FinCrossSection.SQUARE: 1.00,
    FinCrossSection.ROUNDED: 0.56,
    FinCrossSection.DOUBLE_WEDGE: 0.44,
    FinCrossSection.AIRFOIL: 0.36,
}


def _bounds_arrays(
    bounds: FinDesignBounds,
    active_params: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    lows = np.array([bounds.get_range(p)[0] for p in active_params], dtype=float)
    highs = np.array([bounds.get_range(p)[1] for p in active_params], dtype=float)
    span = np.where(highs - lows > 1e-12, highs - lows, 1.0)
    return lows, lows + span


def normalize_vectors(
    vectors: np.ndarray | list[list[float]],
    bounds: FinDesignBounds,
    active_params: tuple[str, ...],
) -> np.ndarray:
    """Scale continuous vectors into the unit hypercube [0, 1]^d."""
    lows, highs = _bounds_arrays(bounds, active_params)
    arr = np.asarray(vectors, dtype=float)
    return np.clip((arr - lows) / (highs - lows), 0.0, 1.0)


def denormalize_vector(
    unit_vec: np.ndarray,
    bounds: FinDesignBounds,
    active_params: tuple[str, ...],
) -> list[float]:
    """Map a normalized point in [0, 1]^d back into physical parameter units."""
    lows, highs = _bounds_arrays(bounds, active_params)
    phys = lows + np.clip(unit_vec, 0.0, 1.0) * (highs - lows)
    return [float(v) for v in phys]


class GaussianProcessSurrogate:
    """
    Pure NumPy/SciPy Gaussian Process regressor with ARD Matérn-5/2 kernel.

    Designed specifically for tiny simulation budgets (5-30 CFD runs) on continuous
    parameter spaces up to ~6-8 dimensions without requiring external ML packages.
    """

    def __init__(self, bounds: FinDesignBounds, active_params: tuple[str, ...]) -> None:
        self.bounds = bounds
        self.active_params = active_params
        self.dim = len(active_params)
        self._x_norm: np.ndarray = np.empty((0, self.dim), dtype=float)
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._y_scaled: np.ndarray = np.empty(0, dtype=float)
        self._sigma_f2: float = 1.0
        self._lengthscales: np.ndarray = np.full(self.dim, 0.35, dtype=float)
        self._noise: float = 1e-4
        self._chol: tuple[np.ndarray, bool] | None = None
        self._alpha: np.ndarray = np.empty(0, dtype=float)

    def _matern52_matrix(
        self,
        xa: np.ndarray,
        xb: np.ndarray,
        sigma_f2: float,
        lengthscales: np.ndarray,
    ) -> np.ndarray:
        ls = np.maximum(lengthscales, 1e-3)
        diff = (xa[:, None, :] - xb[None, :, :]) / ls[None, None, :]
        r2 = np.sum(diff * diff, axis=2)
        r = np.sqrt(np.maximum(r2, 1e-18))
        sqrt5_r = math.sqrt(5.0) * r
        return sigma_f2 * (1.0 + sqrt5_r + (5.0 / 3.0) * r2) * np.exp(-sqrt5_r)

    def _cholesky_with_jitter(
        self,
        k_mat: np.ndarray,
        base_noise: float,
    ) -> tuple[np.ndarray, bool]:
        n = k_mat.shape[0]
        eye = np.eye(n, dtype=float)
        for jitter in (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
            try:
                return cho_factor(k_mat + (base_noise + jitter) * eye, lower=True)
            except LinAlgError:
                continue
        return cho_factor(k_mat + (base_noise + 0.05) * eye, lower=True)

    def fit(self, vectors: list[list[float]], targets: list[float]) -> None:
        """Fit ARD Matérn-5/2 hyperparameters via multi-start marginal likelihood."""
        x_raw = np.asarray(vectors, dtype=float)
        y_raw = np.asarray(targets, dtype=float)
        self._x_norm = normalize_vectors(x_raw, self.bounds, self.active_params)
        self._y_mean = float(np.mean(y_raw))
        y_std_raw = float(np.std(y_raw))
        self._y_std = y_std_raw if y_std_raw > 1e-6 else 1e-3
        self._y_scaled = (y_raw - self._y_mean) / self._y_std

        n = len(y_raw)
        if n <= 2:
            self._sigma_f2 = 1.0
            self._lengthscales = np.full(self.dim, 0.35, dtype=float)
            self._noise = 1e-4
            k_mat = self._matern52_matrix(
                self._x_norm, self._x_norm, self._sigma_f2, self._lengthscales
            )
            self._chol = self._cholesky_with_jitter(k_mat, self._noise)
            self._alpha = cho_solve(self._chol, self._y_scaled)
            return

        def _neg_log_marginal_likelihood(theta: np.ndarray) -> float:
            sigma_f2 = math.exp(2.0 * float(theta[0]))
            ell = np.exp(theta[1 : 1 + self.dim])
            noise = math.exp(2.0 * float(theta[-1]))
            k_mat = self._matern52_matrix(self._x_norm, self._x_norm, sigma_f2, ell)
            try:
                chol = self._cholesky_with_jitter(k_mat, noise)
                alpha = cho_solve(chol, self._y_scaled)
            except LinAlgError:
                return 1e12
            log_det = 2.0 * float(np.sum(np.log(np.maximum(np.diag(chol[0]), 1e-18))))
            nll = 0.5 * float(np.dot(self._y_scaled, alpha)) + 0.5 * log_det + 0.5 * n * math.log(
                2.0 * math.pi
            )
            return nll if math.isfinite(nll) else 1e12

        param_bounds = [(-2.0, 2.0)] + [(-2.6, 1.1)] * self.dim + [(-6.0, -0.7)]
        starts = [
            np.array([0.0] + [-1.0] * self.dim + [-4.0]),
            np.array([0.2] + [-0.4] * self.dim + [-3.0]),
            np.array([-0.3] + [-1.6] * self.dim + [-4.6]),
        ]

        best_theta = starts[0]
        best_val = _neg_log_marginal_likelihood(best_theta)
        for init in starts:
            res = minimize(
                _neg_log_marginal_likelihood,
                init,
                method="L-BFGS-B",
                bounds=param_bounds,
                options={"maxiter": 60},
            )
            if float(res.fun) < best_val:
                best_val = float(res.fun)
                best_theta = np.asarray(res.x, dtype=float)

        self._sigma_f2 = math.exp(2.0 * float(best_theta[0]))
        self._lengthscales = np.exp(best_theta[1 : 1 + self.dim])
        self._noise = math.exp(2.0 * float(best_theta[-1]))
        k_mat = self._matern52_matrix(
            self._x_norm, self._x_norm, self._sigma_f2, self._lengthscales
        )
        self._chol = self._cholesky_with_jitter(k_mat, self._noise)
        self._alpha = cho_solve(self._chol, self._y_scaled)

    def predict(
        self,
        vectors: np.ndarray | list[list[float]],
        *,
        already_normalized: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict surrogate posterior (mean, std) in original Cd units."""
        if self._chol is None:
            raise Ork2CfdError("GaussianProcessSurrogate must be fit before calling predict().")
        x_q = (
            np.asarray(vectors, dtype=float)
            if already_normalized
            else normalize_vectors(vectors, self.bounds, self.active_params)
        )
        if x_q.ndim == 1:
            x_q = x_q[None, :]
        k_star = self._matern52_matrix(x_q, self._x_norm, self._sigma_f2, self._lengthscales)
        mean_scaled = k_star @ self._alpha
        v_mat = cho_solve(self._chol, k_star.T)
        var_scaled = np.maximum(self._sigma_f2 - np.sum(k_star.T * v_mat, axis=0), 1e-12)
        mean = self._y_mean + self._y_std * mean_scaled
        std = self._y_std * np.sqrt(var_scaled)
        return mean, std

    def expected_improvement(
        self,
        vectors: np.ndarray | list[list[float]],
        y_best: float,
        xi: float = 0.002,
        *,
        already_normalized: bool = False,
    ) -> np.ndarray:
        """Compute expected improvement (minimization) across query points."""
        mu, sigma = self.predict(vectors, already_normalized=already_normalized)
        improvement = y_best - mu - xi
        ei = np.zeros_like(mu)
        mask = sigma > 1e-12
        if np.any(mask):
            z = improvement[mask] / sigma[mask]
            ei[mask] = improvement[mask] * norm.cdf(z) + sigma[mask] * norm.pdf(z)
        return np.maximum(ei, 0.0)


def sample_feasible_lhs(
    bounds: FinDesignBounds,
    active_params: tuple[str, ...],
    template_candidate: FinCandidate,
    feasibility_fn: Callable[[FinCandidate], StabilityReport],
    n_samples: int,
    existing_vectors: list[list[float]] | None = None,
    seed: int = 42,
) -> list[FinCandidate]:
    """Sample maximin space-filling Latin Hypercube points inside the feasible constraint set."""
    if n_samples <= 0:
        return []
    dim = len(active_params)
    sampler = LatinHypercube(d=dim, seed=seed)
    pool_unit = sampler.random(n=3500)

    feasible_pairs: list[tuple[np.ndarray, FinCandidate]] = []
    for u in pool_unit:
        vals = denormalize_vector(u, bounds, active_params)
        cand = template_candidate.with_vector(active_params, vals)
        if feasibility_fn(cand).feasible:
            feasible_pairs.append((u, cand))

    if not feasible_pairs:
        raise Ork2CfdError(
            "No feasible fin geometries found within design bounds matching stability/geometry "
            "constraints. Relax `--min-sm`, adjust `--cg-m`, or widen fin bounds."
        )

    existing_unit = (
        list(normalize_vectors(existing_vectors, bounds, active_params))
        if existing_vectors
        else []
    )
    chosen: list[FinCandidate] = []
    chosen_units = list(existing_unit)

    remaining = list(feasible_pairs)
    for _ in range(min(n_samples, len(remaining))):
        if not chosen_units:
            pick_idx = 0
        else:
            existing_arr = np.vstack(chosen_units)
            rem_arr = np.vstack([p[0] for p in remaining])
            dists = np.linalg.norm(rem_arr[:, None, :] - existing_arr[None, :, :], axis=2)
            min_dists = np.min(dists, axis=1)
            pick_idx = int(np.argmax(min_dists))
        picked_u, picked_cand = remaining.pop(pick_idx)
        chosen.append(picked_cand)
        chosen_units.append(picked_u)

    return chosen


def propose_next_bo_candidate(
    gp: GaussianProcessSurrogate,
    bounds: FinDesignBounds,
    active_params: tuple[str, ...],
    template_candidate: FinCandidate,
    feasibility_fn: Callable[[FinCandidate], StabilityReport],
    evaluated_vectors: list[list[float]],
    y_best: float,
    seed: int = 42,
    exploit_only: bool = False,
) -> FinCandidate:
    """Find the constrained global maximum of Expected Improvement (or minimum GP mean)."""
    dim = len(active_params)
    rng = np.random.default_rng(seed)
    lhs = LatinHypercube(d=dim, seed=seed).random(n=3000)

    eval_norm = normalize_vectors(evaluated_vectors, bounds, active_params)
    local_perturbs: list[np.ndarray] = []
    for base_u in eval_norm:
        local_perturbs.append(np.clip(base_u + rng.normal(0.0, 0.12, size=(120, dim)), 0.0, 1.0))
    pool_unit = np.vstack([lhs, *local_perturbs])

    feasible_units: list[np.ndarray] = []
    feasible_cands: list[FinCandidate] = []
    min_dup_dist = 0.025
    for u in pool_unit:
        if np.min(np.linalg.norm(eval_norm - u[None, :], axis=1)) < min_dup_dist:
            continue
        vals = denormalize_vector(u, bounds, active_params)
        cand = template_candidate.with_vector(active_params, vals)
        if feasibility_fn(cand).feasible:
            feasible_units.append(u)
            feasible_cands.append(cand)

    if not feasible_cands:
        # Relax duplicate distance if feasible region is very tight
        for u in pool_unit:
            vals = denormalize_vector(u, bounds, active_params)
            cand = template_candidate.with_vector(active_params, vals)
            if feasibility_fn(cand).feasible:
                feasible_units.append(u)
                feasible_cands.append(cand)

    if not feasible_cands:
        raise Ork2CfdError("No feasible infill candidate could be generated under constraints.")

    feas_arr = np.vstack(feasible_units)
    if exploit_only:
        mu, _ = gp.predict(feas_arr, already_normalized=True)
        scores = -mu
    else:
        scores = gp.expected_improvement(feas_arr, y_best=y_best, xi=0.002, already_normalized=True)

    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    best_cand = feasible_cands[best_idx]

    def _make_neg_acq():
        def _neg_acq(u_raw: np.ndarray) -> float:
            u_clamped = np.clip(u_raw, 0.0, 1.0)
            if np.any(np.abs(u_raw - u_clamped) > 1e-6):
                return 1e6 + float(np.sum((u_raw - u_clamped) ** 2))
            if np.min(np.linalg.norm(eval_norm - u_clamped[None, :], axis=1)) < min_dup_dist:
                return 1e5
            vals = denormalize_vector(u_clamped, bounds, active_params)
            cand = template_candidate.with_vector(active_params, vals)
            stab = feasibility_fn(cand)
            if not stab.feasible:
                return 1e5 + max(0.0, 2.0 - stab.static_margin_cal) * 100.0
            if exploit_only:
                mu_loc, _ = gp.predict(u_clamped[None, :], already_normalized=True)
                return float(mu_loc[0])
            ei_loc = gp.expected_improvement(
                u_clamped[None, :], y_best=y_best, xi=0.002, already_normalized=True
            )
            return -float(ei_loc[0])

        return _neg_acq

    neg_acq_fn = _make_neg_acq()
    top_indices = np.argsort(scores)[::-1][:6]
    for idx in top_indices:
        u0 = feasible_units[int(idx)]
        res = minimize(neg_acq_fn, u0, method="Nelder-Mead", options={"maxiter": 80})
        u_opt = np.clip(res.x, 0.0, 1.0)
        if np.min(np.linalg.norm(eval_norm - u_opt[None, :], axis=1)) >= min_dup_dist:
            cand_opt = template_candidate.with_vector(
                active_params, denormalize_vector(u_opt, bounds, active_params)
            )
            if feasibility_fn(cand_opt).feasible:
                val_opt = -float(res.fun)
                if val_opt > best_score:
                    best_score = val_opt
                    best_cand = cand_opt

    return best_cand


def estimate_analytic_drag(
    rocket: RocketModel,
    target_fin: FinSet,
    config: FinOptimizationConfig,
) -> tuple[float, float]:
    """
    Fast analytical Cd proxy for `--dry-run` testing without launching ANSYS Fluent.

    Captures Reynolds skin friction, cross-section pressure/form drag, thickness ratio (t/c),
    leading-edge sweep wave/compressibility relief, and fin-body interference.
    """
    from ..models.cfd import DomainMode, FlightCondition

    bl = compute_boundary_layer(
        rocket,
        FlightCondition(mach=config.mach, altitude_m=config.altitude_m, aoa_deg=config.aoa_deg),
        DomainMode.FLUID_DOMAIN,
    )
    s_ref = max(rocket.reference_area, 1e-9)
    mach = max(0.05, config.mach)

    # Smooth + roughness body & base drag baseline around Cd_body ~ 0.30 at transonic Mach
    wetted_body = sum(
        math.pi * (comp.fore_radius + comp.aft_radius) * comp.length
        for comp in rocket.all_components
    )
    cf_body = bl.skin_friction_cf
    cd_body_friction = 1.12 * cf_body * (wetted_body / s_ref)
    cd_base_and_wave = 0.145 + 0.095 * max(0.0, (mach - 0.65) / 0.35) ** 2
    cd_body = cd_body_friction + cd_base_and_wave

    cd_fins_total = 0.0
    for fin in rocket.all_fins:
        c_avg = max(0.5 * (fin.root_chord + fin.tip_chord), 1e-4)
        s_wet_fin = 2.0 * 0.5 * (fin.root_chord + fin.tip_chord) * fin.span * fin.fin_count
        t_over_c = fin.thickness / c_avg
        sweep_rad = math.atan2(fin.sweep_length, max(fin.span, 1e-4))
        mach_n = min(0.95, mach * math.cos(sweep_rad))
        comp_amplify = 1.0 / math.sqrt(max(0.25, 1.0 - mach_n * mach_n))
        form_factor = _CROSS_SECTION_FORM_FACTOR[fin.cross_section]

        # Friction + profile pressure drag + transonic thickness/sweep wave drag + tip vortex
        re_fin = max(1e4, bl.reynolds_length * (c_avg / max(rocket.total_length, 0.1)))
        cf_fin = 0.074 / (re_fin**0.2)
        cd_profile = cf_fin * (
            1.0 + 2.0 * form_factor * t_over_c + 60.0 * form_factor * (t_over_c**4)
        )
        cd_wave = (
            2.2
            * form_factor
            * (t_over_c**2)
            * (max(0.0, mach_n - 0.35) ** 2)
            * comp_amplify
        )
        cd_interference = 0.12 * t_over_c * form_factor
        cd_f = (cd_profile + cd_wave + cd_interference) * (s_wet_fin / s_ref)
        if fin is target_fin:
            pass
        cd_fins_total += cd_f

    return cd_body + cd_fins_total, cd_fins_total
