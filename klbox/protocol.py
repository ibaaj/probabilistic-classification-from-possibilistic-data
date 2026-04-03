from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from klbox.dykstra_cpp import dykstra_kl_project_cpp
from klbox.gaps import choose_gap_parameters
from klbox.kl_types import FloatArray
from klbox.np_utils import clip_strictly_positive, normalize_to_simplex
from klbox.possibility import antipignistic_reverse_mapping, compute_possibility_order


def sample_pi(n: int, rng: np.random.Generator, eps: float) -> FloatArray:
    # Strictly positive normalized possibility distribution π on X={1,...,n}.
    # Simple synthetic choice: π_k = eps + (1-eps)U_k, then normalize by max(π)=1.
    u = rng.random(n, dtype=np.float64)
    pi = eps + (1.0 - eps) * u
    pi = pi / float(np.max(pi))
    return pi


def sample_q(n: int, rng: np.random.Generator, eps: float) -> FloatArray:
    # Strictly positive q ∈ Δ_n, with small clipping.
    v = rng.random(n, dtype=np.float64)
    v = clip_strictly_positive(v, eps)
    return normalize_to_simplex(v)


@dataclass(frozen=True)
class RunStats:
    convergence_rate: float
    mean_cycles: float
    p90_cycles: float
    mean_final_V: float
    min_final_V: float
    max_final_V: float
    mean_time_s: float
    p90_time_s: float


def aggregate_stats(
    cycles: np.ndarray,
    final_V: np.ndarray,
    times_s: np.ndarray,
    tau: float,
) -> RunStats:
    return RunStats(
        convergence_rate=float(np.mean(final_V <= tau)),
        mean_cycles=float(np.mean(cycles)),
        p90_cycles=float(np.quantile(cycles, 0.9)),
        mean_final_V=float(np.mean(final_V)),
        min_final_V=float(np.min(final_V)),
        max_final_V=float(np.max(final_V)),
        mean_time_s=float(np.mean(times_s)),
        p90_time_s=float(np.quantile(times_s, 0.9)),
    )


def latex_table(
    rows: List[Tuple[float, RunStats]],
    *,
    n: int,
    runs: int,
    K_max: int,
    label: str,
) -> str:
    out: List[str] = []
    out.append(r"\begin{table}[H]")
    out.append(r"\centering")
    out.append(r"\begin{tabular}{lrrrrr}")
    out.append(r"\toprule")
    out.append(
        r"Tolerance $\tau$ & Convergence rate & Mean cycles & 90th perc. cycles & Mean final violation & Mean time [s]\\"
    )
    out.append(r"\midrule")
    for tau, st in rows:
        out.append(
            rf"${tau:.0e}$ & {st.convergence_rate:.3f} & {st.mean_cycles:.1f} & {st.p90_cycles:.1f} & "
            rf"{st.mean_final_V:.2e} & {st.mean_time_s:.3f}\\"
        )
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(
        rf"\caption{{KL projection on $\mathcal{{F}}^{{\mathrm{{box}}}}$ for $n={int(n)}$, "
        rf"with a maximum of $K_{{\max}}={int(K_max)}$ Dykstra cycles and {int(runs)} random runs per tolerance.}}"
    )
    out.append(rf"\label{{{label}}}")
    out.append(r"\end{table}")
    return "\n".join(out)


def run_sweep(
    n: int,
    runs: int,
    taus: Sequence[float],
    K_max: int,
    seed: int,
    tie_tol: float,
    eps_cap: float,
    pi_eps: float,
    log_clip_eps: float,
    *,
    include_prefix_constraints: bool = True,
    no_progress: bool = False,
) -> List[Tuple[float, RunStats]]:
    rng = np.random.default_rng(seed)
    rows: List[Tuple[float, RunStats]] = []

    for tau in tqdm(list(taus), desc=f"dykstra-sweep (Kmax={K_max})", disable=bool(no_progress)):
        cycles = np.zeros(runs, dtype=np.int64)
        final_V = np.zeros(runs, dtype=np.float64)
        times_s = np.zeros(runs, dtype=np.float64)

        for t in tqdm(range(runs), desc=f"tau={tau:.0e}", leave=False, disable=bool(no_progress)):
            pi = sample_pi(n, rng, eps=pi_eps)
            order = compute_possibility_order(pi)
            anti = antipignistic_reverse_mapping(order)

            gaps = choose_gap_parameters(
                tilde_pi=order.tilde_pi,
                dot_g=anti.dot_g,
                tie_tol=float(tie_tol),
                eps_cap=float(eps_cap),
            )

            q = sample_q(n, rng, eps=log_clip_eps)
            res = dykstra_kl_project_cpp(
                q=q,
                order=order,
                gaps=gaps,
                tau=float(tau),
                K_max=int(K_max),
                log_clip_eps=float(log_clip_eps),
                include_prefix_constraints=bool(include_prefix_constraints),
            )

            if (not np.isfinite(res.final_V)) or (not np.all(np.isfinite(res.p_star))):
                np.savez(
                    "nan_case.npz",
                    pi=pi,
                    q=q,
                    sigma=order.sigma,
                    tilde_pi=order.tilde_pi,
                    underline=gaps.underline,
                    overline=gaps.overline,
                    tau=tau,
                    K_max=K_max,
                    run_index=t,
                )
                raise FloatingPointError(
                    f"Non-finite output saved to nan_case.npz (run={t}, tau={tau}, K_max={K_max})"
                )

            cycles[t] = int(res.cycles)
            final_V[t] = float(res.final_V)
            times_s[t] = float(res.elapsed_s)

        st = aggregate_stats(cycles, final_V, times_s, tau=float(tau))
        rows.append((float(tau), st))

    return rows