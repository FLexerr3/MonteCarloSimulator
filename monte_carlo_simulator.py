"""Regime-switching Monte Carlo market simulator.

This module simulates market paths where returns and volatility depend on
latent market regimes that evolve according to a Markov transition matrix.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import random
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class Regime:
    """Parameters for one market regime."""

    name: str
    drift: float
    vol: float


@dataclass
class SimulationResult:
    """Holds Monte Carlo output."""

    prices: List[List[float]]
    log_returns: List[List[float]]
    regimes: List[List[int]]

    def summary(self, _regimes: Sequence[Regime]) -> Dict[str, float]:
        terminal = sorted(path[-1] for path in self.prices)
        n = len(terminal)

        def percentile(p: float) -> float:
            if n == 1:
                return terminal[0]
            idx = p * (n - 1)
            lo = int(math.floor(idx))
            hi = int(math.ceil(idx))
            if lo == hi:
                return terminal[lo]
            w = idx - lo
            return terminal[lo] * (1 - w) + terminal[hi] * w

        return {
            "expected_terminal": sum(terminal) / n,
            "median_terminal": percentile(0.5),
            "p05_terminal": percentile(0.05),
            "p95_terminal": percentile(0.95),
            "worst_terminal": terminal[0],
            "best_terminal": terminal[-1],
        }


def _validate_transition_matrix(matrix: Sequence[Sequence[float]]) -> None:
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("Transition matrix must be square and non-empty.")

    for row in matrix:
        if any(v < 0 for v in row):
            raise ValueError("Transition matrix cannot contain negative probabilities.")
        if not math.isclose(sum(row), 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"Each transition row must sum to 1. Got {sum(row)}")


def _sample_categorical(weights: Sequence[float], rng: random.Random) -> int:
    x = rng.random()
    running = 0.0
    for idx, w in enumerate(weights):
        running += w
        if x <= running:
            return idx
    return len(weights) - 1


def simulate_regime_switching_market(
    n_paths: int,
    n_steps: int,
    dt: float,
    s0: float,
    regimes: Sequence[Regime],
    transition_matrix: Sequence[Sequence[float]],
    initial_regime_probs: Sequence[float],
    seed: int | None = None,
) -> SimulationResult:
    """Run a regime-switching Monte Carlo simulation."""

    if n_paths <= 0 or n_steps <= 0:
        raise ValueError("n_paths and n_steps must be positive integers.")
    if s0 <= 0:
        raise ValueError("Initial price s0 must be positive.")

    _validate_transition_matrix(transition_matrix)
    n_regimes = len(regimes)
    if len(transition_matrix) != n_regimes:
        raise ValueError("Transition matrix size must match number of regimes.")
    if len(initial_regime_probs) != n_regimes:
        raise ValueError("initial_regime_probs length must match number of regimes.")
    if any(p < 0 for p in initial_regime_probs):
        raise ValueError("initial_regime_probs cannot contain negative values.")
    if not math.isclose(sum(initial_regime_probs), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("initial_regime_probs must sum to 1.")

    rng = random.Random(seed)
    sqrt_dt = math.sqrt(dt)

    prices: List[List[float]] = [[s0] + [0.0] * n_steps for _ in range(n_paths)]
    log_returns: List[List[float]] = [[0.0] * n_steps for _ in range(n_paths)]
    regime_states: List[List[int]] = [[0] * n_steps for _ in range(n_paths)]

    current_regimes = [_sample_categorical(initial_regime_probs, rng) for _ in range(n_paths)]

    for t in range(n_steps):
        for p in range(n_paths):
            r_idx = current_regimes[p]
            regime_states[p][t] = r_idx
            regime = regimes[r_idx]

            z = rng.gauss(0.0, 1.0)
            step_log_return = (
                (regime.drift - 0.5 * regime.vol * regime.vol) * dt
                + regime.vol * sqrt_dt * z
            )

            log_returns[p][t] = step_log_return
            prices[p][t + 1] = prices[p][t] * math.exp(step_log_return)
            current_regimes[p] = _sample_categorical(transition_matrix[r_idx], rng)

    return SimulationResult(prices=prices, log_returns=log_returns, regimes=regime_states)


DEFAULT_REGIMES: List[Regime] = [
    Regime(name="bull", drift=0.12, vol=0.15),
    Regime(name="sideways", drift=0.04, vol=0.20),
    Regime(name="bear", drift=-0.10, vol=0.30),
]

DEFAULT_TRANSITION: List[List[float]] = [
    [0.90, 0.08, 0.02],
    [0.10, 0.80, 0.10],
    [0.06, 0.18, 0.76],
]

DEFAULT_INITIAL_PROBS: List[float] = [0.60, 0.30, 0.10]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regime-switching Monte Carlo market simulator")
    parser.add_argument("--paths", type=int, default=10000, help="Number of simulation paths")
    parser.add_argument("--steps", type=int, default=252, help="Number of time steps")
    parser.add_argument("--dt", type=float, default=1 / 252, help="Step size in years")
    parser.add_argument("--s0", type=float, default=100.0, help="Initial asset price")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    result = simulate_regime_switching_market(
        n_paths=args.paths,
        n_steps=args.steps,
        dt=args.dt,
        s0=args.s0,
        regimes=DEFAULT_REGIMES,
        transition_matrix=DEFAULT_TRANSITION,
        initial_regime_probs=DEFAULT_INITIAL_PROBS,
        seed=args.seed,
    )

    output = result.summary(DEFAULT_REGIMES)
    output["regimes"] = [r.name for r in DEFAULT_REGIMES]
    output["paths"] = args.paths
    output["steps"] = args.steps

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
