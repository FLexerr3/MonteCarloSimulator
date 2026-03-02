# Monte Carlo Simulator with Regime Switching

This project simulates market prices using a **regime-switching Monte Carlo model**.
Instead of assuming one fixed drift and volatility, the process switches between
multiple market states (for example: bull, sideways, bear), each with its own return
and volatility profile.

## Model intuition

- Regime transitions follow a discrete-time Markov chain.
- Within each time step, returns are sampled from a geometric Brownian motion update:
  - `log_return_t = (mu_r - 0.5*sigma_r^2) * dt + sigma_r * sqrt(dt) * Z`
  - where `r` is the active regime and `Z ~ N(0,1)`.

This makes paths more realistic than single-regime GBM because periods of high/low
volatility and positive/negative drift can cluster.

## Quick start

```bash
python3 monte_carlo_simulator.py --paths 10000 --steps 252 --s0 100 --seed 42
```

Example output:

```json
{
  "expected_terminal": 108.74,
  "median_terminal": 105.89,
  "p05_terminal": 63.28,
  "p95_terminal": 166.14,
  "worst_terminal": 32.91,
  "best_terminal": 289.47,
  "regimes": ["bull", "sideways", "bear"],
  "paths": 10000,
  "steps": 252
}
```

## Customization

Edit the defaults in `monte_carlo_simulator.py`:

- `DEFAULT_REGIMES`: per-regime drift/volatility
- `DEFAULT_TRANSITION`: transition probabilities between regimes
- `DEFAULT_INITIAL_PROBS`: starting regime distribution

You can also import `simulate_regime_switching_market(...)` directly from Python.
