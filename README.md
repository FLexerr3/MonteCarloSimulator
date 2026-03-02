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

## Matplotlib regime/volatility/price table

You can generate a matplotlib image containing a table with **step**, **regime**,
**volatility**, and **price** for a selected simulation path:

```bash
python3 monte_carlo_simulator.py \
  --paths 2000 \
  --steps 126 \
  --seed 7 \
  --table-path-index 0 \
  --plot-table-max-rows 40 \
  --plot-table-output artifacts/path_table.png
```

The JSON output includes `plot_table_output` when the image is created.

> Note: this feature requires `matplotlib` to be installed in your Python environment.

## Customization

Edit the defaults in `monte_carlo_simulator.py`:

- `DEFAULT_REGIMES`: per-regime drift/volatility
- `DEFAULT_TRANSITION`: transition probabilities between regimes
- `DEFAULT_INITIAL_PROBS`: starting regime distribution

You can also import `simulate_regime_switching_market(...)` directly from Python.
