# Lagged Drivers Model Analysis

This repository contains a simple analysis pipeline for the `data/lags.csv` results.  The `analysis/analyze_lags.py` script

* ranks model specifications using AICc, computes ΔAICc, and generates Akaike weights;
* parses the `FixedEffects` column to recover per-model coefficient estimates;
* builds model-averaged (weight-adjusted) coefficients together with uncertainty summaries;
* creates light-weight SVG visualisations for the top model weights, model-averaged coefficients, and summed predictor importance; and
* exports a few compact CSV summaries that make it easier to explore the results set by scope or predictor.

## Usage

```bash
python analysis/analyze_lags.py
```

Running the script writes several artifacts to the `outputs/` directory:

* `model_weights.csv` – model metadata sorted by AICc with ΔAICc and Akaike weights.
* `weighted_coefficients.csv` – model-averaged coefficient estimates and variability.
* `predictor_importance.csv` – summed model weights for each predictor.
* `scope_dv_summary.csv` – quick comparison of the best scores by scope and dependent variable.
* `top_model_weights.svg`, `weighted_coefficients.svg`, `predictor_importance.svg` – visual summaries of the AICc weights, the averaged coefficient magnitudes, and variable importance, respectively.

The SVG visualisations are generated without external plotting libraries, so the script can run in minimal Python environments.
