# AGENTS.md

## Purpose
This agent automates the workflow for analyzing outbreak–shock relationships using pre-saved model results (`data/lags.csv`, `data/leads.csv`, `data/contemp.csv`). The pipeline will:  

1. **Load and preprocess** model results.  
2. **Filter models** to exclude implausible predictors, irrelevant scopes, and models with too many variables.  
3. **Apply variable encoding decisions** (count vs. binary) consistently.  
4. **Compute model-averaged estimates** with unconditional standard errors, confidence intervals, and inclusion weights.  
5. **Generate visualizations** (coefficient plots, predictor importance).  
6. **Automate a results section draft** written in plain, journal-ready language.  

---

## Data Handling

- **Exclude Oceania** from analyses due to low outbreak counts and limited representativeness.  
- Always drop intercepts, `scale(Year)`, and outbreak controls (`Outbreak_lag`, `Outbreak_lead`).  
- Apply **variable cap = 3 predictors per model** (excluding the above). This avoids overfitting and ensures interpretability.  

---

## Variable Encoding Rules

Use the **count vs. binary decision table** from the descriptive report:

- **ECOLOGICAL → binary** (few repeated events per year).  
- **GEOPHYSICAL → count** (many repeated events per year).  
- **INTERSTATE → binary** (only single events per year).  
- **POISONING → binary** (rare and seldom exceeds 1).  

Ensure only one encoding per hazard type is kept in the pipeline.  

---

## Variable Plausibility Filter

### Lags (prior shocks → outbreaks)
- **Keep:** Economic, Climatic, Geophysical, Interstate, Intrastate, Ecological.  
- **Drop:** Poisoning, Technological, Terrorism.  

### Leads (outbreaks → subsequent shocks)
- **Keep:** Economic, Interstate, Intrastate (optionally Terrorism if theory supports).  
- **Drop:** Geophysical, Climatic, Ecological, Poisoning, Technological.  

### Contemporaneous models
- Keep variables according to count/binary encoding rules.  
- Apply same plausibility logic when reporting results.  

### IMPORTANT RULE
- Any model that contains a variable listed in the **Drop** category must be **excluded entirely** from analysis.  
- Only models composed entirely of **Keep** variables (plus allowed controls) should proceed to model averaging.  

---

## Human-Readable Variable Labels

Replace technical column names with plain, interpretable labels in all **tables, plots, and automated text**:  

- `ECONOMIC_lag_avg` → *Economic shocks (prior)*  
- `ECONOMIC_lead_avg` → *Economic shocks (subsequent)*  
- `GEOPHYSICAL_lag_avg` → *Geophysical shocks (prior)*  
- `INTERSTATE__bin_lag_avg` → *Interstate conflict (prior)*  
- `INTRASTATE_lag_avg` → *Intrastate conflict (prior)*  
- `ECOLOGICAL__bin_lag_avg` → *Ecological disruption (prior)*  
- … mirror this naming convention across prior, subsequent, and contemporaneous models.  

---

## Model Averaging

For each scope (Global, Africa, America, Asia, Europe):  

1. Sort models by AICc.  
2. Compute **ΔAICc** and **Akaike weights**.  
3. Derive **model-averaged coefficients**.  
4. Calculate **unconditional SEs and 95% CIs**.  
5. Report **inclusion weights** (sum of model weights including variable).  

---

## Visualization Outputs

- **Coefficient forest plots:** model-averaged estimates with 95% CIs.  
- **Predictor importance plots:** variable inclusion weights.  
- **Scope-specific dashboards:** Africa, America, Asia, Europe, Global (excluding Oceania).  

---

## Automated Results Writing

Generate a markdown draft of results section of a journal article style report with:  

- **Descriptive statistics** (outbreak counts, deaths, rates by continent).  
- **Scope-specific findings:**  
  - Highlight variables with significant effects (CI excludes 0).  
  - Report inclusion weights.  
  - Translate into plain English (e.g., *“Prior economic shocks increased outbreak risk, while subsequent economic shocks reflected outbreak-driven economic disruption”*).  
- **Global overview:** Summarize cross-continent patterns and robustness.  
- **Tables:** Model-averaged coefficients, SEs, CIs, inclusion weights.  
- **Figures:** Coefficient plots and predictor importance.  

---

## Deliverables

- Filtered model datasets.  
- Tables (CSV + Markdown).  
- Figures (PNG).  
- Draft text for journal-ready results section.  
