# Results

## Analytical Workflow

- Screening: we restricted analysis to the five focal scopes, removed Oceania, retained only outbreak outcomes, dropped intercept, scaled year, and outbreak controls, and capped models at three substantive predictors to preserve interpretability.
- Encoding: hazard variables followed predetermined rules so Ecological, Interstate, and Poisoning shocks entered as binary indicators while Geophysical shocks remained counts, preventing mixed representations within a hazard class.
- Plausibility: we discarded any model containing hazards flagged as implausible for each timing window (for example, technological or poisoning lags) so inference rests on theoretically defensible predictors only.
- Model averaging: within each scope we ranked eligible models by AICc, transformed delta AICc values into Akaike weights, and combined estimates with unconditional standard errors, confidence intervals, and inclusion weights to capture model-selection uncertainty.

## Descriptive Patterns

Outbreak activity remains uneven across continents once Oceania is excluded. The table summarises annual outbreaks, deaths, reported cases, and the share of observations with at least one outbreak.

| Continent | outbreaks | deaths | cases | observations | outbreak_rate |
| --- | --- | --- | --- | --- | --- |
| Global | 868 | 174018 | 4704083 | 3134 | 0.28 |
| Africa | 381 | 131866 | 2907941 | 1085 | 0.35 |
| America | 147 | 17427 | 605070 | 554 | 0.27 |
| Asia | 201 | 24261 | 924508 | 765 | 0.26 |
| Europe | 139 | 464 | 266564 | 730 | 0.19 |

## Scope-Specific Findings

### Global

- Prior shocks: Climatic shocks (prior) increased outbreak risk (beta=0.11, 95% CI [0.07, 0.14], inclusion weight 1.00).
![Figure 1. Global Prior shocks coefficients. Climatic shocks (prior) increased outbreak risk while the remaining predictors show weaker or uncertain effects.](outputs/figures/lags_Global_coefficients.png)
*Figure 1. Global Prior shocks coefficients. Climatic shocks (prior) increased outbreak risk while the remaining predictors show weaker or uncertain effects.*

![Figure 2. Global Prior shocks inclusion weights. Model weight concentrates on Climatic shocks (prior) (inclusion weight 1.00), indicating consistent selection across top models.](outputs/figures/lags_Global_inclusion.png)
*Figure 2. Global Prior shocks inclusion weights. Model weight concentrates on Climatic shocks (prior) (inclusion weight 1.00), indicating consistent selection across top models.*

- Subsequent shocks: Intrastate conflict (subsequent) increased outbreak risk (beta=0.31, 95% CI [0.12, 0.50], inclusion weight 1.00).
![Figure 3. Global Subsequent shocks coefficients. Intrastate conflict (subsequent) increased outbreak risk while the remaining predictors show weaker or uncertain effects.](outputs/figures/leads_Global_coefficients.png)
*Figure 3. Global Subsequent shocks coefficients. Intrastate conflict (subsequent) increased outbreak risk while the remaining predictors show weaker or uncertain effects.*

![Figure 4. Global Subsequent shocks inclusion weights. Model weight concentrates on Intrastate conflict (subsequent) (inclusion weight 1.00), indicating consistent selection across top models.](outputs/figures/leads_Global_inclusion.png)
*Figure 4. Global Subsequent shocks inclusion weights. Model weight concentrates on Intrastate conflict (subsequent) (inclusion weight 1.00), indicating consistent selection across top models.*

- Concurrent shocks: Climatic shocks (concurrent) increased outbreak risk (beta=0.08, 95% CI [0.05, 0.11], inclusion weight 1.00).
![Figure 5. Global Concurrent shocks coefficients. Climatic shocks (concurrent) increased outbreak risk while the remaining predictors show weaker or uncertain effects.](outputs/figures/contemp_Global_coefficients.png)
*Figure 5. Global Concurrent shocks coefficients. Climatic shocks (concurrent) increased outbreak risk while the remaining predictors show weaker or uncertain effects.*

![Figure 6. Global Concurrent shocks inclusion weights. Model weight concentrates on Climatic shocks (concurrent) (inclusion weight 1.00), indicating consistent selection across top models.](outputs/figures/contemp_Global_inclusion.png)
*Figure 6. Global Concurrent shocks inclusion weights. Model weight concentrates on Climatic shocks (concurrent) (inclusion weight 1.00), indicating consistent selection across top models.*


### Africa

- Prior shocks: No predictors retained with 95% confidence.
![Figure 7. Africa Prior shocks coefficients. Coefficient intervals overlap zero; Economic shocks (prior) has the largest point estimate but remains statistically uncertain.](outputs/figures/lags_Africa_coefficients.png)
*Figure 7. Africa Prior shocks coefficients. Coefficient intervals overlap zero; Economic shocks (prior) has the largest point estimate but remains statistically uncertain.*

![Figure 8. Africa Prior shocks inclusion weights. Economic shocks (prior) carries the highest support (inclusion weight 0.61) but alternative predictors still appear in the model set.](outputs/figures/lags_Africa_inclusion.png)
*Figure 8. Africa Prior shocks inclusion weights. Economic shocks (prior) carries the highest support (inclusion weight 0.61) but alternative predictors still appear in the model set.*

- Subsequent shocks: No predictors retained with 95% confidence.
![Figure 9. Africa Subsequent shocks coefficients. Coefficient intervals overlap zero; Interstate conflict (subsequent) has the largest point estimate but remains statistically uncertain.](outputs/figures/leads_Africa_coefficients.png)
*Figure 9. Africa Subsequent shocks coefficients. Coefficient intervals overlap zero; Interstate conflict (subsequent) has the largest point estimate but remains statistically uncertain.*

![Figure 10. Africa Subsequent shocks inclusion weights. Model weight concentrates on Interstate conflict (subsequent) (inclusion weight 0.92), indicating consistent selection across top models.](outputs/figures/leads_Africa_inclusion.png)
*Figure 10. Africa Subsequent shocks inclusion weights. Model weight concentrates on Interstate conflict (subsequent) (inclusion weight 0.92), indicating consistent selection across top models.*

- Concurrent shocks: No predictors retained with 95% confidence.
![Figure 11. Africa Concurrent shocks coefficients. Coefficient intervals overlap zero; Climatic shocks (concurrent) has the largest point estimate but remains statistically uncertain.](outputs/figures/contemp_Africa_coefficients.png)
*Figure 11. Africa Concurrent shocks coefficients. Coefficient intervals overlap zero; Climatic shocks (concurrent) has the largest point estimate but remains statistically uncertain.*

![Figure 12. Africa Concurrent shocks inclusion weights. Model weight concentrates on Climatic shocks (concurrent) (inclusion weight 0.89), indicating consistent selection across top models.](outputs/figures/contemp_Africa_inclusion.png)
*Figure 12. Africa Concurrent shocks inclusion weights. Model weight concentrates on Climatic shocks (concurrent) (inclusion weight 0.89), indicating consistent selection across top models.*


### America

- Prior shocks: Climatic shocks (prior) increased outbreak risk (beta=0.06, 95% CI [0.02, 0.11], inclusion weight 0.98).
![Figure 13. America Prior shocks coefficients. Climatic shocks (prior) increased outbreak risk while the remaining predictors show weaker or uncertain effects.](outputs/figures/lags_America_coefficients.png)
*Figure 13. America Prior shocks coefficients. Climatic shocks (prior) increased outbreak risk while the remaining predictors show weaker or uncertain effects.*

![Figure 14. America Prior shocks inclusion weights. Model weight concentrates on Climatic shocks (prior) (inclusion weight 0.98), indicating consistent selection across top models.](outputs/figures/lags_America_inclusion.png)
*Figure 14. America Prior shocks inclusion weights. Model weight concentrates on Climatic shocks (prior) (inclusion weight 0.98), indicating consistent selection across top models.*

- Subsequent shocks: No predictors retained with 95% confidence.
![Figure 15. America Subsequent shocks coefficients. Coefficient intervals overlap zero; Interstate conflict (subsequent) has the largest point estimate but remains statistically uncertain.](outputs/figures/leads_America_coefficients.png)
*Figure 15. America Subsequent shocks coefficients. Coefficient intervals overlap zero; Interstate conflict (subsequent) has the largest point estimate but remains statistically uncertain.*

![Figure 16. America Subsequent shocks inclusion weights. Model weight concentrates on Interstate conflict (subsequent) (inclusion weight 0.87), indicating consistent selection across top models.](outputs/figures/leads_America_inclusion.png)
*Figure 16. America Subsequent shocks inclusion weights. Model weight concentrates on Interstate conflict (subsequent) (inclusion weight 0.87), indicating consistent selection across top models.*

- Concurrent shocks: No eligible models.

### Asia

- Prior shocks: Climatic shocks (prior) increased outbreak risk (beta=0.13, 95% CI [0.07, 0.19], inclusion weight 1.00).
![Figure 17. Asia Prior shocks coefficients. Climatic shocks (prior) increased outbreak risk while the remaining predictors show weaker or uncertain effects.](outputs/figures/lags_Asia_coefficients.png)
*Figure 17. Asia Prior shocks coefficients. Climatic shocks (prior) increased outbreak risk while the remaining predictors show weaker or uncertain effects.*

![Figure 18. Asia Prior shocks inclusion weights. Model weight concentrates on Climatic shocks (prior) (inclusion weight 1.00), indicating consistent selection across top models.](outputs/figures/lags_Asia_inclusion.png)
*Figure 18. Asia Prior shocks inclusion weights. Model weight concentrates on Climatic shocks (prior) (inclusion weight 1.00), indicating consistent selection across top models.*

- Subsequent shocks: No predictors retained with 95% confidence.
![Figure 19. Asia Subsequent shocks coefficients. Coefficient intervals overlap zero; Interstate conflict (subsequent) has the largest point estimate but remains statistically uncertain.](outputs/figures/leads_Asia_coefficients.png)
*Figure 19. Asia Subsequent shocks coefficients. Coefficient intervals overlap zero; Interstate conflict (subsequent) has the largest point estimate but remains statistically uncertain.*

![Figure 20. Asia Subsequent shocks inclusion weights. Model weight concentrates on Interstate conflict (subsequent) (inclusion weight 0.78), indicating consistent selection across top models.](outputs/figures/leads_Asia_inclusion.png)
*Figure 20. Asia Subsequent shocks inclusion weights. Model weight concentrates on Interstate conflict (subsequent) (inclusion weight 0.78), indicating consistent selection across top models.*

- Concurrent shocks: No predictors retained with 95% confidence.
![Figure 21. Asia Concurrent shocks coefficients. Coefficient intervals overlap zero; Climatic shocks (concurrent) has the largest point estimate but remains statistically uncertain.](outputs/figures/contemp_Asia_coefficients.png)
*Figure 21. Asia Concurrent shocks coefficients. Coefficient intervals overlap zero; Climatic shocks (concurrent) has the largest point estimate but remains statistically uncertain.*

![Figure 22. Asia Concurrent shocks inclusion weights. Model weight concentrates on Climatic shocks (concurrent) (inclusion weight 0.87), indicating consistent selection across top models.](outputs/figures/contemp_Asia_inclusion.png)
*Figure 22. Asia Concurrent shocks inclusion weights. Model weight concentrates on Climatic shocks (concurrent) (inclusion weight 0.87), indicating consistent selection across top models.*


### Europe

- Prior shocks: Climatic shocks (prior) increased outbreak risk (beta=0.31, 95% CI [0.20, 0.43], inclusion weight 1.00).
![Figure 23. Europe Prior shocks coefficients. Climatic shocks (prior) increased outbreak risk while the remaining predictors show weaker or uncertain effects.](outputs/figures/lags_Europe_coefficients.png)
*Figure 23. Europe Prior shocks coefficients. Climatic shocks (prior) increased outbreak risk while the remaining predictors show weaker or uncertain effects.*

![Figure 24. Europe Prior shocks inclusion weights. Model weight concentrates on Climatic shocks (prior) (inclusion weight 1.00), indicating consistent selection across top models.](outputs/figures/lags_Europe_inclusion.png)
*Figure 24. Europe Prior shocks inclusion weights. Model weight concentrates on Climatic shocks (prior) (inclusion weight 1.00), indicating consistent selection across top models.*

- Subsequent shocks: Intrastate conflict (subsequent) increased outbreak risk (beta=1.07, 95% CI [0.22, 1.92], inclusion weight 0.97).
![Figure 25. Europe Subsequent shocks coefficients. Intrastate conflict (subsequent) increased outbreak risk while the remaining predictors show weaker or uncertain effects.](outputs/figures/leads_Europe_coefficients.png)
*Figure 25. Europe Subsequent shocks coefficients. Intrastate conflict (subsequent) increased outbreak risk while the remaining predictors show weaker or uncertain effects.*

![Figure 26. Europe Subsequent shocks inclusion weights. Model weight concentrates on Intrastate conflict (subsequent) (inclusion weight 0.97), indicating consistent selection across top models.](outputs/figures/leads_Europe_inclusion.png)
*Figure 26. Europe Subsequent shocks inclusion weights. Model weight concentrates on Intrastate conflict (subsequent) (inclusion weight 0.97), indicating consistent selection across top models.*

- Concurrent shocks: No predictors retained with 95% confidence.
![Figure 27. Europe Concurrent shocks coefficients. Coefficient intervals overlap zero; Climatic shocks (concurrent) has the largest point estimate but remains statistically uncertain.](outputs/figures/contemp_Europe_coefficients.png)
*Figure 27. Europe Concurrent shocks coefficients. Coefficient intervals overlap zero; Climatic shocks (concurrent) has the largest point estimate but remains statistically uncertain.*

![Figure 28. Europe Concurrent shocks inclusion weights. Model weight concentrates on Climatic shocks (concurrent) (inclusion weight 0.90), indicating consistent selection across top models.](outputs/figures/contemp_Europe_inclusion.png)
*Figure 28. Europe Concurrent shocks inclusion weights. Model weight concentrates on Climatic shocks (concurrent) (inclusion weight 0.90), indicating consistent selection across top models.*


## Global Overview

Global estimates highlight: Climatic shocks (prior) increased outbreak risk (beta=0.11, 95% CI [0.07, 0.14], inclusion weight 1.00); Climatic shocks (concurrent) increased outbreak risk (beta=0.08, 95% CI [0.05, 0.11], inclusion weight 1.00); Intrastate conflict (subsequent) increased outbreak risk (beta=0.31, 95% CI [0.12, 0.50], inclusion weight 1.00).

Cross-continent patterns show that:
- Prior shocks: Climatic shocks (prior) showed positive associations in 3 scope(s).
- Subsequent shocks: Intrastate conflict (subsequent) showed positive associations in 1 scope(s).

## Tables and Figures

- Model-averaged estimates: `outputs/tables/model_averaged_estimates.csv` and `outputs/tables/model_averaged_estimates.md`.
- Model weights: `outputs/tables/model_weights.csv`.
- Descriptive summary: `outputs/tables/descriptive_stats.csv`.
- Coefficient and inclusion-weight plots are saved in `outputs/figures/` for each scope and shock timing.

All analyses exclude Oceania and remove non-plausible predictors according to `AGENTS.md`.