# Model card: Credit Risk Intelligence demo

## Intended use
Educational prototyping and model-risk workflow demonstration. The synthetic fallback
is not representative of a lending population and must not be used for production credit
decisions, adverse-action notices, or pricing.

## Data and method
`make_synthetic_data` generates a deterministic, seeded dataset. A user-provided CSV can
be used instead. Numeric values are median-imputed/scaled and categories one-hot encoded
inside a pipeline. The target is `default` (1 = default). IDs are intentionally excluded.

## Evaluation
Train/validation/test are stratified and disjoint. The report includes ROC-AUC, PR-AUC,
Brier calibration loss, threshold metrics, expected FP/FN cost, and age-group slices.
Calibration is evaluated with sigmoid cross-validation. SHAP is optional; permutation
importance is the fallback.

## Limitations, fairness and governance
Age-group slices are descriptive monitoring, not proof of fairness or legal compliance.
Check equal opportunity, calibration, subgroup sample sizes, drift, missingness, and
performance by geography/product before deployment. Obtain consent and document lawful
basis, retention, access control, human review, explainability, adverse-action reasons,
model versioning, rollback, and periodic independent validation. Never use protected
attributes for automated decisions without a documented legal and governance review.
