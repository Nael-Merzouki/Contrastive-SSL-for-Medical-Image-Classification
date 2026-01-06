Solid first run. AUROC is strong for 5 epochs on a small dataset. I think your write-up shows you grasp discrimination vs calibration. A few fixes will tighten reproducibility and correctness, and might set you up for Week 2–3 comparisons

# Corrections:
README accuracy mismatch:
You wrote “accuracy > 0.8,” but your test_metrics.json shows 0.763.

AUROC 0.825 shows the model ranks positives above negatives fairly well.
Calibration: ECE 0.239 means over-confident probabilities

# Suggestion (just based on my thoughts, you can refute me if you think I'm wrong)
Note on your claim: dataset imbalance mostly affects how accuracy is interpreted; ECE reflects match between predicted confidence and empirical accuracy per bin. Imbalance doesn’t directly cause high ECE, though it can interact with decision thresholds and class prevalence. Temperature scaling is still the right next step.
