# Feature Importance Research

## Setup
- Model: `xgboost`
- Train rows used: `2000000`
- Validation rows used: `600000`
- Permutation repeats: `5`
- Memory budget target: `20.0 GB`

## How to read importance
- Value = `baseline_pr_auc - pr_auc_after_shuffle(feature)`.
- Bigger positive value => feature is more useful for quality.
- Near zero => feature contributes little in this setup.
- Negative value => possible noise / interaction artifact; candidate for review.

## Charts
- Most important: `top_features.png`
- Least important: `least_features.png`

## Top most important features
1. `sum_amount_last_24h`: `0.00129731`
2. `amount_to_median`: `0.00085639`
3. `std_delta_last_k`: `0.00079093`
4. `time_since_prev_transaction`: `0.00076716`
5. `event_descr`: `0.00066358`
6. `transactions_last_24h`: `0.00060716`
7. `device_freq`: `0.00058613`
8. `mcc_code`: `0.00057618`
9. `delta_1`: `0.00038202`
10. `sum_amount_last_1h`: `0.00022234`
11. `operation_amt`: `0.00022183`
12. `amount_zscore`: `0.00012700`
13. `delta_3`: `0.00010039`
14. `amount_zscore_x_is_new_mcc`: `0.00008965`
15. `session_duration`: `0.00008590`

## Top least important features
1. `amount_ratio_prev`: `-0.00019080`
2. `max_amount_last_24h`: `-0.00018395`
3. `day_of_week`: `-0.00017389`
4. `tr_amount`: `-0.00010841`
5. `transactions_last_1h`: `-0.00010369`
6. `trend_mean_last_3_to_10`: `-0.00009295`
7. `session_mean_amount`: `-0.00008701`
8. `time_since_last_device_change`: `-0.00008366`
9. `amount_diff_prev`: `-0.00007729`
10. `log_1_plus_transactions_seen`: `-0.00007686`
11. `cv_delta_last_k`: `-0.00006024`
12. `timezone_missing`: `-0.00002073`
13. `phone_voip_call_state`: `-0.00001802`
14. `is_device_switch`: `-0.00001203`
15. `is_amount_high`: `-0.00000950`

## Practical interpretation
- Keep top features as priority signals in future iterations.
- For least-important features: test ablation/removal and compare PR-AUC + stability.
- Re-run this analysis after major feature-engineering or dataset rebuild.