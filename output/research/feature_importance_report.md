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
1. `event_descr`: `0.00590829`
2. `sum_amount_last_24h`: `0.00328486`
3. `max_amount_last_24h`: `0.00310900`
4. `mcc_code`: `0.00295186`
5. `event_type_nm`: `0.00293905`
6. `sum_amount_last_1h`: `0.00286409`
7. `operation_amt`: `0.00284280`
8. `device_freq`: `0.00253971`
9. `std_time_deltas`: `0.00236550`
10. `time_since_last_device_change`: `0.00210789`
11. `std_delta_last_k`: `0.00203804`
12. `hour`: `0.00178820`
13. `high_amount_ratio_last_24h`: `0.00172302`
14. `amount_percentile_rank`: `0.00146055`
15. `delta_3`: `0.00107583`

## Top least important features
1. `delta_2`: `-0.00050902`
2. `tr_amount`: `-0.00038155`
3. `event_type_nm_freq_last_1h`: `-0.00037988`
4. `session_duration`: `-0.00034167`
5. `transactions_in_session`: `-0.00031471`
6. `delta_1`: `-0.00028232`
7. `amount_relative_to_mcc_median_5_days`: `-0.00018164`
8. `transactions_last_24h`: `-0.00017666`
9. `transactions_last_10m`: `-0.00010583`
10. `is_new_browser_language`: `-0.00001039`
11. `is_new_device_tz_pair`: `-0.00000058`
12. `is_compromised_device`: `0.00000124`
13. `is_new_mcc`: `0.00000831`
14. `is_weekend`: `0.00002933`
15. `is_night_transaction`: `0.00002976`

## Practical interpretation
- Keep top features as priority signals in future iterations.
- For least-important features: test ablation/removal and compare PR-AUC + stability.
- Re-run this analysis after major feature-engineering or dataset rebuild.