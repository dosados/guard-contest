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
1. `mcc_code`: `0.03360816`
2. `event_descr`: `0.02128706`
3. `event_type_nm`: `0.02014056`
4. `max_amount_last_24h`: `0.01863168`
5. `device_freq`: `0.01737742`
6. `distinct_event_descr_count_last_24h_norm`: `0.01447661`
7. `std_delta_last_k`: `0.01384109`
8. `event_type_nm_share_in_suffix`: `0.01224410`
9. `operation_amt`: `0.01065952`
10. `transactions_per_span_hour`: `0.01060109`
11. `high_amount_ratio_last_24h`: `0.00969576`
12. `hour`: `0.00920554`
13. `mean_amount_last_3_transactions`: `0.00871898`
14. `amount_ratio_to_min_amount_24h`: `0.00803395`
15. `screen_size_cat`: `0.00779973`

## Top least important features
1. `mcc_freq_last_1h`: `-0.00146445`
2. `tr_amount`: `-0.00113383`
3. `transactions_last_10m`: `-0.00106143`
4. `session_duration`: `-0.00075755`
5. `sum_amount_last_10m`: `-0.00063061`
6. `delta_3`: `-0.00052407`
7. `is_amount_high`: `-0.00023358`
8. `transactions_last_5m`: `-0.00021399`
9. `transactions_in_session`: `-0.00007565`
10. `unique_browser_language_count_suffix`: `-0.00006906`
11. `amount_change_sign`: `-0.00005247`
12. `compromised_history_count_suffix`: `-0.00004877`
13. `seconds_since_last_compromised_tx`: `-0.00004698`
14. `web_rdp_connection`: `-0.00003636`
15. `is_new_timezone`: `-0.00001491`

## Practical interpretation
- Keep top features as priority signals in future iterations.
- For least-important features: test ablation/removal and compare PR-AUC + stability.
- Re-run this analysis after major feature-engineering or dataset rebuild.