# Feature Importance Research

## Setup
- Model: `xgboost`
- Train rows used: `12000000`
- Validation rows used: `3600000`
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
1. `event_descr`: `0.00701671`
2. `sum_amount_last_24h`: `0.00603672`
3. `operation_amt`: `0.00423519`
4. `device_freq`: `0.00422493`
5. `max_amount_last_24h`: `0.00386340`
6. `std_time_deltas`: `0.00336062`
7. `sum_amount_last_1h`: `0.00282691`
8. `amount_zscore`: `0.00275085`
9. `time_since_last_device_change`: `0.00257875`
10. `std_delta_last_k`: `0.00248231`
11. `mcc_code`: `0.00246410`
12. `high_amount_ratio_last_24h`: `0.00245588`
13. `hour`: `0.00146809`
14. `delta_2`: `0.00112910`
15. `time_since_prev_transaction`: `0.00103554`

## Top least important features
1. `session_mean_amount`: `-0.00012965`
2. `is_amount_high`: `-0.00007495`
3. `tr_amount`: `0.00000089`
4. `is_new_browser_language`: `0.00000152`
5. `is_compromised_device`: `0.00000249`
6. `is_new_device_tz_pair`: `0.00000277`
7. `log_1_plus_transactions_seen`: `0.00000583`
8. `is_device_switch`: `0.00000692`
9. `is_night_transaction`: `0.00000908`
10. `is_weekend`: `0.00001193`
11. `is_new_channel`: `0.00001431`
12. `transactions_in_session`: `0.00002013`
13. `phone_voip_call_state`: `0.00002967`
14. `is_new_device`: `0.00004675`
15. `is_new_mcc`: `0.00005452`

## Practical interpretation
- Keep top features as priority signals in future iterations.
- For least-important features: test ablation/removal and compare PR-AUC + stability.
- Re-run this analysis after major feature-engineering or dataset rebuild.