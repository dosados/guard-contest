# Feature Importance Research

## Setup
- Models: `xgboost, lightgbm`
- Train rows used: `4000000`
- Validation rows used: `1200000`
- Permutation repeats: `10`
- Memory budget target: `20.0 GB`

## How to read importance
- Value = `baseline_pr_auc - pr_auc_after_shuffle(feature)`.
- Bigger positive value => feature is more useful for quality.
- Near zero => feature contributes little in this setup.
- Negative value => possible noise / interaction artifact; candidate for review.

## Charts
- Most important: `top_features.png`
- Least important: `least_features.png`
- Per-model heatmap: `importance_heatmap.png`

## Top most important features
1. `time_since_2nd_prev_transaction`: `0.00150089`
2. `mean_time_between_tx`: `0.00133769`
3. `event_descr`: `0.00108626`
4. `mcc_code`: `0.00099849`
5. `time_since_prev_to_mean_gap`: `0.00090833`
6. `channel_freq`: `0.00088291`
7. `device_freq`: `0.00072606`
8. `time_since_last_channel`: `0.00062432`
9. `time_since_prev_transaction`: `0.00055851`
10. `event_descr_freq`: `0.00055324`
11. `time_since_last_device`: `0.00055016`
12. `event_type_nm_freq`: `0.00030510`
13. `time_since_last_mcc`: `0.00029864`
14. `mcc_freq`: `0.00026718`
15. `std_time_between_tx`: `0.00024497`

## Top least important features
1. `day_of_week`: `-0.00001374`
2. `is_night_transaction`: `-0.00000716`
3. `is_weekend`: `-0.00000491`
4. `transactions_last_24h_norm`: `0.00000000`
5. `transactions_last_10m_to_1h`: `0.00000000`
6. `transactions_last_24h`: `0.00000000`
7. `transactions_last_1h_to_24h`: `0.00000000`
8. `transactions_last_1h_norm`: `0.00000000`
9. `sum_amount_last_24h`: `0.00000000`
10. `sum_amount_last_1h_norm`: `0.00000000`
11. `operation_amt`: `0.00000000`
12. `sum_1h_to_24h`: `0.00000000`
13. `sum_amount_last_1h`: `0.00000000`
14. `transactions_last_1h`: `0.00000000`
15. `transactions_last_10m_norm`: `0.00000000`

## Practical interpretation
- Keep top features as priority signals in future iterations.
- For least-important features: test ablation/removal and compare PR-AUC + stability.
- Re-run this analysis after major feature-engineering or dataset rebuild.