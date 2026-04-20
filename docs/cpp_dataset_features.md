# Признаки модели из C++ (`build_dataset` + `build_global_aggregates`)

В матрицу модели входят **244** float-признака: **98** локальных + **146** глобальных. Порядок как в `dataset_cpp/build_dataset.cpp`, `dataset_cpp/global_category_features.hpp` и `shared/features.py` (`FEATURE_NAMES`).

Глобальные признаки считаются **по разным осям агрегирования** (MCC, канал, timezone×валюта и т.д.) на полной выборке pretrain+train; **отдельных календарных окон** (1h / 24h / …) у глобального блока в коде **нет**. Календарные окна есть только у **локальных** имён (ниже).

---

## Локальные признаки (98)

### Окно **1 час** до `event_dttm`

- `sum_amount_last_1h`
- `transactions_last_1h`
- `event_descr_freq_last_1h`
- `event_type_nm_freq_last_1h`

### Окно **6 часов**

- `event_descr_freq_last_6h`
- `event_type_nm_freq_last_6h`
- `mcc_freq_last_6h`

### Окно **24 часа**

- `transactions_last_24h`
- `sum_amount_last_24h`
- `max_amount_last_24h`
- `event_descr_freq_last_24h`
- `event_type_nm_freq_last_24h`
- `mcc_freq_last_24h`
- `high_amount_ratio_last_24h`
- `distinct_hours_active_last_24h`
- `amount_ratio_to_min_amount_24h`
- `web_rdp_count_last_24h`
- `voip_flag_count_last_24h`
- `distinct_event_descr_count_last_24h_norm`

### Окно **5 суток**

- `amount_relative_to_mcc_median_5_days`
- `mcc_amount_std_same_5d`

### Окно **7 суток**

- `weekend_transaction_share_last_7d`

### Окно **последние 20 транзакций** (по счёту в хвосте окна)

- `mcc_switch_count_last_20_tx`
- `device_switch_count_last_20_tx`
- `channel_switch_count_last_20_tx`
- `session_switch_count_last_20_tx`

### Окно **последние транзакции пользователя** (только уже накопленные в deque до текущей строки)

- `operation_amt`
- `log_1_plus_transactions_seen`
- `amount_zscore`
- `is_new_device`
- `is_new_mcc`
- `is_new_channel`
- `phone_voip_call_state`
- `hour`
- `day_of_week`
- `is_night_transaction`
- `is_weekend`
- `time_since_prev_transaction`
- `timezone_missing`
- `trend_mean_last_3_to_10`
- `amount_percentile_rank`
- `std_time_deltas`
- `is_device_switch`
- `is_mcc_switch`
- `session_mean_amount`
- `device_freq`
- `delta_1`
- `delta_2`
- `acceleration_delta_1_over_2`
- `std_delta_last_k`
- `time_since_last_device_change`
- `time_since_last_mcc_change`
- `mcc_event_descr_pair_new`
- `amount_ratio_to_window_median`
- `amount_iqr_normalized`
- `amount_cv_in_window`
- `unique_mcc_count_suffix`
- `unique_device_key_count_suffix`
- `unique_channel_key_count_suffix`
- `unique_timezone_count_suffix`
- `mean_amount_last_3_transactions`
- `channel_relative_freq`
- `timezone_relative_freq`
- `browser_language_relative_freq`
- `event_type_nm_share_in_suffix`
- `mcc_consecutive_streak_length`
- `mean_gap_seconds_last_5_intervals`
- `suffix_time_span_hours_log1p`
- `transactions_per_span_hour`
- `battery_level`
- `developer_tools_flag`
- `accept_lang_browser_lang_mismatch`
- `amount_diff_prev`
- `amount_ratio_prev`
- `amount_increase_streak_suffix`
- `amount_decrease_streak_suffix`
- `is_new_session_id`
- `seconds_since_session_start_in_window`
- `mcc_same_count_suffix`
- `sum_amount_same_mcc_suffix`
- `mean_amount_same_mcc_suffix`
- `std_amount_same_mcc_suffix`
- `min_amount_same_mcc_suffix`
- `max_amount_same_mcc_suffix`
- `share_mcc_cnt_suffix`
- `share_mcc_sum_suffix`
- `amount_ratio_mean_same_mcc`
- `amount_minus_mean_same_mcc`
- `zscore_amount_same_mcc`
- `days_since_last_same_mcc`
- `mcc_channel_same_count_suffix`
- `is_new_mcc_channel_pair`
- `mcc_device_same_count_suffix`
- `is_new_mcc_device_pair`
- `mcc_rdp_count_suffix`
- `share_mcc_rdp_suffix`
- `hour_mean_same_mcc_suffix`
- `hour_std_same_mcc_suffix`

---

## Глобальные признаки (146): разные **оси** (одна популяция, без календарных окон)


### Ось **MCC** (20)

- `global_mean_amount_mcc`
- `global_std_amount_mcc`
- `global_median_amount_mcc`
- `global_q25_mcc`
- `global_q75_mcc`
- `global_q95_mcc`
- `global_cnt_mcc`
- `global_cv_mcc`
- `fraud_rate_mcc`
- `fraud_count_mcc`
- `train_total_count_mcc`
- `woe_mcc`
- `amount_ratio_global_mean_mcc`
- `global_zscore_mcc`
- `inv_global_cnt_mcc`
- `global_cnt_clean_mcc`
- `global_q90_mcc`
- `global_q99_mcc`
- `global_zscore_median_iqr_mcc`
- `amount_percentile_in_mcc`

### Ось **channel_indicator_type + subtype** (20)

- `global_mean_amount_channel`
- `global_std_amount_channel`
- `global_median_amount_channel`
- `global_q25_channel`
- `global_q75_channel`
- `global_q95_channel`
- `global_cnt_channel`
- `global_cv_channel`
- `fraud_rate_channel`
- `fraud_count_channel`
- `train_total_count_channel`
- `woe_channel`
- `amount_ratio_global_mean_channel`
- `global_zscore_channel`
- `inv_global_cnt_channel`
- `global_cnt_clean_channel`
- `global_q90_channel`
- `global_q99_channel`
- `amount_z_vs_channel_median`
- `amount_percentile_in_channel`

### Ось **timezone × currency_iso_cd** (19)

- `global_mean_amount_tz_currency`
- `global_std_amount_tz_currency`
- `global_median_amount_tz_currency`
- `global_q25_tz_currency`
- `global_q75_tz_currency`
- `global_q95_tz_currency`
- `global_cnt_tz_currency`
- `global_cv_tz_currency`
- `fraud_rate_tz_currency`
- `fraud_count_tz_currency`
- `train_total_count_tz_currency`
- `woe_tz_currency`
- `global_cnt_clean_tz_currency`
- `global_q90_tz_currency`
- `global_q99_tz_currency`
- `amount_z_vs_tz_median`
- `amount_percentile_in_tz_currency`
- `inv_global_cnt_tz_currency`
- `global_zscore_median_iqr_tz_currency`

### Ось **event_type_nm × currency_iso_cd** (22)

- `global_mean_amount_event_type_currency`
- `global_std_amount_event_type_currency`
- `global_median_amount_event_type_currency`
- `global_q25_event_type_currency`
- `global_q75_event_type_currency`
- `global_q95_event_type_currency`
- `global_cnt_event_type_currency`
- `global_cv_event_type_currency`
- `fraud_rate_event_type_currency`
- `fraud_count_event_type_currency`
- `train_total_count_event_type_currency`
- `woe_event_type_currency`
- `global_cnt_clean_event_type_currency`
- `global_q90_event_type_currency`
- `global_q99_event_type_currency`
- `global_type_frequency_log_event_type_currency`
- `amount_ratio_global_mean_event_type_currency`
- `global_zscore_event_type_currency`
- `inv_global_cnt_event_type_currency`
- `amount_z_vs_event_type_median`
- `amount_percentile_in_event_type_currency`
- `global_zscore_median_iqr_event_type_currency`

### Совместные оси **MCC × channel / currency / timezone** и **channel × MCC** (5)

- `channel_rarity_neglog_in_mcc`
- `currency_freq_in_mcc`
- `timezone_freq_in_mcc`
- `surprise_mcc_given_channel_neglog`
- `mcc_not_in_channel_top3_flag`

### Ось **event_descr** (20)

- `global_mean_amount_event_descr`
- `global_std_amount_event_descr`
- `global_median_amount_event_descr`
- `global_q25_event_descr`
- `global_q75_event_descr`
- `global_q95_event_descr`
- `global_cnt_event_descr`
- `global_cv_event_descr`
- `fraud_rate_event_descr`
- `fraud_count_event_descr`
- `train_total_count_event_descr`
- `woe_event_descr`
- `amount_ratio_global_mean_event_descr`
- `global_zscore_event_descr`
- `inv_global_cnt_event_descr`
- `global_cnt_clean_event_descr`
- `global_q90_event_descr`
- `global_q99_event_descr`
- `amount_z_vs_event_descr_median`
- `amount_percentile_in_event_descr`

### Ось **pos_cd** (20)

- `global_mean_amount_pos_cd`
- `global_std_amount_pos_cd`
- `global_median_amount_pos_cd`
- `global_q25_pos_cd`
- `global_q75_pos_cd`
- `global_q95_pos_cd`
- `global_cnt_pos_cd`
- `global_cv_pos_cd`
- `fraud_rate_pos_cd`
- `fraud_count_pos_cd`
- `train_total_count_pos_cd`
- `woe_pos_cd`
- `amount_ratio_global_mean_pos_cd`
- `global_zscore_pos_cd`
- `inv_global_cnt_pos_cd`
- `global_cnt_clean_pos_cd`
- `global_q90_pos_cd`
- `global_q99_pos_cd`
- `amount_z_vs_pos_cd_median`
- `amount_percentile_in_pos_cd`

### Ось **timezone** без валюты (20)

- `global_mean_amount_tz_alone`
- `global_std_amount_tz_alone`
- `global_median_amount_tz_alone`
- `global_q25_tz_alone`
- `global_q75_tz_alone`
- `global_q95_tz_alone`
- `global_cnt_tz_alone`
- `global_cv_tz_alone`
- `fraud_rate_tz_alone`
- `fraud_count_tz_alone`
- `train_total_count_tz_alone`
- `woe_tz_alone`
- `amount_ratio_global_mean_tz_alone`
- `global_zscore_tz_alone`
- `inv_global_cnt_tz_alone`
- `global_cnt_clean_tz_alone`
- `global_q90_tz_alone`
- `global_q99_tz_alone`
- `amount_z_vs_tz_alone_median`
- `amount_percentile_in_tz_alone`
