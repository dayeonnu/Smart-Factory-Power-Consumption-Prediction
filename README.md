# Smart-Factory-Power-Consumption-Prediction
Data analysis and prediction project for smart factory power consumption

# 데이터 로딩
df = pd.read_csv("rtu_1224.csv")

# 시간 컬럼 파싱
df["hour_dt"] = pd.to_datetime(df["hour_dt"])

# 실제 전력 사용 구간만 분석
df = df[df["hourly_usage"] > 0].copy()
# 공통 기준 (고정)
Q99 = 3072 

# anomaly 정의
df["anomaly"] = df["hourly_usage"] > Q99

# q99 초과분 정의
df["excess"] = np.where(df["anomaly"], df["hourly_usage"] - Q99, 0)

#설비별 관측 시간 분포 검토
obs_table = (
    df.groupby("equipment_id")
      .size()
      .reset_index(name="n_total_hours")
      .sort_values("n_total_hours")
)

#설비별 리스크 지표 계산 (risk table v1)
risk_v1 = (
    df.groupby("equipment_id")
      .agg(
          n_total_hours=("hourly_usage", "count"),
          n_anomaly=("anomaly", "sum"),
          anomaly_rate=("anomaly", "mean"),
          severity=("excess", lambda x: x[x > 0].mean())
      )
      .reset_index()
)

# risk_score 계산
risk_v1["risk_score"] = risk_v1["n_anomaly"] * risk_v1["severity"]

# risk_score 기준 정렬
risk_v1 = risk_v1.sort_values("risk_score", ascending=False)

risk_v1

#Top 설비 sanity check
TOP_N = 5
top_risk = risk_v1.head(TOP_N)

top_risk

#민감도 점검
# q95 계산
Q95 = df["hourly_usage"].quantile(0.95)

# q95 기준 anomaly
df["anomaly_q95"] = df["hourly_usage"] > Q95

# 설비별 anomaly 수 비교
risk_q95 = (
    df.groupby("equipment_id")["anomaly_q95"]
      .sum()
      .reset_index(name="n_anomaly_q95")
)

sensitivity_table = (
    risk_v1[["equipment_id", "n_anomaly"]]
    .merge(risk_q95, on="equipment_id", how="left")
    .sort_values("n_anomaly", ascending=False)
)

sensitivity_table
#빈도형 vs강도형 설비 구분
# 중앙값 기준
freq_median = risk_v1["anomaly_rate"].median()
sev_median = risk_v1["severity"].median()

risk_v1["risk_type"] = np.where(
    (risk_v1["anomaly_rate"] > freq_median) &
    (risk_v1["severity"] <= sev_median),
    "빈도형",
    "강도형"
)

risk_v1[[
    "equipment_id",
    "n_anomaly",
    "anomaly_rate",
    "severity",
    "risk_score",
    "risk_type"
]]

#q95 기준 anomaly 계산
# q95 기준값
Q95 = df["hourly_usage"].quantile(0.95)

# q95 anomaly 정의
df["anomaly_q95"] = df["hourly_usage"] > Q95

print(f"Q95 threshold: {Q95:.2f}")
print(f"Anomaly rate (q95): {df['anomaly_q95'].mean():.3f}")

#Top 설비 교집합 확인 (q99 vs q95)
# q99 기준 Top 설비
top_q99 = set(
    risk_v1.sort_values("risk_score", ascending=False)
           .head(5)["equipment_id"]
)

# q95 기준 anomaly 수 집계
risk_q95 = (
    df.groupby("equipment_id")["anomaly_q95"]
      .sum()
      .reset_index(name="n_anomaly_q95")
)

# q95 기준 Top 설비
top_q95 = set(
    risk_q95.sort_values("n_anomaly_q95", ascending=False)
            .head(5)["equipment_id"]
)

# 교집합
intersection = top_q99 & top_q95
intersection

#비용·탄소 초과분 계산 (q99 기준)
# 단가 설정
COST_PER_KWH = 180
CARBON_PER_KWH = 0.424

# q99 초과분
df["excess_q99"] = np.where(df["anomaly"], df["hourly_usage"] - Q99, 0)

# 비용·탄소 계산
df["excess_cost"] = df["excess_q99"] * COST_PER_KWH
df["excess_carbon"] = df["excess_q99"] * CARBON_PER_KWH

#초과분 계산이 제대로 됐는지 확인
df.loc[df["anomaly"], ["hourly_usage", "excess_q99"]].head()

#설비별 비용·탄소 영향 집계
cost_carbon_table = (
    df.groupby("equipment_id")
      .agg(
          total_excess_cost=("excess_cost", "sum"),
          total_excess_carbon=("excess_carbon", "sum")
      )
      .reset_index()
      .sort_values("total_excess_cost", ascending=False)
)

cost_carbon_table.head()

#리스크 테이블 v2 생성
risk_v2 = (
    df.groupby("equipment_id")
      .agg(
          n_total_hours=("hourly_usage", "count"),
          n_anomaly=("anomaly", "sum"),
          anomaly_rate=("anomaly", "mean"),
          severity=("excess_q99", lambda x: x[x > 0].mean()),
          risk_score=("excess_q99", lambda x: x[x > 0].mean() * x.gt(0).sum()),
          total_excess_cost=("excess_cost", "sum"),
          total_excess_carbon=("excess_carbon", "sum")
      )
      .reset_index()
      .sort_values("risk_score", ascending=False)
)

risk_v2

#시각화
plt.figure(figsize=(8, 4))
top10 = risk_v2.head(10)

plt.bar(top10["equipment_id"].astype(str), top10["risk_score"])
plt.xlabel("Equipment ID")
plt.ylabel("Risk Score")
plt.title("Top 10 Equipment by Risk Score")
plt.show()

#anomaly_rate vs severity Scatter
plt.figure(figsize=(6, 5))

plt.scatter(
    risk_v2["anomaly_rate"],
    risk_v2["severity"]
)

plt.xlabel("Anomaly Rate")
plt.ylabel("Severity")
plt.title("Frequency vs Severity")
plt.show()

#Top 설비 anomaly Timeline
top_equipment = risk_v2.head(3)["equipment_id"].tolist()

plt.figure(figsize=(10, 4))

for eid in top_equipment:
    temp = df[(df["equipment_id"] == eid) & (df["anomaly"])]
    plt.scatter(temp["hour_dt"], temp["hourly_usage"], label=f"Equip {eid}", s=10)

plt.legend()
plt.xlabel("Time")
plt.ylabel("Hourly Usage")
plt.title("Anomaly Timeline (Top Equipment)")
plt.show()

# A단계와 동일: 실제 사용(>0) 구간만 분석
df = df[df["hourly_usage"] > 0].copy()

df["anomaly"] = df["hourly_usage"] > Q99
df["excess_q99"] = np.where(df["anomaly"], df["hourly_usage"] - Q99, 0)

# start 기준 요약용 파생 변수
df["hour_of_day"] = df["hour_dt"].dt.hour

#이벤트(event_id) 생성 로직(핵심)
df = df.sort_values(["equipment_id", "hour_dt"]).reset_index(drop=True)

prev_eid  = df["equipment_id"].shift(1)
prev_time = df["hour_dt"].shift(1)
prev_anom = df["anomaly"].shift(1).fillna(False)

is_new_equip = df["equipment_id"] != prev_eid
is_gap = (df["hour_dt"] - prev_time) != pd.Timedelta(hours=1)

# anomaly 연속구간 시작점
start_seq = df["anomaly"] & (is_new_equip | is_gap | (~prev_anom))

#event_id 부여
df["event_seq"] = start_seq.groupby(df["equipment_id"]).cumsum()
df["event_id"] = np.where(df["anomaly"], df["event_seq"], np.nan)

#events_table 생성(Top5 설비)
events_table = (
    df[df["anomaly"] & df["equipment_id"].isin(TOP5)]
    .groupby(["equipment_id", "event_id"])
    .agg(
        start_time=("hour_dt", "min"),
        end_time=("hour_dt", "max"),
        duration_hours=("hour_dt", lambda x: int((x.max() - x.min()).total_seconds()/3600) + 1),
        peak_usage=("hourly_usage", "max"),
        peak_excess=("excess_q99", "max"),
        event_excess_sum=("excess_q99", "sum")
    )
    .reset_index()
)

events_table["event_id"] = events_table["event_id"].astype(int)

# start_time 기준 weekday/hour_of_day 결합
events_table = events_table.merge(
    df[["equipment_id", "event_id", "hour_dt", "weekday", "hour_of_day"]]
      .rename(columns={"hour_dt": "start_time"})
      .drop_duplicates(subset=["equipment_id", "event_id", "start_time"]),
    on=["equipment_id", "event_id", "start_time"],
    how="left"
)

#결과 테이블 확인
events_table.head(10)

COST_PER_KWH = 180      # 원/kWh
CARBON_PER_KWH = 0.424  # kgCO2/kWh


events_table["event_cost"] = events_table["event_excess_sum"] * COST_PER_KWH
events_table["event_carbon"] = events_table["event_excess_sum"] * CARBON_PER_KWH

# 시간당 평균 초과강도
events_table["event_excess_mean"] = events_table["event_excess_sum"] / events_table["duration_hours"]

#평가지표 함수
def smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom != 0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100



def calc_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "SMAPE": smape(y_true, y_pred)
    }


def calc_tail_metrics(y_true, y_pred, q95, q99):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    out = {}

    mask_95_99 = (y_true >= q95) & (y_true < q99)
    mask_99p   = (y_true >= q99)

    if mask_95_99.sum() > 0:
        out["q95_q99_MAE"] = mean_absolute_error(
            y_true[mask_95_99],
            y_pred[mask_95_99]
        )
        out["q95_q99_SMAPE"] = smape(
            y_true[mask_95_99],
            y_pred[mask_95_99]
        )
    else:
        out["q95_q99_MAE"] = np.nan
        out["q95_q99_SMAPE"] = np.nan

    if mask_99p.sum() > 0:
        out["q99p_MAE"] = mean_absolute_error(
            y_true[mask_99p],
            y_pred[mask_99p]
        )
        out["q99p_SMAPE"] = smape(
            y_true[mask_99p],
            y_pred[mask_99p]
        )
    else:
        out["q99p_MAE"] = np.nan
        out["q99p_SMAPE"] = np.nan

    return out

    #Train/Valid 분할(시간 기준)
    df = pd.read_csv("rtu_1224.csv")
df[TIME_COL] = pd.to_datetime(df[TIME_COL])

# 실제 사용 구간만 사용
df = df[df[TARGET] > 0].copy()

# 시간 정렬
df = df.sort_values([TIME_COL, "equipment_id"]).reset_index(drop=True)

# 기간 정의
TRAIN_START = pd.to_datetime("2024-12-01")
TRAIN_END   = pd.to_datetime("2025-03-31")

VALID_START = pd.to_datetime("2025-04-01")
VALID_END   = pd.to_datetime("2025-04-30")

BUFFER_HOURS = 24
BUFFER_START = VALID_START - pd.Timedelta(hours=BUFFER_HOURS)

# 분할
train_df = df[(df[TIME_COL] >= TRAIN_START) & (df[TIME_COL] <= TRAIN_END)].copy()
valid_df = df[(df[TIME_COL] >= BUFFER_START) & (df[TIME_COL] <= VALID_END)].copy()

# 공통 설비만 유지
common_equips = set(train_df["equipment_id"]) & set(valid_df["equipment_id"])
train_df = train_df[train_df["equipment_id"].isin(common_equips)]
valid_df = valid_df[valid_df["equipment_id"].isin(common_equips)]

# 산출물: split 요약표
split_summary = pd.DataFrame({
    "dataset": ["train", "valid(+buffer)", "valid(eval)"],
    "start_time": [
        train_df[TIME_COL].min(),
        valid_df[TIME_COL].min(),
        valid_df[valid_df[TIME_COL] >= VALID_START][TIME_COL].min()
    ],
    "end_time": [
        train_df[TIME_COL].max(),
        valid_df[TIME_COL].max(),
        valid_df[valid_df[TIME_COL] >= VALID_START][TIME_COL].max()
    ],
    "n_rows": [
        len(train_df),
        len(valid_df),
        len(valid_df[valid_df[TIME_COL] >= VALID_START])
    ],
    "n_equipment": [
        train_df["equipment_id"].nunique(),
        valid_df["equipment_id"].nunique(),
        valid_df["equipment_id"].nunique()
    ]
})

print(split_summary)

#피처 생성 파이프라인
df_feat = df.copy()
#시간/범주 피처
df_feat["hour"] = df_feat[TIME_COL].dt.hour
df_feat["weekday"] = df_feat[TIME_COL].dt.weekday
df_feat["month"] = df_feat[TIME_COL].dt.month
df_feat["is_weekend"] = (df_feat["weekday"] >= 5).astype(int)
#원천 센서 피처(그대로 사용)
SENSOR_FEATURES = [
    "operation",
    "activePower_mean",
    "activePower_max",
    "reactivePowerLagging_mean",
    "reactivePowerLagging_max",
    "currentR_mean",
    "currentR_max",
    "voltageR_mean",
    "voltageR_min",
    "voltageR_max",
    "powerFactorR_mean",
]

# 선택적 피처 (있을 때만 사용)
OPTIONAL_SENSOR_FEATURES = [
    "accumActiveEnergy_last"
]

# 실제 존재하는 컬럼만 사용
available_optional = [c for c in OPTIONAL_SENSOR_FEATURES if c in df.columns]

SENSOR_FEATURES = SENSOR_FEATURES + available_optional

# sanity check
missing = set(SENSOR_FEATURES) - set(df.columns)
assert len(missing) == 0, f"센서 피처 누락: {missing}"

print("사용되는 센서 피처:")
print(SENSOR_FEATURES)

#공장 전체 동반 피처(total_usage)
total_usage = (
    df_feat.groupby(TIME_COL)[TARGET]
    .sum()
    .rename("total_usage")
    .reset_index()
)

df_feat = df_feat.merge(total_usage, on=TIME_COL, how="left")

# 시간 기준 정렬 (이 줄이 핵심 보강 포인트)
df_feat = df_feat.sort_values(TIME_COL).reset_index(drop=True)

#lag 생성
df_feat["total_usage_lag1"] = df_feat["total_usage"].shift(1)
df_feat["total_usage_lag3"] = df_feat["total_usage"].shift(3)

#타깃 lag 피처 (equipment별)
for l in [1,2,3,6,12,24]:
    df_feat[f"y_lag{l}"] = (
        df_feat.groupby("equipment_id")[TARGET]
        .shift(l)
    )

# rolling 피처 (반드시 lag1 기반으로)
for w in [3,6,24]:
    df_feat[f"y_roll{w}_mean"] = (
        df_feat.groupby("equipment_id")[TARGET]
        .shift(1)
        .rolling(w)
        .mean()
    )
    df_feat[f"y_roll{w}_std"] = (
        df_feat.groupby("equipment_id")[TARGET]
        .shift(1)
        .rolling(w)
        .std()
    )
    df_feat[f"y_roll{w}_max"] = (
        df_feat.groupby("equipment_id")[TARGET]
        .shift(1)
        .rolling(w)
        .max()
    )


# 전조 변화량(Δ) 피처
df_feat["d_reactive"] = (
    df_feat.groupby("equipment_id")["reactivePowerLagging_mean"]
    .shift(1) -
    df_feat.groupby("equipment_id")["reactivePowerLagging_mean"]
    .shift(2)
)

df_feat["d_pf"] = (
    df_feat.groupby("equipment_id")["powerFactorR_mean"]
    .shift(1) -
    df_feat.groupby("equipment_id")["powerFactorR_mean"]
    .shift(2)
)

df_feat["d_vmin"] = (
    df_feat.groupby("equipment_id")["voltageR_min"]
    .shift(1) -
    df_feat.groupby("equipment_id")["voltageR_min"]
    .shift(2)
)

# 결측 처리(피처 생성 후 필수)
FEATURE_COLS = [
    "equipment_id",
    "hour","weekday","month","is_weekend",
    "operation",
    "activePower_mean","activePower_max",
    "reactivePowerLagging_mean","reactivePowerLagging_max",
    "currentR_mean","currentR_max",
    "voltageR_mean","voltageR_min","voltageR_max",
    "powerFactorR_mean","accumActiveEnergy_last",
    "total_usage","total_usage_lag1","total_usage_lag3",
    "y_lag1","y_lag2","y_lag3","y_lag6","y_lag12","y_lag24",
    "y_roll3_mean","y_roll3_std","y_roll3_max",
    "y_roll6_mean","y_roll6_std","y_roll6_max",
    "y_roll24_mean","y_roll24_std","y_roll24_max",
    "d_reactive","d_pf","d_vmin"
]

df_feat_clean = df_feat.dropna(subset=FEATURE_COLS + [TARGET]).copy()

train_df_clean = df_feat_clean[
    (df_feat_clean[TIME_COL] >= TRAIN_START) &
    (df_feat_clean[TIME_COL] <= TRAIN_END)
].copy()

valid_df_clean = df_feat_clean[
    (df_feat_clean[TIME_COL] >= BUFFER_START) &
    (df_feat_clean[TIME_COL] <= VALID_END)
].copy()

valid_eval = valid_df_clean[
    (valid_df_clean[TIME_COL] >= VALID_START) &
    (valid_df_clean[TIME_COL] <= VALID_END)
].copy()

#Naive baseline
# valid(eval) 구간만 분리

# valid(eval) 구간만 사용 (버퍼 제외)
valid_eval = valid_df_clean[
    (valid_df_clean[TIME_COL] >= VALID_START) &
    (valid_df_clean[TIME_COL] <= VALID_END)
].copy()

# Naive 예측값 생성

valid_eval["pred_naive"] = valid_eval["y_lag1"]

# 전체 성능 지표 계산 (MAE / RMSE / SMAPE)

y_true = valid_eval[TARGET]
y_pred = valid_eval["pred_naive"]

naive_metrics = calc_metrics(y_true, y_pred)

# 전체 성능 지표 계산 (MAE / RMSE / SMAPE)

y_true = valid_eval[TARGET]
y_pred = valid_eval["pred_naive"]

naive_metrics = calc_metrics(y_true, y_pred)

# Tail 기준값 계산 (train 기준)

q95 = train_df_clean[TARGET].quantile(0.95)
q99 = train_df_clean[TARGET].quantile(0.99)

# Tail 성능 계산 (q95~q99, q99+)

naive_tail_metrics = calc_tail_metrics(
    y_true=y_true,
    y_pred=y_pred,
    q95=q95,
    q99=q99
)

# 결과 정리

naive_result = {
    **naive_metrics,
    **naive_tail_metrics
}

naive_result_df = pd.DataFrame([naive_result])
print(naive_result_df)

#Ridge baseline
# Ridge에 사용할 피처 정의

RIDGE_FEATURES = [
    # 시간 피처
    "hour", "weekday", "month", "is_weekend",

    # 센서 피처
    "operation",
    "activePower_mean", "activePower_max",
    "reactivePowerLagging_mean", "reactivePowerLagging_max",
    "currentR_mean", "currentR_max",
    "voltageR_mean", "voltageR_min", "voltageR_max",
    "powerFactorR_mean",

    # target lag
    "y_lag1", "y_lag2", "y_lag3", "y_lag6",

    # rolling (짧은 것만)
    "y_roll3_mean", "y_roll6_mean",

    # 전조 변화량
    "d_reactive", "d_pf", "d_vmin",

    # 공장 동반
    "total_usage_lag1", "total_usage_lag3"
]

# Train / Valid(eval) 분리 (clean 기준)

X_train = train_df_clean[RIDGE_FEATURES]
y_train = train_df_clean[TARGET]

X_valid = valid_eval[RIDGE_FEATURES]
y_valid = valid_eval[TARGET]

# 표준화 + Ridge 학습

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

ridge = Ridge(alpha=1.0, random_state=SEED)
ridge.fit(X_train_scaled, y_train)

# Valid 예측

valid_eval["pred_ridge"] = ridge.predict(X_valid_scaled)
# 전체 성능 지표 (MAE / RMSE / SMAPE)

ridge_metrics = calc_metrics(
    y_true=y_valid,
    y_pred=valid_eval["pred_ridge"]
)
# Tail 기준값 계산 (train 기준)

q95 = y_train.quantile(0.95)
q99 = y_train.quantile(0.99)
# Tail 성능 계산 (q95~q99, q99+)

ridge_tail_metrics = calc_tail_metrics(
    y_true=y_valid,
    y_pred=valid_eval["pred_ridge"],
    q95=q95,
    q99=q99
)
# 결과 정리 (산출물)

ridge_result = {
    **ridge_metrics,
    **ridge_tail_metrics
}

ridge_result_df = pd.DataFrame([ridge_result])
print(ridge_result_df)
#baseline 성능표(naive/ridge)
naive_result_df["model"] = "Naive (y_lag1)"
ridge_result_df["model"] = "Ridge"

baseline_perf_table = (
    pd.concat([naive_result_df, ridge_result_df], axis=0)
      .set_index("model")
      .round(3)
)

baseline_perf_table

#메인 모델 CatBoost 학습
# CatBoost 입력 피처 정의
# Ridge보다 조금 더 풍부하게 사용 (CatBoost는 비선형 + 범주 처리 가능)

CAT_FEATURES = [
    # 시간 피처
    "hour", "weekday", "month", "is_weekend",

    # 센서 피처
    "operation",
    "activePower_mean", "activePower_max",
    "reactivePowerLagging_mean", "reactivePowerLagging_max",
    "currentR_mean", "currentR_max",
    "voltageR_mean", "voltageR_min", "voltageR_max",
    "powerFactorR_mean",

    # 공장 동반
    "total_usage", "total_usage_lag1", "total_usage_lag3",

    # 타깃 lag
    "y_lag1", "y_lag2", "y_lag3", "y_lag6", "y_lag12", "y_lag24",

    # rolling
    "y_roll3_mean", "y_roll6_mean", "y_roll24_mean",

    # 전조 변화량
    "d_reactive", "d_pf", "d_vmin"
]

# Train / Valid 데이터 분리

X_train = train_df_clean[CAT_FEATURES]
y_train = train_df_clean[TARGET]

X_valid = valid_eval[CAT_FEATURES]
y_valid = valid_eval[TARGET]

# 범주형 컬럼 지정

cat_cols = ["equipment_id", "weekday"]

cat_features_idx = [
    X_train.columns.get_loc(col)
    for col in cat_cols
    if col in X_train.columns
]

# CatBoostRegressor 학습

from catboost import CatBoostRegressor

cb_model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.05,
    depth=7,
    loss_function="MAE",
    random_seed=SEED,
    eval_metric="MAE",
    early_stopping_rounds=100,
    verbose=200
)

cb_model.fit(
    X_train, y_train,
    eval_set=(X_valid, y_valid),
    cat_features=cat_features_idx,
    use_best_model=True
)

```python
# Valid 예측

valid_eval["pred_cb"] = cb_model.predict(X_valid)

# 전체 성능 지표 계산 (MAE / RMSE / SMAPE)

cb_metrics = calc_metrics(
    y_true=y_valid,
    y_pred=valid_eval["pred_cb"]
)

# Tail 성능 지표 계산 (train 기준 q95/q99)

q95 = train_df_clean[TARGET].quantile(0.95)
q99 = train_df_clean[TARGET].quantile(0.99)

cb_tail_metrics = calc_tail_metrics(
    y_true=y_valid,
    y_pred=valid_eval["pred_cb"],
    q95=q95,
    q99=q99
)
# CatBoost 원본 예측값 (반드시 먼저 정의)
pred_cb_raw = cb_model.predict(X_valid)
```

# CatBoost 성능표 (산출물)

cb_result = {
    **cb_metrics,
    **cb_tail_metrics
}

cb_result_df = pd.DataFrame([cb_result])
cb_result_df["model"] = "CatBoost"

cb_result_df = cb_result_df.set_index("model").round(3)
cb_result_df

# Feature Importance 상위 20 (산출물)

fi = pd.DataFrame({
    "feature": CAT_FEATURES,
    "importance": cb_model.get_feature_importance()
})

fi_top20 = (
    fi.sort_values("importance", ascending=False)
      .head(20)
      .reset_index(drop=True)
)

fi_top20

#예측값 안정화(0점 리스크 차단 장치)
# train 기준 q99 계산 (이미 있으면 재사용)

q99_train = train_df_clean[TARGET].quantile(0.99)
q99_train

# valid 예측값 준비 (CatBoost)

# valid(eval) 구간 CatBoost 예측
valid_eval["pred_cb"] = cb_model.predict(
    valid_eval[CAT_FEATURES]
)

# α 후보별 클리핑 + 성능 평가

ALPHAS = [1.5, 2.0, 2.5, 3.0]

clip_results = []

y_true = valid_eval[TARGET]

for alpha in ALPHAS:
    upper = q99_train * alpha

    # 클리핑 적용
    y_pred_clip = valid_eval["pred_cb"].clip(lower=0, upper=upper)

    # 전체 성능
    metrics = calc_metrics(y_true, y_pred_clip)

    # tail 성능
    tail_metrics = calc_tail_metrics(
        y_true=y_true,
        y_pred=y_pred_clip,
        q95=q95,
        q99=q99
    )

    clip_results.append({
        "alpha": alpha,
        **metrics,
        **tail_metrics
    })

# α 후보별 성능 비교표 (산출물)

clip_result_df = (
    pd.DataFrame(clip_results)
      .set_index("alpha")
      .round(3)
)

clip_result_df

# 최적 α 선택 (기준 명확히)

best_alpha = clip_result_df["q99p_MAE"].idxmin()
best_alpha

# 최종 클리핑 예측값 확정

BEST_UPPER = q99_train * best_alpha

valid_eval["pred_cb_clip"] = (
    valid_eval["pred_cb"]
    .clip(lower=0, upper=BEST_UPPER)
)

#operation=0 보정(필요할 때만)
# 클리핑 전
pred_raw = valid_eval["pred_cb"]

raw_metrics = {
    **calc_metrics(valid_eval[TARGET], pred_raw),
    **calc_tail_metrics(
        valid_eval[TARGET],
        pred_raw,
        q95=train_df_clean[TARGET].quantile(0.95),
        q99=q99_train
    )
}

# 클리핑 후 (선택된 alpha = 1.5)
best_alpha = 1.5
upper = q99_train * best_alpha

pred_clipped = pred_raw.clip(lower=0, upper=upper)

clip_metrics = {
    **calc_metrics(valid_eval[TARGET], pred_clipped),
    **calc_tail_metrics(
        valid_eval[TARGET],
        pred_clipped,
        q95=train_df_clean[TARGET].quantile(0.95),
        q99=q99_train
    )
}

clip_compare_df = (
    pd.DataFrame([raw_metrics, clip_metrics],
                 index=["before_clip", "after_clip"])
      .round(3)
)

clip_compare_df

valid_eval = valid_eval.copy()
valid_eval["pred_cb_raw"]  = pred_cb_raw
valid_eval["pred_cb_clip"] = pred_cb_clip

# Tail 성능 평가(꼬리구간에서 무너지는지 확인)

# train 기준 q95, q99 계산

q95_train = train_df_clean[TARGET].quantile(0.95)
q99_train = train_df_clean[TARGET].quantile(0.99)

# valid에서 q95, q99 계산(“train 기준”으로)

def evaluate_all_and_tail(y_true, y_pred, q95, q99):
    result = {}
    result.update(calc_metrics(y_true, y_pred))
    result.update(calc_tail_metrics(y_true, y_pred, q95, q99))
    return result
    
# valid 데이터 준비

y_true = valid_eval[TARGET]

# 클리핑 전 성능
before_result = evaluate_all_and_tail(
    y_true=y_true,
    y_pred=pred_cb_raw,
    q95=q95_train,
    q99=q99_train
)

#클리핑 후 성능
after_result = evaluate_all_and_tail(
    y_true=y_true,
    y_pred=pred_cb_clip,
    q95=q95_train,
    q99=q99_train
)

# Tail 성능표

tail_result_df = (
    pd.DataFrame([before_result, after_result],
                 index=["before_clip", "after_clip"])
      .round(3)
)

tail_result_df

#Top5 설비 성능 평가
TOP5 = [2, 16, 17, 11, 13]

rows = []

for eid in TOP5:
    df_e = valid_eval[valid_eval["equipment_id"] == eid]

    if len(df_e) == 0:
        continue

    y_true_e = df_e[TARGET]
    y_pred_e = df_e["pred_cb_clip"].values

    metrics = evaluate_all_and_tail(
        y_true=y_true_e,
        y_pred=y_pred_e,
        q95=q95_train,
        q99=q99_train
    )

    metrics["equipment_id"] = eid
    rows.append(metrics)

top5_result_df = (
    pd.DataFrame(rows)
      .set_index("equipment_id")
      .round(3)
)

top5_result_df

#Test 예측 + submission.csv 생성
train_valid_df = (
    pd.concat([train_df_clean, valid_eval], axis=0)
      .sort_values(["equipment_id", TIME_COL])
      .reset_index(drop=True)
)

X_train_final = train_valid_df[FEATURE_COLS]
y_train_final = train_valid_df[TARGET]
#CatBoost 최종 재학습
from catboost import CatBoostRegressor

cat_features = [
    X_train_final.columns.get_loc("equipment_id"),
    X_train_final.columns.get_loc("weekday")
]

final_cb = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.05,
    depth=6,
    loss_function="MAE",
    random_seed=42,
    verbose=200
)

final_cb.fit(
    X_train_final,
    y_train_final,
    cat_features=cat_features
)
#Test 데이터 로딩 + 컬럼 정합
test_df = pd.read_csv("test.csv", encoding="cp949")

test_df = test_df.rename(columns={
    "module(equipment)": "equipment_id",
    "datetime": "hour_dt",
    "activePower": "activePower_mean",
    "reactivePowerLagging": "reactivePowerLagging_mean",
    "accumActiveEnergy": "accumActiveEnergy_last"
})

test_df["hour_dt"] = pd.to_datetime(test_df["hour_dt"])

test_df = (
    test_df
    .sort_values(["equipment_id", "hour_dt"])
    .reset_index(drop=True)
)

#Train+Valid+Test 붙여서 피처 생성
full_df = (
    pd.concat([train_valid_df, test_df], axis=0)
      .sort_values(["equipment_id", "hour_dt"])
      .reset_index(drop=True)
)

df_feat = full_df.copy()

# 시간 피처
df_feat["hour"] = df_feat["hour_dt"].dt.hour
df_feat["weekday"] = df_feat["hour_dt"].dt.weekday
df_feat["month"] = df_feat["hour_dt"].dt.month
df_feat["is_weekend"] = (df_feat["weekday"] >= 5).astype(int)

# 공장 전체 동반
df_feat["total_usage"] = df_feat.groupby("hour_dt")[TARGET].transform("sum")
df_feat["total_usage_lag1"] = df_feat["total_usage"].shift(1)
df_feat["total_usage_lag3"] = df_feat["total_usage"].shift(3)

# target lag
for l in [1,2,3,6,12,24]:
    df_feat[f"y_lag{l}"] = (
        df_feat.groupby("equipment_id")[TARGET].shift(l)
    )

# rolling (lag1 기반)
for w in [3,6,24]:
    base = df_feat.groupby("equipment_id")[TARGET].shift(1)
    df_feat[f"y_roll{w}_mean"] = base.rolling(w).mean()
    df_feat[f"y_roll{w}_std"]  = base.rolling(w).std()
    df_feat[f"y_roll{w}_max"]  = base.rolling(w).max()

# Δ 피처
df_feat["d_reactive"] = (
    df_feat.groupby("equipment_id")["reactivePowerLagging_mean"].shift(1)
    - df_feat.groupby("equipment_id")["reactivePowerLagging_mean"].shift(2)
)
df_feat["d_pf"] = (
    df_feat.groupby("equipment_id")["powerFactorR_mean"].shift(1)
    - df_feat.groupby("equipment_id")["powerFactorR_mean"].shift(2)
)
df_feat["d_vmin"] = (
    df_feat.groupby("equipment_id")["voltageR_min"].shift(1)
    - df_feat.groupby("equipment_id")["voltageR_min"].shift(2)
)

#예측 + 클리핑
X_test = test_feat_clean[FEATURE_COLS]

pred_raw = final_cb.predict(X_test)

q99_train = train_df_clean[TARGET].quantile(0.99)

pred_test = np.clip(pred_raw, 0, q99_train * 2.0)

#submission.csv 생성 + 점검
submission = test_feat_clean[["equipment_id", "hour_dt"]].copy()
submission["hourly_usage"] = pred_test

submission = submission.sort_values(
    ["equipment_id", "hour_dt"]
).reset_index(drop=True)

submission.to_csv("submission.csv", index=False)

# 점검 로그
print("행 수:", len(submission))
print("NaN:", submission.isna().sum().sum())
print("min / max:", submission["hourly_usage"].min(),
                  submission["hourly_usage"].max())
print("상위 1%:", submission["hourly_usage"].quantile(0.99))

#result.csv 생성
import pandas as pd
import numpy as np

# ===============================
# 0. 상수 설정
# ===============================
COST_PER_KWH = 180
CARBON_PER_KWH = 0.424

# ===============================
# 1. 4월 공장 전체 시간당 전력 계산
# ===============================
# train_valid_df에 hourly_usage, hour_dt가 있다고 가정

train_valid_df["hour_dt"] = pd.to_datetime(train_valid_df["hour_dt"])

# 4월 데이터만 사용
april_df = train_valid_df[
    (train_valid_df["hour_dt"] >= "2025-04-01") &
    (train_valid_df["hour_dt"] <  "2025-05-01")
]

# 공장 전체 시간당 전력
april_hourly = (
    april_df
    .groupby("hour_dt")["hourly_usage"]
    .mean()
    .reset_index()
)

# 시간 파생
april_hourly["hour"] = april_hourly["hour_dt"].dt.hour
april_hourly["weekday"] = april_hourly["hour_dt"].dt.weekday
april_hourly["is_weekend"] = (april_hourly["weekday"] >= 5).astype(int)

# ===============================
# 2. (요일 × 시간대) 평균 패턴
# ===============================
pattern = (
    april_hourly
    .groupby(["is_weekend", "hour"])["hourly_usage"]
    .mean()
    .reset_index()
)

# 전체 평균 (fallback 용)
global_mean = april_hourly["hourly_usage"].mean()

print("✅ 4월 hourly 평균:", global_mean)

# ===============================
# 3. 5월 시간 그리드 (744시간)
# ===============================
may_hours = pd.date_range(
    start="2025-05-01 00:00:00",
    end="2025-05-31 23:00:00",
    freq="h"
)

may_df = pd.DataFrame({"id": may_hours})
may_df["hour"] = may_df["id"].dt.hour
may_df["weekday"] = may_df["id"].dt.weekday
may_df["is_weekend"] = (may_df["weekday"] >= 5).astype(int)

# ===============================
# 4. 패턴 매핑 → 5월 hourly_pow
# ===============================
may_df = may_df.merge(
    pattern,
    on=["is_weekend", "hour"],
    how="left"
)

# 패턴 없는 경우 4월 평균으로 대체
may_df["hourly_pow"] = may_df["hourly_usage"].fillna(global_mean)
may_df.drop(columns=["hourly_usage"], inplace=True)

# ===============================
# 5. 월 누적값 계산 (정상 방식)
# ===============================
agg_pow = may_df["hourly_pow"].sum()
may_bill = agg_pow * COST_PER_KWH
may_carbon = agg_pow * CARBON_PER_KWH

# 모든 행에 반복 입력
may_df["agg_pow"] = agg_pow
may_df["may_bill"] = may_bill
may_df["may_carbon"] = may_carbon

# ===============================
# 6. 제출 포맷 맞추기
# ===============================
result = may_df[[
    "id",
    "hourly_pow",
    "agg_pow",
    "may_bill",
    "may_carbon"
]]

# ===============================
# 7. 저장 및 검증 출력
# ===============================
result.to_csv("result.csv", index=False)

print("\n✅ result.csv 생성 완료")
print("행 수:", len(result))
print("hourly 평균:", result['hourly_pow'].mean())
print("agg_pow:", agg_pow)
print("may_bill:", may_bill)
print("may_carbon:", may_carbon)

print("4월 hourly 평균:", train_valid_df[
    train_valid_df["hour_dt"].dt.month == 4
]["hourly_usage"].mean())

print("5월 hourly 평균:", result["hourly_pow"].mean())

print("비율:", result["hourly_pow"].mean() /
              train_valid_df[
                  train_valid_df["hour_dt"].dt.month == 4
              ]["hourly_usage"].mean())

