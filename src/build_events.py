from pathlib import Path
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
raw_path = project_root / "data" / "raw" / "spy_2m.csv"
processed_dir = project_root / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(raw_path)

df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
df = df[df["Datetime"].notna()].copy()

numeric_cols = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.sort_values("Datetime").reset_index(drop=True)

k = 3
m = 30
z = 2
cooldown = 10

h0 = 10
hmax = 200
alpha = 2
beta = 5
min_init_move = 0.00015

df["ret_1"] = df["Close"].pct_change()
df["ret_k"] = df["Close"].pct_change(k)
df["sigma_k"] = df["ret_k"].rolling(m).std()
df["event_zscore"] = df["ret_k"].abs() / df["sigma_k"]

df["vol_k"] = df["Volume"].rolling(k).sum()
df["vol_k_rel"] = df["vol_k"] / df["vol_k"].rolling(m).mean()
df["range_k"] = df["High"].rolling(k).max() / df["Low"].rolling(k).min() - 1
df["vol_regime"] = df["ret_1"].rolling(m).std()
df["prev_ret_k"] = df["ret_k"].shift(k)

df["ret_2"] = df["Close"].pct_change(2)
df["ret_3"] = df["Close"].pct_change(3)

df["ret_1_lag1"] = df["ret_1"].shift(1)
df["ret_1_lag2"] = df["ret_1"].shift(2)
df["ret_1_lag3"] = df["ret_1"].shift(3)

df["ret_1_mean_3"] = df["ret_1"].rolling(3).mean()
df["ret_1_std_3"] = df["ret_1"].rolling(3).std()

df["up_bar"] = (df["ret_1"] > 0).astype(int)
df["down_bar"] = (df["ret_1"] < 0).astype(int)
df["up_frac_3"] = df["up_bar"].rolling(3).mean()
df["down_frac_3"] = df["down_bar"].rolling(3).mean()

df["max_ret_1_3"] = df["ret_1"].rolling(3).max()
df["min_ret_1_3"] = df["ret_1"].rolling(3).min()

df["body"] = (df["Close"] - df["Open"]) / df["Open"]
df["bar_range"] = df["High"] / df["Low"] - 1

df["upper_wick"] = (df["High"] - np.maximum(df["Open"], df["Close"])) / df["Open"]
df["lower_wick"] = (np.minimum(df["Open"], df["Close"]) - df["Low"]) / df["Open"]

df["vol_1_rel"] = df["Volume"] / df["Volume"].rolling(m).mean()
df["vol_mean_3"] = df["Volume"].rolling(3).mean()
df["vol_mean_3_rel"] = df["vol_mean_3"] / df["Volume"].rolling(m).mean()

minutes = df["Datetime"].dt.hour * 60 + df["Datetime"].dt.minute
df["mins_from_open"] = minutes - (9 * 60 + 30)

raw_event_mask = df["event_zscore"] >= z

selected = np.zeros(len(df), dtype=bool)
last_event_idx = -10**9

for i in range(len(df)):
    if raw_event_mask.iloc[i] and i - last_event_idx > cooldown:
        selected[i] = True
        last_event_idx = i

event_idx = np.flatnonzero(selected)

close = df["Close"].to_numpy()
rows = []

for i in event_idx:
    if i + hmax >= len(df):
        continue

    event_move = df.loc[i, "ret_k"]
    event_size = abs(event_move)
    event_sign = int(np.sign(event_move))

    if event_sign == 0:
        continue

    event_price = close[i]
    target_move = alpha * event_size

    hit_before_h0 = False
    for j in range(1, h0 + 1):
        move_abs = abs(close[i + j] / event_price - 1)
        if move_abs >= target_move:
            hit_before_h0 = True
            break

    if hit_before_h0:
        continue

    init_post_move = close[i + h0] / event_price - 1

    if abs(init_post_move) < min_init_move:
        continue

    post_sign = int(np.sign(init_post_move))

    running_best = 0.0
    max_drawdown_frac = 0.0
    resolved = False
    hit_bar = np.nan
    hit_move = np.nan

    for j in range(h0 + 1, hmax + 1):
        move_from_event = post_sign * (close[i + j] / event_price - 1)

        if move_from_event > running_best:
            running_best = move_from_event

        if running_best > 0:
            drawdown_frac = (running_best - move_from_event) / running_best
        else:
            drawdown_frac = 0.0

        if drawdown_frac > max_drawdown_frac:
            max_drawdown_frac = drawdown_frac

        if drawdown_frac > beta:
            break

        if move_from_event >= target_move:
            resolved = True
            hit_bar = j
            hit_move = move_from_event
            break

    if not resolved:
        continue

    label_continuation = int(post_sign == event_sign)

    rows.append({
        "Datetime": df.loc[i, "Datetime"],
        "Open": df.loc[i, "Open"],
        "High": df.loc[i, "High"],
        "Low": df.loc[i, "Low"],
        "Close": df.loc[i, "Close"],
        "Volume": df.loc[i, "Volume"],
        "ret_k": df.loc[i, "ret_k"],
        "sigma_k": df.loc[i, "sigma_k"],
        "event_zscore": df.loc[i, "event_zscore"],
        "event_sign": event_sign,
        "event_size": event_size,
        "vol_k_rel": df.loc[i, "vol_k_rel"],
        "range_k": df.loc[i, "range_k"],
        "vol_regime": df.loc[i, "vol_regime"],
        "prev_ret_k": df.loc[i, "prev_ret_k"],
        "mins_from_open": df.loc[i, "mins_from_open"],
        "init_post_move": init_post_move,
        "post_sign": post_sign,
        "target_move": target_move,
        "hit_bar": hit_bar,
        "hit_move": hit_move,
        "max_drawdown_frac": max_drawdown_frac,
        "label_continuation": label_continuation,
        "ret_2": df.loc[i, "ret_2"],
        "ret_3": df.loc[i, "ret_3"],
        "ret_1_lag1": df.loc[i, "ret_1_lag1"],
        "ret_1_lag2": df.loc[i, "ret_1_lag2"],
        "ret_1_lag3": df.loc[i, "ret_1_lag3"],
        "ret_1_mean_3": df.loc[i, "ret_1_mean_3"],
        "ret_1_std_3": df.loc[i, "ret_1_std_3"],
        "up_frac_3": df.loc[i, "up_frac_3"],
        "down_frac_3": df.loc[i, "down_frac_3"],
        "max_ret_1_3": df.loc[i, "max_ret_1_3"],
        "min_ret_1_3": df.loc[i, "min_ret_1_3"],
        "body": df.loc[i, "body"],
        "bar_range": df.loc[i, "bar_range"],
        "upper_wick": df.loc[i, "upper_wick"],
        "lower_wick": df.loc[i, "lower_wick"],
        "vol_1_rel": df.loc[i, "vol_1_rel"],
        "vol_mean_3_rel": df.loc[i, "vol_mean_3_rel"],
    })

events = pd.DataFrame(rows)
events = events.dropna().reset_index(drop=True)

events.to_csv(processed_dir / "spy_event_dataset.csv", index=False)

print(events.shape)
print(events["label_continuation"].value_counts())
print(events.head())

terminal_counts = {
    "hit_before_h0": 0,
    "no_established_bias": 0,
    "resolved": 0,
    "not_resolved_within_hmax": 0,
    "hit_max_drawdown": 0,
    "dropped_near_end": 0,
}

for i in event_idx:
    if i + hmax >= len(df):
        terminal_counts["dropped_near_end"] += 1
        continue

    event_move = df.loc[i, "ret_k"]
    event_size = abs(event_move)
    event_sign = int(np.sign(event_move))

    if event_sign == 0:
        continue

    event_price = close[i]
    target_move = alpha * event_size

    hit_early = False
    for j in range(1, h0 + 1):
        move_abs = abs(close[i + j] / event_price - 1)
        if move_abs >= target_move:
            terminal_counts["hit_before_h0"] += 1
            hit_early = True
            break

    if hit_early:
        continue

    init_post_move = close[i + h0] / event_price - 1

    if abs(init_post_move) < min_init_move:
        terminal_counts["no_established_bias"] += 1
        continue

    post_sign = int(np.sign(init_post_move))
    running_best = 0.0
    resolved = False
    stopped_by_drawdown = False

    for j in range(h0 + 1, hmax + 1):
        move_from_event = post_sign * (close[i + j] / event_price - 1)

        if move_from_event > running_best:
            running_best = move_from_event

        if running_best > 0:
            drawdown_frac = (running_best - move_from_event) / running_best
        else:
            drawdown_frac = 0.0

        if drawdown_frac > beta:
            terminal_counts["hit_max_drawdown"] += 1
            stopped_by_drawdown = True
            break

        if move_from_event >= target_move:
            terminal_counts["resolved"] += 1
            resolved = True
            break

    if (not resolved) and (not stopped_by_drawdown):
        terminal_counts["not_resolved_within_hmax"] += 1

print(terminal_counts)
print("recognised events:", len(event_idx))