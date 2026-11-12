"""
7-day price forecast: one ridge regression per horizon (1…H days ahead), trained on
daily data. Inputs that are known when the forecast is made: recent prices, reservoir
deviation, gas price; inputs for the target day: weather forecast, consumption forecast,
calendar. Hourly shape is added from the last 4 weeks' average daily profile.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_LABELS = {
    "p_last": "Last known price", "p7": "7-day average price", "p28": "28-day average price",
    "temp": "Temperature", "wind": "Wind speed", "precip": "Precipitation (that day)",
    "precip14": "Rain last 14 days", "reservoir": "Reservoir vs normal", "load": "Consumption forecast",
    "weekend": "Weekend", "sunday": "Sunday", "season_cos": "Season (winter ↔ summer)", "season_sin": "Season (spring ↔ autumn)",
}


class Ridge:
    def __init__(self, lam: float):
        self.lam = lam

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd == 0] = 1
        Z = (X - self.mu) / self.sd
        self.ym = y.mean()
        A = Z.T @ Z + self.lam * np.eye(Z.shape[1])
        self.beta = np.linalg.solve(A, Z.T @ (y - self.ym))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.ym + ((X - self.mu) / self.sd) @ self.beta


def build_features(d: pd.DataFrame, h: int) -> pd.DataFrame:
    """d: daily frame indexed by date with columns price, temp, wind, precip, reservoir, load (some may be NaN).
    Row t uses information available on day t-h for prices/reservoir, and day-t values for forecastable inputs."""
    p = d["price"]
    f = pd.DataFrame(index=d.index)
    f["p_last"] = p.shift(h)
    f["p7"] = p.rolling(7, min_periods=5).mean().shift(h)
    f["p28"] = p.rolling(28, min_periods=20).mean().shift(h)
    for c in ["temp", "wind", "precip", "load"]:
        if c in d:
            f[c] = d[c]
    if "precip" in d:
        f["precip14"] = d["precip"].rolling(14, min_periods=10).sum().shift(h)
    if "reservoir" in d:
        f["reservoir"] = d["reservoir"].shift(h)
    wd = f.index.dayofweek
    f["weekend"] = (wd >= 5).astype(float)
    f["sunday"] = (wd == 6).astype(float)
    doy = f.index.dayofyear
    f["season_cos"] = np.cos(2 * np.pi * doy / 365.25)
    f["season_sin"] = np.sin(2 * np.pi * doy / 365.25)
    return f


def forecast(daily_df: pd.DataFrame, hourly_price: pd.Series, horizon: int = 7, min_coverage: float = .75):
    """Returns dict with forecast table, hourly forecast, accuracy table, importance, backtest, used/dropped features."""
    d = daily_df.copy()
    last_known = d["price"].last_valid_index()
    future = pd.date_range(last_known + pd.Timedelta(days=1), periods=horizon, freq="D")
    d = d.reindex(d.index.union(future))
    # carry slow inputs forward over the forecast days
    for c in ["reservoir"]:
        if c in d:
            d[c] = d[c].ffill()

    hist_idx = d.index[(d.index <= last_known)]
    candidates = list(build_features(d, 1).columns)
    f1 = build_features(d, 1).loc[hist_idx]
    usable = [c for c in candidates if c in ("p_last", "p7") or f1[c].notna().mean() >= min_coverage]
    # forecastable inputs must exist for (most of) the future days
    fut_feat = build_features(d, 1).loc[future]
    usable = [c for c in usable if c not in ("temp", "wind", "precip", "load") or fut_feat[c].notna().sum() >= horizon - 1]
    dropped = [c for c in candidates if c not in usable]

    rows, acc, importance, backtest = [], [], {}, None
    for h in range(1, horizon + 1):
        F = build_features(d, h)[usable]
        train = F.loc[hist_idx].join(d["price"]).dropna()
        if len(train) < 60:
            raise ValueError(f"Only {len(train)} complete days to train on — choose a longer training period")
        X, y = train[usable].values, train["price"].values
        cut = int(len(train) * .8)
        best = None
        for lam in (0.3, 3, 30, 100):
            m = Ridge(lam).fit(X[:cut], y[:cut])
            err = y[cut:] - m.predict(X[cut:])
            mae = np.abs(err).mean()
            if best is None or mae < best[1]:
                best = (lam, mae, err, m)
        lam, mae, err, mtest = best
        naive = np.abs(y[cut:] - train["p_last"].values[cut:]).mean()
        sigma = float(np.sqrt((err ** 2).mean()))
        acc.append({"h": h, "mae": mae, "naive": naive, "sigma": sigma, "n_train": len(train)})
        if h in (1, 3, 7):
            importance[h] = pd.Series(mtest.beta, index=usable)
        if h == 3:
            backtest = pd.DataFrame({"actual": y[cut:], "model": mtest.predict(X[cut:])}, index=train.index[cut:])
        model = Ridge(lam).fit(X, y)
        t = future[h - 1]
        x = F.loc[[t]].copy()
        fill = train[usable].iloc[-7:].mean()  # typical recent value if an input is missing
        x = x.fillna(fill)
        v = float(model.predict(x.values)[0])
        rows.append({"date": t, "forecast": v, "low": max(v - 1.2816 * sigma, -500), "high": v + 1.2816 * sigma,
                     **{c: d.at[t, c] if c in d else np.nan for c in ["temp", "wind", "precip", "load", "reservoir"]}})
    fc = pd.DataFrame(rows).set_index("date")

    # hourly shape: average deviation from the daily mean per hour, last 28 days, weekday vs weekend
    hp = hourly_price[hourly_price.index >= hourly_price.index.max() - pd.Timedelta(days=28)].copy()
    hp_df = pd.DataFrame({"p": hp.values}, index=hp.index.tz_localize(None))
    hp_df["day"] = hp_df.index.normalize()
    hp_df["dev"] = hp_df["p"] - hp_df.groupby("day")["p"].transform("mean")
    hp_df["hour"], hp_df["we"] = hp_df.index.hour, hp_df.index.dayofweek >= 5
    shape = hp_df.groupby(["we", "hour"])["dev"].mean()
    hourly = []
    for t, r in fc.iterrows():
        for hr in range(24):
            hourly.append({"time": t + pd.Timedelta(hours=hr), "price": r.forecast + shape.get((t.dayofweek >= 5, hr), 0.0)})
    return {"table": fc, "hourly": pd.DataFrame(hourly).set_index("time"), "accuracy": pd.DataFrame(acc).set_index("h"),
            "importance": pd.DataFrame(importance), "backtest": backtest, "used": usable, "dropped": dropped,
            "last_known": last_known}
