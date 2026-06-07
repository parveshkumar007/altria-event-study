"""
Event Study: Media Articles vs. Regulatory Disclosures — Impact on Altria (MO) Stock Price
Author  : Parvesh Kumar & Dean Reid
Course  : ECON 435 — Financial Economics and Quantitative Methods
Method  : Market model (OLS). Abnormal returns estimated over an estimation
          window of [-153, -6] days; CAR measured over [-5, +15] event window.
          Matches methodology from original research paper.

Usage   : python3 altria_event_study_v2.py
Requires: pip3 install pandas numpy yfinance matplotlib scipy
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ── 1. CONFIGURATION ──────────────────────────────────────────────────────────

STOCK     = "MO"     # Altria Group (formerly Philip Morris)
BENCHMARK = "^GSPC"  # S&P 500

# Windows (in trading days relative to event date, t=0)
EVENT_WINDOW      = (-5, 15)    # Matches 1964 analysis in paper
ESTIMATION_WINDOW = (-153, -6)  # Matches regression window in paper

# ── 2. EVENT DATES ─────────────────────────────────────────────────────────────
# These are the three events from the ECON 435 research paper.
# 1964: Monday Jan 13 used because the report was released on a Saturday (Jan 11).

MEDIA_EVENTS = {
    "1964-01-13": "Surgeon General's Report on Smoking & Health",
}

REGULATORY_EVENTS = {
    "1993-01-07": "EPA designates passive smoking a known carcinogen",
    "2006-08-17": "Judge Kessler's ruling — Big Tobacco racketeering case",
}

# ── 3. DATA DOWNLOAD ───────────────────────────────────────────────────────────

def download_returns(ticker, start="1963-01-01", end="2007-12-31"):
    """Download adjusted close prices and return daily simple returns."""
    print(f"  Downloading {ticker} ...")
    prices = yf.download(ticker, start=start, end=end,
                         auto_adjust=True, progress=False)["Close"]
    prices = prices.squeeze()
    if prices.empty:
        raise ValueError(f"No data returned for {ticker}. "
                         "Check ticker or date range.")
    returns = prices.pct_change().dropna()
    returns.name = ticker
    return returns

print("Downloading price data ...")
stock_ret = download_returns(STOCK)
bench_ret = download_returns(BENCHMARK)

data = pd.concat([stock_ret, bench_ret], axis=1).dropna()
data.columns = ["MO", "SPY"]
print(f"  Data range : {data.index[0].date()} → {data.index[-1].date()} "
      f"({len(data)} trading days)\n")

# ── 4. EVENT STUDY ENGINE ──────────────────────────────────────────────────────

def run_event_study(event_dates, data,
                    est_window=(-153, -6),
                    evt_window=(-5, 15)):
    """
    For each event:
      1. Fit OLS market model over the estimation window:
            R_stock = α + β × R_market + ε
      2. Compute abnormal returns (AR) over event window:
            AR = R_actual - R_expected
      3. Cumulate into CAR and test significance (t-test).

    Returns per-event results and the average CAR across all events.
    """
    trading_days = data.index
    results, all_ars = [], []

    for date_str, label in event_dates.items():
        event_date = pd.Timestamp(date_str)
        idx = trading_days.searchsorted(event_date)

        # Use the nearest available trading day if exact date is a non-trading day
        if idx >= len(trading_days):
            print(f"  ⚠ {date_str}: falls outside data range, skipping.")
            continue
        t0 = idx

        est_start = t0 + est_window[0]
        est_end   = t0 + est_window[1]
        evt_start = t0 + evt_window[0]
        evt_end   = t0 + evt_window[1] + 1

        if est_start < 0:
            print(f"  ⚠ {date_str}: not enough history for estimation window, skipping.")
            continue
        if evt_end > len(trading_days):
            print(f"  ⚠ {date_str}: event window extends beyond data, skipping.")
            continue

        # ── Market model ────────────────────────────────────────────────────
        est_data = data.iloc[est_start:est_end]
        slope, intercept, *_ = stats.linregress(est_data["SPY"], est_data["MO"])

        # ── Abnormal returns ────────────────────────────────────────────────
        evt_data = data.iloc[evt_start:evt_end]
        expected = intercept + slope * evt_data["SPY"].values
        actual   = evt_data["MO"].values
        ar       = actual - expected
        car      = np.cumsum(ar)
        rel_days = list(range(evt_window[0], evt_window[1] + 1))

        t_stat, p_val = stats.ttest_1samp(ar, 0)

        actual_date = trading_days[t0].date()
        print(f"  ✓ {actual_date}  [{label}]")
        print(f"    Beta: {slope:.3f}  |  CAR total: {car[-1]*100:.2f}%  "
              f"|  t={t_stat:.2f}  p={p_val:.4f}  "
              f"{'✓ Significant' if p_val < 0.05 else '✗ Not significant'}\n")

        results.append({
            "date"        : actual_date,
            "label"       : label,
            "beta"        : round(slope, 3),
            "CAR_%"       : round(car[-1] * 100, 2),
            "t_stat"      : round(t_stat, 3),
            "p_value"     : round(p_val, 4),
            "significant" : p_val < 0.05,
        })
        all_ars.append(pd.Series(ar, index=rel_days,
                                 name=str(actual_date)))

    summary = pd.DataFrame(results)
    if all_ars:
        ar_matrix = pd.concat(all_ars, axis=1).T
        avg_car   = ar_matrix.mean().cumsum()
    else:
        avg_car = pd.Series(dtype=float)

    return {"summary": summary, "avg_car": avg_car}


print("=" * 60)
print("MEDIA EVENTS")
print("=" * 60)
media_res = run_event_study(MEDIA_EVENTS, data,
                             ESTIMATION_WINDOW, EVENT_WINDOW)

print("=" * 60)
print("REGULATORY / LEGAL EVENTS")
print("=" * 60)
reg_res = run_event_study(REGULATORY_EVENTS, data,
                           ESTIMATION_WINDOW, EVENT_WINDOW)

# ── 5. CHARTS ─────────────────────────────────────────────────────────────────

days = list(range(EVENT_WINDOW[0], EVENT_WINDOW[1] + 1))

fig = plt.figure(figsize=(16, 10))
fig.suptitle(
    "Altria (MO) — Cumulative Abnormal Returns Around Key Information Events\n"
    "ECON 435 — Parvesh Kumar & Dean Reid, UNBC",
    fontsize=13, fontweight="bold", y=0.98
)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── Panel A: Average CAR — Media Events ───────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
if not media_res["avg_car"].empty:
    car_pct = media_res["avg_car"] * 100
    ax1.plot(car_pct.index, car_pct.values, color="#2A5BA8",
             linewidth=2.5, marker="o", markersize=5)
    ax1.fill_between(car_pct.index, 0, car_pct.values, alpha=0.12, color="#2A5BA8")
ax1.axvline(x=0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
ax1.axhline(y=0, color="gray", linewidth=0.8, alpha=0.4)
ax1.set_title("Average CAR — Media Events", fontweight="bold", fontsize=11)
ax1.set_xlabel("Days Relative to Event (t=0)")
ax1.set_ylabel("CAR (%)")
ax1.set_xticks(days[::2])
ax1.grid(True, linestyle="--", alpha=0.4)

# ── Panel B: Average CAR — Regulatory Events ──────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
if not reg_res["avg_car"].empty:
    car_pct = reg_res["avg_car"] * 100
    ax2.plot(car_pct.index, car_pct.values, color="#C0392B",
             linewidth=2.5, marker="o", markersize=5)
    ax2.fill_between(car_pct.index, 0, car_pct.values, alpha=0.12, color="#C0392B")
ax2.axvline(x=0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
ax2.axhline(y=0, color="gray", linewidth=0.8, alpha=0.4)
ax2.set_title("Average CAR — Regulatory Events", fontweight="bold", fontsize=11)
ax2.set_xlabel("Days Relative to Event (t=0)")
ax2.set_ylabel("CAR (%)")
ax2.set_xticks(days[::2])
ax2.grid(True, linestyle="--", alpha=0.4)

# ── Panel C: Individual event CARs ────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, :])
colors = {"1964": "#2A5BA8", "1993": "#C0392B", "2006": "#27AE60"}
labels = {
    "1964": "1964 — Surgeon General's Report",
    "1993": "1993 — EPA Passive Smoking Ruling",
    "2006": "2006 — Judge Kessler's Ruling",
}

all_events = {**MEDIA_EVENTS, **REGULATORY_EVENTS}
for date_str, desc in all_events.items():
    year = date_str[:4]
    event_date = pd.Timestamp(date_str)
    t0 = data.index.searchsorted(event_date)
    est_start = t0 + ESTIMATION_WINDOW[0]
    est_end   = t0 + ESTIMATION_WINDOW[1]
    evt_start = t0 + EVENT_WINDOW[0]
    evt_end   = t0 + EVENT_WINDOW[1] + 1

    if est_start < 0 or evt_end > len(data):
        continue

    est_data = data.iloc[est_start:est_end]
    slope, intercept, *_ = stats.linregress(est_data["SPY"], est_data["MO"])
    evt_data = data.iloc[evt_start:evt_end]
    ar  = evt_data["MO"].values - (intercept + slope * evt_data["SPY"].values)
    car = np.cumsum(ar) * 100

    ax3.plot(days, car, color=colors.get(year, "gray"),
             linewidth=2.2, marker="o", markersize=4,
             label=labels.get(year, desc))

ax3.axvline(x=0, color="black", linestyle="--", linewidth=1.2,
            alpha=0.7, label="Event date (t=0)")
ax3.axhline(y=0, color="gray", linewidth=0.8, alpha=0.4)
ax3.set_title("Individual Event CARs — All Three Events",
              fontweight="bold", fontsize=11)
ax3.set_xlabel("Days Relative to Event (t=0)")
ax3.set_ylabel("CAR (%)")
ax3.set_xticks(days[::2])
ax3.legend(fontsize=9)
ax3.grid(True, linestyle="--", alpha=0.4)

plt.savefig("altria_event_study_results.png", dpi=150, bbox_inches="tight")
print("Chart saved → altria_event_study_results.png")
plt.show()
print("\nDone.")
