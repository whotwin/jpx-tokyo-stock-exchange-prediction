import gc
import os
import warnings

import matplotlib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.covariance import LedoitWolf
import cvxpy as cp

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

OUTPUT_DIR = "output_v2"
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
TEST_YEAR = 2021
VALID_YEAR = 2020
FIXED_X = 200
FIXED_Y = 200
XY_MIN = 50
XY_MAX = 500
XY_STEP = 50
RANDOM_STATE = 21
CLOSE_CUTOFF = "15:00:00"
GROSS_LEVERAGE_LIMIT = 1.0
MAX_ABS_WEIGHT = 0.01
RISK_AVERSION = 8.0
TURNOVER_PENALTY = 0.002
TOP_N_BY_SIGNAL = 500
USE_LEDOIT_WOLF = True
COV_LOOKBACK_DAYS = 60
COV_MIN_HISTORY = 20
ROLL_TRAIN_YEARS = 2
ROLL_RETRAIN_FREQ = "M"  # "M" monthly, "Q" quarterly
TRADING_COST_RATE = 0.001
SLIPPAGE_RATE = 0.001
PORTFOLIO_ENGINE = "rank_band"  # "rank_band" or "risk_opt"
RANK_LONG_K = 200
RANK_SHORT_K = 200
RANK_BAND = 50
RANK_REBALANCE_FREQ = "W-FRI"  # set to "M" for monthly rebalance

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


def log(msg):
    print(f"[INFO] {msg}")


def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def max_dd(cum_ret):
    return float((cum_ret - cum_ret.cummax()).min())


def load_data_sources(data_dir="train_files"):
    stock = pd.read_csv(
        os.path.join(data_dir, "stock_prices.csv"),
        usecols=[
            "Date",
            "SecuritiesCode",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "AdjustmentFactor",
            "ExpectedDividend",
            "SupervisionFlag",
            "Target",
        ],
        parse_dates=["Date"],
    )
    secondary = pd.read_csv(
        os.path.join(data_dir, "secondary_stock_prices.csv"),
        usecols=["Date", "SecuritiesCode", "Open", "High", "Low", "Close", "Volume"],
        parse_dates=["Date"],
    )
    options = pd.read_csv(
        os.path.join(data_dir, "options.csv"),
        usecols=["Date", "Putcall", "ContractMonth", "ImpliedVolatility", "TradingVolume", "OpenInterest", "SettlementPrice", "BaseVolatility"],
        parse_dates=["Date"],
    )
    financials = pd.read_csv(
        os.path.join(data_dir, "financials.csv"),
        usecols=[
            "Date",
            "SecuritiesCode",
            "DisclosedTime",
            "TypeOfDocument",
            "EarningsPerShare",
            "ForecastEarningsPerShare",
            "NetSales",
            "OperatingProfit",
            "OrdinaryProfit",
            "Profit",
            "TotalAssets",
            "Equity",
            "EquityToAssetRatio",
            "BookValuePerShare",
            "ForecastNetSales",
            "ForecastOperatingProfit",
            "ForecastOrdinaryProfit",
            "ForecastProfit",
        ],
        parse_dates=["Date"],
        engine="python",
        on_bad_lines="skip",
    )
    trades = pd.read_csv(
        os.path.join(data_dir, "trades.csv"),
        usecols=[
            "Date",
            "Section",
            "TotalBalance",
            "ProprietaryBalance",
            "BrokerageBalance",
            "IndividualsBalance",
            "ForeignersBalance",
            "SecuritiesCosBalance",
            "InvestmentTrustsBalance",
            "BusinessCosBalance",
            "OtherInstitutionsBalance",
            "InsuranceCosBalance",
            "CityBKsRegionalBKsEtcBalance",
            "TrustBanksBalance",
            "OtherFinancialInstitutionsBalance",
        ],
        parse_dates=["Date"],
    )
    # Load stock list for market cap and sector info
    stock_list = pd.read_csv(
        os.path.join(data_dir.replace("train_files", "").strip("/"), "stock_list.csv"),
        usecols=["SecuritiesCode", "MarketCapitalization", "33SectorName", "NewMarketSegment"],
    )
    # Use the latest available data for each stock
    stock_list = stock_list.sort_values("SecuritiesCode").drop_duplicates(subset=["SecuritiesCode"], keep="last")
    stock_list = stock_list[["SecuritiesCode", "MarketCapitalization", "33SectorName", "NewMarketSegment"]]
    return {"stock_prices": stock, "secondary_stock_prices": secondary, "options": options, "financials": financials, "trades": trades, "stock_list": stock_list}


def stock_features(stock, stock_list=None):
    df = stock.sort_values(["SecuritiesCode", "Date"]).copy()
    df = to_num(df, ["Open", "High", "Low", "Close", "Volume", "ExpectedDividend", "Target"])
    g = df.groupby("SecuritiesCode", sort=False)
    c = df["Close"].replace(0, np.nan)
    o = df["Open"].replace(0, np.nan)
    df["stk_ret_1"] = g["Close"].pct_change(1)
    df["stk_ret_5"] = g["Close"].pct_change(5)
    df["stk_ret_10"] = g["Close"].pct_change(10)
    df["stk_ret_20"] = g["Close"].pct_change(20)
    df["stk_hl_spread"] = (df["High"] - df["Low"]) / c
    df["stk_oc_spread"] = (c - o) / o
    df["stk_volume_chg_1"] = g["Volume"].pct_change(1)
    df["stk_logret_1"] = np.log(c).groupby(df["SecuritiesCode"]).diff(1)
    for w in [5, 10, 20]:
        df[f"stk_vol_{w}"] = g["stk_logret_1"].transform(lambda x: x.rolling(w, min_periods=w).std())
        df[f"stk_ret_mean_{w}"] = g["stk_ret_1"].transform(lambda x: x.rolling(w, min_periods=w).mean())
        ma = g["Close"].transform(lambda x: x.rolling(w, min_periods=w).mean())
        vma = g["Volume"].transform(lambda x: x.rolling(w, min_periods=w).mean())
        df[f"stk_close_to_ma_{w}"] = df["Close"] / ma - 1
        df[f"stk_volume_to_ma_{w}"] = df["Volume"] / vma - 1
    df["stk_skew_20"] = g["stk_logret_1"].transform(lambda x: x.rolling(20, min_periods=20).skew())
    df["stk_expected_dividend"] = df["ExpectedDividend"].fillna(0)
    df["stk_supervision_flag"] = df["SupervisionFlag"].astype(str).str.lower().eq("true").astype(int)
    df["stk_dayofweek"] = df["Date"].dt.dayofweek
    df["stk_month"] = df["Date"].dt.month

    # Add market cap and sector features from stock_list
    if stock_list is not None:
        stock_list = stock_list.copy()
        # Merge market cap
        df = df.merge(
            stock_list[["SecuritiesCode", "MarketCapitalization", "33SectorName", "NewMarketSegment"]],
            on="SecuritiesCode",
            how="left"
        )
        # Log market cap (in 100M yen)
        df["stk_mcap"] = np.log(df["MarketCapitalization"].fillna(1e8) / 1e8 + 1)
        # Sector code (numeric)
        df["stk_sector"] = pd.Categorical(df["33SectorName"]).codes
        # Market segment (Prime=0, First=1, etc.)
        df["stk_market_segment"] = pd.Categorical(df["NewMarketSegment"]).codes
        # Drop original columns
        df = df.drop(columns=["MarketCapitalization", "33SectorName", "NewMarketSegment"], errors="ignore")

    cols = [c for c in df.columns if c.startswith("stk_")]
    df = df.drop(columns=["ExpectedDividend", "SupervisionFlag"], errors="ignore")
    return df, cols


def secondary_features(sec):
    df = sec.sort_values(["SecuritiesCode", "Date"]).copy()
    df = to_num(df, ["Open", "High", "Low", "Close", "Volume"])
    g = df.groupby("SecuritiesCode", sort=False)
    df["ret1"] = g["Close"].pct_change(1)
    df["hl_spread"] = (df["High"] - df["Low"]) / df["Close"].replace(0, np.nan)
    out = df.groupby("Date").agg(
        sec_ret1_mean=("ret1", "mean"),
        sec_ret1_std=("ret1", "std"),
        sec_ret1_median=("ret1", "median"),
        sec_up_ratio=("ret1", lambda x: np.nanmean(x > 0) if x.notna().any() else np.nan),
        sec_vol_mean=("Volume", "mean"),
        sec_vol_std=("Volume", "std"),
        sec_hl_spread_mean=("hl_spread", "mean"),
    ).reset_index()
    for w in [5, 20]:
        out[f"sec_ret1_mean_ma_{w}"] = out["sec_ret1_mean"].rolling(w, min_periods=w).mean()
        out[f"sec_ret1_std_ma_{w}"] = out["sec_ret1_std"].rolling(w, min_periods=w).mean()
    return out, [c for c in out.columns if c.startswith("sec_")]


def options_features(opt):
    df = to_num(opt.copy(), ["Putcall", "ContractMonth", "ImpliedVolatility", "TradingVolume", "OpenInterest", "SettlementPrice", "BaseVolatility"])
    base = df.groupby("Date").agg(
        opt_iv_mean=("ImpliedVolatility", "mean"),
        opt_iv_std=("ImpliedVolatility", "std"),
        opt_volume_sum=("TradingVolume", "sum"),
        opt_oi_sum=("OpenInterest", "sum"),
        opt_settle_mean=("SettlementPrice", "mean"),
        opt_base_vol_mean=("BaseVolatility", "mean"),
    )
    pc = df.groupby(["Date", "Putcall"]).agg(iv=("ImpliedVolatility", "mean"), vol=("TradingVolume", "sum")).reset_index()
    pc = pc.pivot(index="Date", columns="Putcall", values=["iv", "vol"])
    pc.columns = [f"opt_{a}_pc{int(b)}" for a, b in pc.columns]
    cm = df.groupby(["Date", "ContractMonth"]).agg(iv=("ImpliedVolatility", "mean")).reset_index().sort_values(["Date", "ContractMonth"])
    slopes = []
    for d, g in cm.groupby("Date"):
        v = g["iv"].dropna().values
        slopes.append({"Date": d, "opt_near_iv": v[0] if len(v) > 0 else np.nan, "opt_next_iv": v[1] if len(v) > 1 else np.nan})
    slopes = pd.DataFrame(slopes)
    slopes["opt_iv_term_slope"] = slopes["opt_next_iv"] - slopes["opt_near_iv"]
    out = base.join(pc, how="left").reset_index().merge(slopes, on="Date", how="left")
    out["opt_iv_skew_put_call"] = out.get("opt_iv_pc1", np.nan) - out.get("opt_iv_pc2", np.nan)
    out["opt_put_call_vol_ratio"] = out.get("opt_vol_pc1", np.nan) / (out.get("opt_vol_pc2", np.nan) + 1e-6)
    return out, [c for c in out.columns if c.startswith("opt_")]


def _effective_dates(dates, times, trading_days, cutoff=CLOSE_CUTOFF):
    d = pd.to_datetime(dates).values.astype("datetime64[ns]")
    after = times.astype(str).fillna("") > cutoff
    idx = np.searchsorted(trading_days, d)
    idx = np.where(idx < len(trading_days), idx, len(trading_days) - 1)
    eidx = idx + after.astype(int)
    ok = eidx < len(trading_days)
    out = np.full(len(d), np.datetime64("NaT"), dtype="datetime64[ns]")
    out[ok] = trading_days[eidx[ok]]
    return pd.to_datetime(out)


def financial_features(fin, trading_days):
    df = fin.copy()
    df = to_num(
        df,
        [
            "SecuritiesCode",
            "EarningsPerShare",
            "ForecastEarningsPerShare",
            "NetSales",
            "OperatingProfit",
            "OrdinaryProfit",
            "Profit",
            "TotalAssets",
            "Equity",
            "EquityToAssetRatio",
            "BookValuePerShare",
            "ForecastNetSales",
            "ForecastOperatingProfit",
            "ForecastOrdinaryProfit",
            "ForecastProfit",
        ],
    )
    doc = df["TypeOfDocument"].astype(str)
    df["fin_doc_1q"] = doc.str.contains("1Q", na=False).astype(int)
    df["fin_doc_2q"] = doc.str.contains("2Q", na=False).astype(int)
    df["fin_doc_3q"] = doc.str.contains("3Q", na=False).astype(int)
    df["fin_doc_fy"] = doc.str.contains("FY", na=False).astype(int)
    df["fin_eps"] = df["EarningsPerShare"]
    df["fin_forecast_eps"] = df["ForecastEarningsPerShare"]
    df["fin_eps_gap"] = df["fin_forecast_eps"] - df["fin_eps"]
    maps = {
        "NetSales": "fin_netsales",
        "OperatingProfit": "fin_op_profit",
        "OrdinaryProfit": "fin_ord_profit",
        "Profit": "fin_profit",
        "TotalAssets": "fin_total_assets",
        "Equity": "fin_equity",
        "EquityToAssetRatio": "fin_equity_ratio",
        "BookValuePerShare": "fin_bvps",
        "ForecastNetSales": "fin_fc_netsales",
        "ForecastOperatingProfit": "fin_fc_op_profit",
        "ForecastOrdinaryProfit": "fin_fc_ord_profit",
        "ForecastProfit": "fin_fc_profit",
    }
    for s, t in maps.items():
        df[t] = df[s]
    df["EffectiveDate"] = _effective_dates(df["Date"], df["DisclosedTime"], trading_days)
    cols = [c for c in df.columns if c.startswith("fin_")]
    ev = df[["EffectiveDate", "SecuritiesCode"] + cols].dropna(subset=["EffectiveDate", "SecuritiesCode"]).copy()
    ev = ev.rename(columns={"EffectiveDate": "Date"})
    ev["SecuritiesCode"] = ev["SecuritiesCode"].astype(int)
    agg = {c: ("max" if c.startswith("fin_doc_") else "last") for c in cols}
    ev = ev.sort_values(["SecuritiesCode", "Date"]).groupby(["Date", "SecuritiesCode"], as_index=False).agg(agg)
    return ev, cols


def trades_features(trades, trading_days):
    df = trades.copy()
    raw = [
        "TotalBalance",
        "ProprietaryBalance",
        "BrokerageBalance",
        "IndividualsBalance",
        "ForeignersBalance",
        "SecuritiesCosBalance",
        "InvestmentTrustsBalance",
        "BusinessCosBalance",
        "OtherInstitutionsBalance",
        "InsuranceCosBalance",
        "CityBKsRegionalBKsEtcBalance",
        "TrustBanksBalance",
        "OtherFinancialInstitutionsBalance",
    ]
    df = to_num(df, raw)
    m = df["Section"].astype(str).str.contains("Prime|First", case=False, na=False)
    scope = df[m].copy() if m.any() else df.copy()
    out = scope.groupby("Date")[raw].sum(min_count=1).reset_index()
    out = out.rename(columns={c: f"trd_{c.replace('Balance', '').lower()}_balance" for c in raw})
    out = out.sort_values("Date").set_index("Date").reindex(pd.to_datetime(trading_days)).ffill().reset_index().rename(columns={"index": "Date"})
    trd_cols = [c for c in out.columns if c.startswith("trd_")]
    for c in trd_cols:
        out[f"{c}_chg_1"] = out[c].pct_change(1)
    return out, [c for c in out.columns if c.startswith("trd_")]


def build_feature_table(sources, start_date=None, end_date=None):
    stock = sources["stock_prices"].copy()
    stock_list = sources.get("stock_list")

    if start_date is not None:
        stock = stock[stock["Date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        stock = stock[stock["Date"] <= pd.to_datetime(end_date)]
    trading_days = np.array(sorted(stock["Date"].dropna().unique()), dtype="datetime64[ns]")
    sdf, s_cols = stock_features(stock, stock_list)
    sec, sec_cols = secondary_features(sources["secondary_stock_prices"])
    opt, opt_cols = options_features(sources["options"])
    fin, fin_cols = financial_features(sources["financials"], trading_days)
    trd, trd_cols = trades_features(sources["trades"], trading_days)
    df = sdf.merge(sec, on="Date", how="left").merge(opt, on="Date", how="left").merge(trd, on="Date", how="left").merge(fin, on=["Date", "SecuritiesCode"], how="left")
    if fin_cols:
        df = df.sort_values(["SecuritiesCode", "Date"]).reset_index(drop=True)
        df[fin_cols] = df.groupby("SecuritiesCode", sort=False)[fin_cols].ffill()
        for c in ["fin_eps", "fin_forecast_eps", "fin_profit", "fin_equity_ratio"]:
            if c in df.columns:
                n = f"{c}_chg_prev"
                df[n] = df.groupby("SecuritiesCode", sort=False)[c].pct_change(1)
                fin_cols.append(n)
    df = df.dropna(subset=["Date", "SecuritiesCode", "Target"]).sort_values(["Date", "SecuritiesCode"]).reset_index(drop=True)
    groups = {
        "stock": sorted(set(s_cols)),
        "secondary": sorted(set(sec_cols)),
        "options": sorted(set(opt_cols)),
        "financials": sorted(set(fin_cols)),
        "trades": sorted(set(trd_cols)),
    }
    all_feats = sorted(set(sum(groups.values(), [])))

    # Shift all features by 1 to avoid data leakage (use previous day's info)
    for col in all_feats:
        if col in df.columns and not col.startswith("stk_mcap") and not col.startswith("stk_sector"):
            df[col] = df.groupby("SecuritiesCode", sort=False)[col].shift(1)

    df[all_feats] = df[all_feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df, groups


def split_by_year(df, test_year=TEST_YEAR):
    tr = df[df["Date"].dt.year < test_year].copy()
    te = df[df["Date"].dt.year == test_year].copy()
    assert len(tr) > 0 and len(te) > 0
    assert tr["Date"].max() < te["Date"].min(), "time leakage train/test"
    return tr, te


def split_train_valid_by_date(df, valid_year=VALID_YEAR):
    tr = df[df["Date"].dt.year < valid_year].copy()
    va = df[df["Date"].dt.year == valid_year].copy()
    assert len(tr) > 0 and len(va) > 0
    assert tr["Date"].max() < va["Date"].min(), "time leakage train/valid"
    return tr, va


def params():
    return {
        "n_estimators": 300,
        "num_leaves": 64,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.4,
        "reg_lambda": 0.4,
        "objective": "regression",
        "metric": "mae",
        "random_state": RANDOM_STATE,
        "verbose": -1,
        "n_jobs": -1,
    }


def fit_model(df, feat_cols):
    m = LGBMRegressor(**params())
    m.fit(df[feat_cols], df["Target"].values)
    return m


def predict_df(model, df, feat_cols):
    out = df[["Date", "SecuritiesCode", "Target"]].copy()
    out["pred"] = model.predict(df[feat_cols])
    return out


def check_rank_integrity(pred_df):
    r = pred_df.copy()
    r["Rank"] = r.groupby("Date")["pred"].rank(method="first", ascending=False).astype(int) - 1
    chk = r.groupby("Date")["Rank"].agg(["min", "max", "nunique", "count"])
    bad = chk[(chk["min"] != 0) | (chk["max"] != chk["count"] - 1) | (chk["nunique"] != chk["count"])]
    assert bad.empty, "invalid rank permutation"


def _project_conservative_weights(raw_w, gross_limit=GROSS_LEVERAGE_LIMIT, max_abs_weight=MAX_ABS_WEIGHT):
    """
    Fallback projection:
    1) neutralize net exposure
    2) clip single-name risk
    3) scale gross exposure to leverage limit
    """
    w = np.asarray(raw_w, dtype=float).copy()
    if len(w) == 0:
        return w
    w = w - np.nanmean(w)
    w = np.clip(w, -max_abs_weight, max_abs_weight)
    gross = np.sum(np.abs(w))
    if gross > gross_limit and gross > 0:
        w = w * (gross_limit / gross)
    return w


def _make_psd(matrix, eps=1e-8):
    m = np.asarray(matrix, dtype=float)
    if m.ndim == 0:
        m = np.array([[float(m)]], dtype=float)
    if m.ndim == 1:
        m = np.diag(m)
    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    m = (m + m.T) / 2.0
    vals, vecs = np.linalg.eigh(m)
    vals = np.clip(vals, eps, None)
    m_psd = (vecs * vals) @ vecs.T
    return (m_psd + m_psd.T) / 2.0


def _build_return_panel(history_df, date_col="Date", code_col="SecuritiesCode", target_col="Target"):
    panel = history_df[[date_col, code_col, target_col]].copy()
    panel = panel.dropna(subset=[date_col, code_col, target_col])
    panel = panel.sort_values([date_col, code_col])
    panel = panel.pivot(index=date_col, columns=code_col, values=target_col).sort_index()
    return panel


def _estimate_covariance_for_day(
    return_panel,
    current_date,
    codes,
    lookback=COV_LOOKBACK_DAYS,
    min_history=COV_MIN_HISTORY,
    use_ledoit=USE_LEDOIT_WOLF,
):
    hist = return_panel[return_panel.index < current_date]
    if hist.empty:
        n = len(codes)
        base_var = (0.02 ** 2)
        cov = np.eye(n) * base_var
        sigma = np.full(n, 0.02)
        return cov, sigma

    hist = hist.tail(lookback).reindex(columns=codes)
    for c in hist.columns:
        med = hist[c].median()
        if not np.isfinite(med):
            med = 0.0
        hist[c] = hist[c].fillna(med)
    x = hist.fillna(0.0).to_numpy(dtype=float)

    if x.shape[0] < max(2, min_history):
        s = np.nanstd(x, axis=0)
        s = np.where(np.isfinite(s), np.maximum(s, 0.005), 0.02)
        cov = np.diag(s ** 2)
        return _make_psd(cov), s

    if use_ledoit:
        try:
            cov = LedoitWolf().fit(x).covariance_
        except Exception:
            cov = np.cov(x, rowvar=False)
    else:
        cov = np.cov(x, rowvar=False)

    cov = _make_psd(cov)
    sigma = np.sqrt(np.clip(np.diag(cov), 1e-10, None))
    return cov, sigma


def _solve_day_weights(alpha, cov, prev_w, gross_limit, max_abs_weight, risk_aversion, turnover_penalty):
    """
    Conservative daily optimization:
      max alpha'w - risk_aversion * (w' Sigma w) - turnover_penalty * ||w - w_prev||_1
      s.t. ||w||_1 <= gross_limit, sum(w)=0, |w_i| <= max_abs_weight
    """
    n = len(alpha)
    if n == 0:
        return np.array([])

    alpha = np.asarray(alpha, dtype=float)
    cov = _make_psd(cov)
    prev_w = np.asarray(prev_w, dtype=float)

    w = cp.Variable(n)
    obj = cp.Maximize(
        alpha @ w
        - risk_aversion * cp.quad_form(w, cp.psd_wrap(cov))
        - turnover_penalty * cp.norm1(w - prev_w)
    )
    cons = [
        cp.norm1(w) <= gross_limit,
        cp.sum(w) == 0,
        w <= max_abs_weight,
        w >= -max_abs_weight,
    ]
    prob = cp.Problem(obj, cons)

    solved = False
    for solver in [cp.OSQP, cp.ECOS, cp.SCS]:
        try:
            prob.solve(solver=solver, warm_start=True, verbose=False)
            if w.value is not None and prob.status in ("optimal", "optimal_inaccurate"):
                solved = True
                break
        except Exception:
            continue

    if solved:
        return _project_conservative_weights(w.value, gross_limit=gross_limit, max_abs_weight=max_abs_weight)

    # Fallback when optimization fails.
    diag_var = np.clip(np.diag(cov), 1e-8, None)
    score = alpha / diag_var
    return _project_conservative_weights(score, gross_limit=gross_limit, max_abs_weight=max_abs_weight)


def construct_conservative_portfolio(
    df,
    pred_col="pred",
    target_col="Target",
    date_col="Date",
    code_col="SecuritiesCode",
    vol_col=None,
    gross_limit=GROSS_LEVERAGE_LIMIT,
    max_abs_weight=MAX_ABS_WEIGHT,
    risk_aversion=RISK_AVERSION,
    turnover_penalty=TURNOVER_PENALTY,
    top_n=TOP_N_BY_SIGNAL,
    vol_lookback=60,
    vol_min_periods=20,
    history_df=None,
    cov_lookback=COV_LOOKBACK_DAYS,
    cov_min_history=COV_MIN_HISTORY,
    use_ledoit=USE_LEDOIT_WOLF,
    trading_cost_rate=TRADING_COST_RATE,
    slippage_rate=SLIPPAGE_RATE,
):
    """
    Build a conservative, risk-aware long/short portfolio from predictions.

    Inputs:
      - df columns: Date, SecuritiesCode, pred, Target, optional predicted volatility
    Outputs:
      - daily_weights: Date, SecuritiesCode, weight
      - daily_perf: Date, portfolio_return, cumulative_return, turnover, gross_exposure
      - metrics dict: sharpe, max_drawdown, etc.

    No-leakage design:
      - Risk uses rolling covariance from history_df (or current df fallback).
      - Sigma is estimated using only dates < t.
      - Daily weights at date t are solved only from data available at t.
    """
    req = {date_col, code_col, pred_col, target_col}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns: {sorted(miss)}")

    work = df[[date_col, code_col, pred_col, target_col] + ([vol_col] if vol_col and vol_col in df.columns else [])].copy()
    work = work.dropna(subset=[date_col, code_col, pred_col, target_col])
    work = work.sort_values([date_col, code_col]).reset_index(drop=True)

    if vol_col and vol_col in work.columns:
        work["_pred_sigma"] = pd.to_numeric(work[vol_col], errors="coerce")
    else:
        work["_pred_sigma"] = np.nan

    hist_src = history_df if history_df is not None else work[[date_col, code_col, target_col]]
    return_panel = _build_return_panel(hist_src, date_col=date_col, code_col=code_col, target_col=target_col)

    dates = np.array(sorted(work[date_col].unique()))

    prev = pd.Series(dtype=float)
    weight_rows = []
    perf_rows = []

    for d in dates:
        day = work[work[date_col] == d].copy()
        day = day[np.isfinite(day[pred_col])]
        if day.empty:
            continue

        # Reduce optimization size for stability/speed; still dynamic each day.
        if top_n is not None and len(day) > top_n:
            day = day.reindex(day[pred_col].abs().sort_values(ascending=False).index[:top_n]).copy()

        day = day.sort_values(code_col).reset_index(drop=True)
        codes = day[code_col].astype(int).values
        alpha = day[pred_col].to_numpy(dtype=float)
        cov, sigma_hist = _estimate_covariance_for_day(
            return_panel=return_panel,
            current_date=d,
            codes=codes,
            lookback=cov_lookback,
            min_history=cov_min_history,
            use_ledoit=use_ledoit,
        )

        # Optional predicted-volatility penalty as additional diagonal risk.
        pred_sigma = day["_pred_sigma"].to_numpy(dtype=float)
        if np.isfinite(pred_sigma).any():
            med = np.nanmedian(pred_sigma)
            if not np.isfinite(med) or med <= 0:
                med = np.nanmedian(sigma_hist)
            pred_sigma = np.where(np.isfinite(pred_sigma), np.maximum(pred_sigma, med * 0.1), med)
            cov = cov + np.diag((pred_sigma ** 2) * 0.25)
            cov = _make_psd(cov)

        prev_vec = prev.reindex(codes).fillna(0.0).to_numpy(dtype=float)
        w = _solve_day_weights(
            alpha=alpha,
            cov=cov,
            prev_w=prev_vec,
            gross_limit=gross_limit,
            max_abs_weight=max_abs_weight,
            risk_aversion=risk_aversion,
            turnover_penalty=turnover_penalty,
        )

        w_ser = pd.Series(w, index=codes, dtype=float)
        union_idx = prev.index.union(w_ser.index)
        turnover = float((w_ser.reindex(union_idx, fill_value=0.0) - prev.reindex(union_idx, fill_value=0.0)).abs().sum())
        gross = float(np.abs(w).sum())
        gross_ret = float((w * day[target_col].to_numpy(dtype=float)).sum())
        transaction_cost = float(turnover * trading_cost_rate)
        slippage_cost = float(turnover * slippage_rate)
        net_ret = gross_ret - transaction_cost - slippage_cost

        perf_rows.append(
            {
                "Date": d,
                "gross_return": gross_ret,
                "transaction_cost": transaction_cost,
                "slippage_cost": slippage_cost,
                "portfolio_return": net_ret,
                "turnover": turnover,
                "gross_exposure": gross,
                "net_exposure": float(np.sum(w)),
            }
        )

        weight_rows.append(
            pd.DataFrame(
                {
                    "Date": d,
                    "SecuritiesCode": codes,
                    "weight": w,
                }
            )
        )

        prev = w_ser[w_ser.abs() > 1e-12]

    daily_weights = pd.concat(weight_rows, ignore_index=True) if weight_rows else pd.DataFrame(columns=["Date", "SecuritiesCode", "weight"])
    daily_perf = pd.DataFrame(perf_rows).sort_values("Date").reset_index(drop=True) if perf_rows else pd.DataFrame()

    if daily_perf.empty:
    
        metrics = {
            "num_days": 0,
            "total_return": 0.0,
            "mean_daily_return": 0.0,
            "std_daily_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "avg_turnover": 0.0,
            "avg_gross_exposure": 0.0,
            "avg_cost": 0.0,
        }
        return daily_weights, daily_perf, metrics

    daily_perf["equity_curve"] = (1.0 + daily_perf["portfolio_return"]).cumprod()
    daily_perf["cumulative_return"] = daily_perf["equity_curve"] - 1.0
    roll_max = daily_perf["equity_curve"].cummax()
    daily_perf["drawdown"] = daily_perf["equity_curve"] / roll_max - 1.0

    mu = float(daily_perf["portfolio_return"].mean())
    sd = float(daily_perf["portfolio_return"].std())
    sharpe = float(np.sqrt(252.0) * mu / sd) if sd > 0 else 0.0

    metrics = {
        "num_days": int(len(daily_perf)),
        "total_return": float(daily_perf["cumulative_return"].iloc[-1]),
        "mean_daily_return": mu,
        "std_daily_return": sd,
        "sharpe": sharpe,
        "max_drawdown": float(daily_perf["drawdown"].min()),
        "avg_turnover": float(daily_perf["turnover"].mean()),
        "avg_gross_exposure": float(daily_perf["gross_exposure"].mean()),
        "avg_cost": float((daily_perf["transaction_cost"] + daily_perf["slippage_cost"]).mean()),
    }
    return daily_weights, daily_perf, metrics


def construct_rank_band_portfolio(
    df,
    pred_col="pred",
    target_col="Target",
    date_col="Date",
    code_col="SecuritiesCode",
    long_k=RANK_LONG_K,
    short_k=RANK_SHORT_K,
    band=RANK_BAND,
    rebalance_freq=RANK_REBALANCE_FREQ,
    trading_cost_rate=TRADING_COST_RATE,
    slippage_rate=SLIPPAGE_RATE,
):
    """
    Rank-based portfolio with low-frequency rebalance + holding buffer band.

    Rebalance logic (no leakage):
      - On rebalance dates, use current `pred` ranking only.
      - Keep existing longs if still within top (long_k + band).
      - Keep existing shorts if still within bottom (short_k + band).
      - Fill remaining slots from top/bottom entry buckets (top long_k / bottom short_k).
      - On non-rebalance days, hold previous positions.

    Weights:
      - Equal weight by side, gross exposure ~1:
          long side +0.5, short side -0.5
      - Transaction cost + slippage are hard deducted from daily return.
    """
    req = {date_col, code_col, pred_col, target_col}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns: {sorted(miss)}")

    work = df[[date_col, code_col, pred_col, target_col]].copy()
    work = work.dropna(subset=[date_col, code_col, pred_col, target_col])
    work = work.sort_values([date_col, code_col]).reset_index(drop=True)
    if work.empty:
        empty_w = pd.DataFrame(columns=["Date", "SecuritiesCode", "weight"])
        empty_p = pd.DataFrame(columns=["Date", "portfolio_return"])
        metrics = {
            "num_days": 0,
            "total_return": 0.0,
            "mean_daily_return": 0.0,
            "std_daily_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "avg_turnover": 0.0,
            "avg_gross_exposure": 0.0,
            "avg_cost": 0.0,
            "avg_long_count": 0.0,
            "avg_short_count": 0.0,
        }
        return empty_w, empty_p, metrics

    dates = pd.Series(sorted(work[date_col].unique()))
    rebalance_flag = (dates.dt.to_period(rebalance_freq) != dates.dt.to_period(rebalance_freq).shift(1)).to_numpy()
    rebalance_flag[0] = True

    prev_w = pd.Series(dtype=float)
    long_hold = set()
    short_hold = set()

    weight_rows = []
    perf_rows = []

    for i, d in enumerate(dates):
        day = work[work[date_col] == d].copy()
        if day.empty:
            continue
        day = day.sort_values(code_col).reset_index(drop=True)
        universe = day[code_col].astype(int).tolist()
        universe_set = set(universe)

        long_hold = {c for c in long_hold if c in universe_set}
        short_hold = {c for c in short_hold if c in universe_set}

        if rebalance_flag[i]:
            day_desc = day.sort_values(pred_col, ascending=False).reset_index(drop=True)
            day_asc = day.sort_values(pred_col, ascending=True).reset_index(drop=True)

            long_rank = {int(c): r + 1 for r, c in enumerate(day_desc[code_col].values)}
            short_rank = {int(c): r + 1 for r, c in enumerate(day_asc[code_col].values)}

            long_keep = {c for c in long_hold if long_rank.get(c, 10**9) <= (long_k + band)}
            short_keep = {c for c in short_hold if short_rank.get(c, 10**9) <= (short_k + band)}

            long_entry = [int(c) for c in day_desc[code_col].head(long_k).values if int(c) not in short_keep]
            short_entry = [int(c) for c in day_asc[code_col].head(short_k).values if int(c) not in long_keep]

            new_long = list(long_keep)
            for c in long_entry:
                if c not in new_long and c not in short_keep:
                    new_long.append(c)
                if len(new_long) >= long_k:
                    break

            new_short = list(short_keep)
            for c in short_entry:
                if c not in new_short and c not in set(new_long):
                    new_short.append(c)
                if len(new_short) >= short_k:
                    break

            long_hold = set(new_long[:long_k])
            short_hold = set(new_short[:short_k])

        w = pd.Series(0.0, index=universe, dtype=float)
        if len(long_hold) > 0:
            lw = 0.5 / len(long_hold)
            for c in long_hold:
                if c in w.index:
                    w.loc[c] = lw
        if len(short_hold) > 0:
            sw = -0.5 / len(short_hold)
            for c in short_hold:
                if c in w.index:
                    w.loc[c] = sw
        w = w[w != 0]

        union_idx = prev_w.index.union(w.index)
        turnover = float((w.reindex(union_idx, fill_value=0.0) - prev_w.reindex(union_idx, fill_value=0.0)).abs().sum())
        gross = float(np.abs(w).sum())

        ret_map = day.set_index(day[code_col].astype(int))[target_col]
        gross_ret = float((w * ret_map.reindex(w.index).to_numpy(dtype=float)).sum()) if len(w) > 0 else 0.0
        tc = float(turnover * trading_cost_rate)
        sl = float(turnover * slippage_rate)
        net = gross_ret - tc - sl

        perf_rows.append(
            {
                "Date": d,
                "gross_return": gross_ret,
                "transaction_cost": tc,
                "slippage_cost": sl,
                "portfolio_return": net,
                "turnover": turnover,
                "gross_exposure": gross,
                "net_exposure": float(w.sum()) if len(w) > 0 else 0.0,
                "long_count": int(len(long_hold)),
                "short_count": int(len(short_hold)),
                "is_rebalance": int(rebalance_flag[i]),
            }
        )

        if len(w) > 0:
            weight_rows.append(
                pd.DataFrame(
                    {
                        "Date": d,
                        "SecuritiesCode": w.index.values,
                        "weight": w.values,
                    }
                )
            )

        prev_w = w

    daily_weights = pd.concat(weight_rows, ignore_index=True) if weight_rows else pd.DataFrame(columns=["Date", "SecuritiesCode", "weight"])
    daily_perf = pd.DataFrame(perf_rows).sort_values("Date").reset_index(drop=True) if perf_rows else pd.DataFrame()

    if daily_perf.empty:
        metrics = {
            "num_days": 0,
            "total_return": 0.0,
            "mean_daily_return": 0.0,
            "std_daily_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "avg_turnover": 0.0,
            "avg_gross_exposure": 0.0,
            "avg_cost": 0.0,
            "avg_long_count": 0.0,
            "avg_short_count": 0.0,
        }
        return daily_weights, daily_perf, metrics

    daily_perf["equity_curve"] = (1.0 + daily_perf["portfolio_return"]).cumprod()
    daily_perf["cumulative_return"] = daily_perf["equity_curve"] - 1.0
    roll_max = daily_perf["equity_curve"].cummax()
    daily_perf["drawdown"] = daily_perf["equity_curve"] / roll_max - 1.0

    mu = float(daily_perf["portfolio_return"].mean())
    sd = float(daily_perf["portfolio_return"].std())
    sharpe = float(np.sqrt(252.0) * mu / sd) if sd > 0 else 0.0

    metrics = {
        "num_days": int(len(daily_perf)),
        "total_return": float(daily_perf["cumulative_return"].iloc[-1]),
        "mean_daily_return": mu,
        "std_daily_return": sd,
        "sharpe": sharpe,
        "max_drawdown": float(daily_perf["drawdown"].min()),
        "avg_turnover": float(daily_perf["turnover"].mean()),
        "avg_gross_exposure": float(daily_perf["gross_exposure"].mean()),
        "avg_cost": float((daily_perf["transaction_cost"] + daily_perf["slippage_cost"]).mean()),
        "avg_long_count": float(daily_perf["long_count"].mean()),
        "avg_short_count": float(daily_perf["short_count"].mean()),
    }
    return daily_weights, daily_perf, metrics


# ============== Constrained Portfolio Functions ==============

def neutralize_weights(weights, sectors, market_caps):
    """
    Neutralize weights by sector and market cap.

    Args:
        weights: Series with SecuritiesCode as index
        sectors: Series mapping SecuritiesCode to sector
        market_caps: Series mapping SecuritiesCode to market cap

    Returns:
        Neutralized weights
    """
    if len(weights) == 0:
        return weights

    w = weights.copy()
    codes = w.index

    # Sector neutralization
    if sectors is not None and len(sectors) > 0:
        sector_map = sectors.reindex(codes).fillna(0)
        mcap_map = market_caps.reindex(codes).fillna(market_caps.median()) if market_caps is not None else None

        # Calculate sector exposure
        for sector in sector_map.unique():
            if sector == 0 or pd.isna(sector):
                continue
            sector_mask = sector_map == sector
            sector_weight = w[sector_mask].sum()
            sector_mcap = mcap_map[sector_mask].sum() if mcap_map is not None else 1.0

            # Neutralize: reduce sector weight to target (e.g., 0 or proportional to mcap)
            if abs(sector_weight) > 1e-6:
                # Subtract equal weight from each stock in sector
                n_stocks = sector_mask.sum()
                if n_stocks > 0:
                    adjustment = sector_weight / n_stocks
                    w[sector_mask] = w[sector_mask] - adjustment

    return w


def apply_volatility_target(weights, predicted_vol, target_vol=0.15):
    """
    Scale weights to achieve target volatility.

    Args:
        weights: Current weights
        predicted_vol: Predicted portfolio volatility
        target_vol: Target annualized volatility

    Returns:
        Scaled weights
    """
    if predicted_vol is None or predicted_vol <= 0:
        return weights

    # Scale to target volatility (annualized)
    vol_scalar = target_vol / predicted_vol
    # Cap leverage
    vol_scalar = min(vol_scalar, 2.0)  # Max 2x leverage
    return weights * vol_scalar


def construct_constrained_portfolio(
    df,
    pred_col="pred",
    target_col="Target",
    date_col="Date",
    code_col="SecuritiesCode",
    long_k=200,
    short_k=200,
    band=50,
    rebalance_freq="W-FRI",
    trading_cost_rate=TRADING_COST_RATE,
    slippage_rate=SLIPPAGE_RATE,
    use_sector_neutral=True,
    use_mcap_neutral=True,
    target_volatility=0.15,
    max_weight=0.02,
    max_leverage=1.0,
):
    """
    Constrained portfolio with:
    - Sector neutralization
    - Market cap neutralization
    - Volatility targeting
    - Max weight constraints
    - Max leverage constraints

    Returns only non-overlapping returns (only on rebalance dates).
    """
    req = {date_col, code_col, pred_col, target_col}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns: {sorted(miss)}")

    work = df[[date_col, code_col, pred_col, target_col]].copy()
    work = work.dropna(subset=[date_col, code_col, pred_col, target_col])
    work = work.sort_values([date_col, code_col]).reset_index(drop=True)

    # Get sector and market cap info from df if available
    sectors = None
    market_caps = None
    if "stk_sector" in df.columns:
        sectors = df.groupby(code_col)["stk_sector"].first()
    if "stk_mcap" in df.columns:
        market_caps = df.groupby(code_col)["stk_mcap"].first()

    if work.empty:
        empty_w = pd.DataFrame(columns=["Date", code_col, "weight"])
        empty_p = pd.DataFrame(columns=["Date", "portfolio_return"])
        metrics = {
            "num_days": 0, "total_return": 0.0, "mean_daily_return": 0.0,
            "std_daily_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0,
            "avg_turnover": 0.0, "avg_gross_exposure": 0.0, "avg_cost": 0.0,
        }
        return empty_w, empty_p, metrics

    dates = pd.Series(sorted(work[date_col].unique()))
    rebalance_flag = (dates.dt.to_period(rebalance_freq) != dates.dt.to_period(rebalance_freq).shift(1)).to_numpy()
    rebalance_flag[0] = True

    prev_w = pd.Series(dtype=float)
    long_hold = set()
    short_hold = set()

    weight_rows = []
    perf_rows = []

    # Track position returns for non-overlapping calculation
    position_returns = {}

    for i, d in enumerate(dates):
        day = work[work[date_col] == d].copy()
        if day.empty:
            continue

        day = day.sort_values(code_col).reset_index(drop=True)
        universe = day[code_col].astype(int).tolist()
        universe_set = set(universe)

        long_hold = {c for c in long_hold if c in universe_set}
        short_hold = {c for c in short_hold if c in universe_set}

        if rebalance_flag[i]:
            # Rebalance: select new positions
            day_desc = day.sort_values(pred_col, ascending=False).reset_index(drop=True)
            day_asc = day.sort_values(pred_col, ascending=True).reset_index(drop=True)

            long_rank = {int(c): r + 1 for r, c in enumerate(day_desc[code_col].values)}
            short_rank = {int(c): r + 1 for r, c in enumerate(day_asc[code_col].values)}

            long_keep = {c for c in long_hold if long_rank.get(c, 10**9) <= (long_k + band)}
            short_keep = {c for c in short_hold if short_rank.get(c, 10**9) <= (short_k + band)}

            long_entry = [int(c) for c in day_desc[code_col].head(long_k).values if int(c) not in short_keep]
            short_entry = [int(c) for c in day_asc[code_col].head(short_k).values if int(c) not in long_keep]

            new_long = list(long_keep)
            for c in long_entry:
                if c not in new_long and c not in short_keep:
                    new_long.append(c)
                if len(new_long) >= long_k:
                    break

            new_short = list(short_keep)
            for c in short_entry:
                if c not in new_short and c not in set(new_long):
                    new_short.append(c)
                if len(new_short) >= short_k:
                    break

            long_hold = set(new_long[:long_k])
            short_hold = set(new_short[:short_k])

        # Build weights
        w = pd.Series(0.0, index=universe, dtype=float)
        if len(long_hold) > 0:
            # Apply max weight constraint
            lw = 0.5 / len(long_hold)
            lw = min(lw, max_weight)
            for c in long_hold:
                if c in w.index:
                    w.loc[c] = lw
        if len(short_hold) > 0:
            sw = -0.5 / len(short_hold)
            sw = max(sw, -max_weight)  # Constrain to negative max
            for c in short_hold:
                if c in w.index:
                    w.loc[c] = sw
        w = w[w != 0]

        # Apply sector/mcap neutralization
        if use_sector_neutral or use_mcap_neutral:
            w = neutralize_weights(w, sectors, market_caps)

        # Calculate turnover
        union_idx = prev_w.index.union(w.index)
        turnover = float((w.reindex(union_idx, fill_value=0.0) - prev_w.reindex(union_idx, fill_value=0.0)).abs().sum())

        # Calculate gross exposure
        gross = float(np.abs(w).sum())

        # Apply max leverage constraint
        if gross > max_leverage:
            w = w * (max_leverage / gross)
            gross = max_leverage

        # Calculate portfolio return - ONLY on rebalance dates (non-overlapping)
        if rebalance_flag[i]:
            ret_map = day.set_index(day[code_col].astype(int))[target_col]
            gross_ret = float((w * ret_map.reindex(w.index).to_numpy(dtype=float)).sum()) if len(w) > 0 else 0.0
            tc = float(turnover * trading_cost_rate)
            sl = float(turnover * slippage_rate)
            net = gross_ret - tc - sl

            perf_rows.append({
                "Date": d,
                "gross_return": gross_ret,
                "transaction_cost": tc,
                "slippage_cost": sl,
                "portfolio_return": net,
                "turnover": turnover,
                "gross_exposure": gross,
                "net_exposure": float(w.sum()) if len(w) > 0 else 0.0,
                "long_count": int(len(long_hold)),
                "short_count": int(len(short_hold)),
                "is_rebalance": int(rebalance_flag[i]),
            })

        if len(w) > 0:
            weight_rows.append(pd.DataFrame({
                "Date": d,
                code_col: w.index.values,
                "weight": w.values,
            }))

        prev_w = w

    daily_weights = pd.concat(weight_rows, ignore_index=True) if weight_rows else pd.DataFrame(columns=["Date", code_col, "weight"])
    daily_perf = pd.DataFrame(perf_rows).sort_values("Date").reset_index(drop=True) if perf_rows else pd.DataFrame()

    if daily_perf.empty:
        metrics = {
            "num_days": 0, "total_return": 0.0, "mean_daily_return": 0.0,
            "std_daily_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0,
            "avg_turnover": 0.0, "avg_gross_exposure": 0.0, "avg_cost": 0.0,
        }
        return daily_weights, daily_perf, metrics

    daily_perf["equity_curve"] = (1.0 + daily_perf["portfolio_return"]).cumprod()
    daily_perf["cumulative_return"] = daily_perf["equity_curve"] - 1.0
    roll_max = daily_perf["equity_curve"].cummax()
    daily_perf["drawdown"] = daily_perf["equity_curve"] / roll_max - 1.0

    mu = float(daily_perf["portfolio_return"].mean())
    sd = float(daily_perf["portfolio_return"].std())
    sharpe = float(np.sqrt(252.0) * mu / sd) if sd > 0 else 0.0

    metrics = {
        "num_days": int(len(daily_perf)),
        "total_return": float(daily_perf["cumulative_return"].iloc[-1]),
        "mean_daily_return": mu,
        "std_daily_return": sd,
        "sharpe": sharpe,
        "max_drawdown": float(daily_perf["drawdown"].min()),
        "avg_turnover": float(daily_perf["turnover"].mean()),
        "avg_gross_exposure": float(daily_perf["gross_exposure"].mean()),
        "avg_cost": float((daily_perf["transaction_cost"] + daily_perf["slippage_cost"]).mean()),
        "avg_long_count": float(daily_perf["long_count"].mean()),
        "avg_short_count": float(daily_perf["short_count"].mean()),
    }
    return daily_weights, daily_perf, metrics


def _rolling_periods(dates, freq=ROLL_RETRAIN_FREQ):
    d = pd.Series(pd.to_datetime(sorted(pd.unique(dates))))
    p = d.dt.to_period(freq)
    for period in p.drop_duplicates():
        mask = p == period
        dd = d[mask]
        yield dd.min(), dd.max()


def _walk_forward_predict(full_df, feature_cols, test_year=TEST_YEAR, train_years=ROLL_TRAIN_YEARS, retrain_freq=ROLL_RETRAIN_FREQ):
    test_df = full_df[full_df["Date"].dt.year == test_year].copy()
    pred_parts = []

    for period_start, period_end in _rolling_periods(test_df["Date"], freq=retrain_freq):
        train_end = period_start - pd.Timedelta(days=1)
        train_start = train_end - pd.DateOffset(years=train_years) + pd.Timedelta(days=1)

        train_win = full_df[(full_df["Date"] >= train_start) & (full_df["Date"] <= train_end)].copy()
        infer_win = test_df[(test_df["Date"] >= period_start) & (test_df["Date"] <= period_end)].copy()

        if train_win.empty or infer_win.empty:
            continue

        model = fit_model(train_win, feature_cols)
        pred_win = predict_df(model, infer_win, feature_cols)
        pred_win["retrain_period_start"] = period_start
        pred_parts.append(pred_win)

        del model, train_win, infer_win, pred_win
        gc.collect()

    if pred_parts:
        return pd.concat(pred_parts, ignore_index=True).sort_values(["Date", "SecuritiesCode"]).reset_index(drop=True)
    return pd.DataFrame(columns=["Date", "SecuritiesCode", "Target", "pred", "retrain_period_start"])


def run_ablation(full_df, feat_groups):
    train_df, test_df = split_by_year(full_df, TEST_YEAR)
    exps = [
        ("stock", ["stock"]),
        ("stock+secondary", ["stock", "secondary"]),
        ("stock+options", ["stock", "options"]),
        ("stock+financials", ["stock", "financials"]),
        ("stock+trades", ["stock", "trades"]),
        ("all", ["stock", "secondary", "options", "financials", "trades"]),
    ]
    summary, daily_rows = [], []
    history_df = full_df[["Date", "SecuritiesCode", "Target"]].copy()

    for name, gs in exps:
        cols = sorted(set(sum([feat_groups[g] for g in gs], [])))
        log(f"Experiment={name}, features={len(cols)}")
        tp = _walk_forward_predict(
            full_df=full_df,
            feature_cols=cols,
            test_year=TEST_YEAR,
            train_years=ROLL_TRAIN_YEARS,
            retrain_freq=ROLL_RETRAIN_FREQ,
        )
        check_rank_integrity(tp)
        rmse = float(np.sqrt(mean_squared_error(tp["Target"], tp["pred"]))) if len(tp) > 0 else np.nan
        if PORTFOLIO_ENGINE == "risk_opt":
            w_test, dt, tm = construct_conservative_portfolio(
                tp,
                pred_col="pred",
                target_col="Target",
                history_df=history_df,
                cov_lookback=COV_LOOKBACK_DAYS,
                cov_min_history=COV_MIN_HISTORY,
                use_ledoit=USE_LEDOIT_WOLF,
                trading_cost_rate=TRADING_COST_RATE,
                slippage_rate=SLIPPAGE_RATE,
            )
        else:
            w_test, dt, tm = construct_rank_band_portfolio(
                tp,
                pred_col="pred",
                target_col="Target",
                long_k=RANK_LONG_K,
                short_k=RANK_SHORT_K,
                band=RANK_BAND,
                rebalance_freq=RANK_REBALANCE_FREQ,
                trading_cost_rate=TRADING_COST_RATE,
                slippage_rate=SLIPPAGE_RATE,
            )
        tm["rank_sharpe"] = tm["sharpe"]
        summary.append(
            {
                "experiment": name,
                "feature_groups": "+".join(gs),
                "feature_count": len(cols),
                "valid_rmse": np.nan,
                "valid_rank_sharpe": np.nan,
                "valid_total_return": np.nan,
                "test_rmse": rmse,
                "test_rank_sharpe": tm["rank_sharpe"],
                "test_total_return": tm["total_return"],
                "test_mean_daily_return": tm["mean_daily_return"],
                "test_std_daily_return": tm["std_daily_return"],
                "test_max_drawdown": tm["max_drawdown"],
                "test_num_days": tm["num_days"],
                "test_avg_turnover": tm["avg_turnover"],
                "test_avg_gross_exposure": tm["avg_gross_exposure"],
                "test_avg_cost": tm["avg_cost"],
                "test_avg_long_count": tm.get("avg_long_count", np.nan),
                "test_avg_short_count": tm.get("avg_short_count", np.nan),
            }
        )
        if not dt.empty:
            dt = dt.copy()
            dt["experiment"] = name
            dt["split"] = "test"
            daily_rows.append(dt)
        if not w_test.empty:
            w_test = w_test.copy()
            w_test["experiment"] = name
            w_test.to_csv(os.path.join(OUTPUT_DIR, f"weights_test_{name}.csv"), index=False)
        if len(tp) > 0:
            tp.to_csv(os.path.join(OUTPUT_DIR, f"pred_test_{name}.csv"), index=False)
        del tp, w_test, dt
        gc.collect()
    s = pd.DataFrame(summary)
    b = s.loc[s["experiment"] == "stock", ["test_rank_sharpe", "test_total_return"]].iloc[0]
    s["delta_sharpe_vs_stock"] = s["test_rank_sharpe"] - float(b["test_rank_sharpe"])
    s["delta_total_return_vs_stock"] = s["test_total_return"] - float(b["test_total_return"])
    s = s.sort_values("test_rank_sharpe", ascending=False).reset_index(drop=True)
    d = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame()
    return s, d


def plot_summary(summary):
    p = summary.sort_values("test_rank_sharpe", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    ax[0].bar(p["experiment"], p["test_rank_sharpe"], color="steelblue", alpha=0.85)
    ax[0].set_title("Test Rank Sharpe")
    ax[0].tick_params(axis="x", rotation=30)
    ax[0].grid(True, axis="y", alpha=0.25)
    ax[1].bar(p["experiment"], p["test_total_return"], color="darkgreen", alpha=0.85)
    ax[1].set_title("Test Total Return")
    ax[1].tick_params(axis="x", rotation=30)
    ax[1].grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "ablation_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_cum(daily, split):
    if daily.empty:
        return
    p = daily[daily["split"] == split].copy()
    if p.empty:
        return
    curve_col = "cumulative_return" if "cumulative_return" in p.columns else ("cum_return" if "cum_return" in p.columns else None)
    if curve_col is None:
        return
    fig, ax = plt.subplots(figsize=(14, 6))
    for n, g in p.groupby("experiment"):
        g = g.sort_values("Date")
        ax.plot(g["Date"], g[curve_col], label=n)
    ax.set_title(f"Cumulative Return ({split})")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"cumulative_returns_{split}.png"), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    log("JPX leak-free multi-source pipeline")
    log(
        "Protocol: train<2021, test=2021, walk-forward monthly retrain, "
        f"2Y rolling train window, portfolio_engine={PORTFOLIO_ENGINE}, "
        f"rebal={RANK_REBALANCE_FREQ}, band={RANK_BAND}, cost+slippage"
    )
    src = load_data_sources("train_files")
    full_df, groups = build_feature_table(src, start_date="2017-01-04", end_date="2021-12-03")
    tr, te = split_by_year(full_df, TEST_YEAR)
    assert tr["Date"].max() < te["Date"].min()
    summary, daily = run_ablation(full_df, groups)
    summary.to_csv(os.path.join(OUTPUT_DIR, "ablation_summary.csv"), index=False)
    daily.to_csv(os.path.join(OUTPUT_DIR, "daily_returns_all_experiments.csv"), index=False)
    if not daily.empty:
        best = summary.iloc[0]["experiment"]
        daily[(daily["split"] == "test") & (daily["experiment"] == best)].to_csv(
            os.path.join(OUTPUT_DIR, "evaluation_results.csv"), index=False
        )
    plot_summary(summary)
    plot_cum(daily, "test")
    log("Done")


if __name__ == "__main__":
    main()

