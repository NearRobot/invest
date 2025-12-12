#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""中国永久投资组合（Permanent Portfolio）回测与分析

本脚本尽量做到“开箱即用”：
- 优先尝试从 yfinance / tushare 拉取数据；
- 若因网络/依赖/权限（tushare token）等原因无法获取，则自动使用可复现的“合成历史数据”
  来完成回测流程与报告/图表/导出。

资产配置（四等分）：
- 沪深300（用 510300/指数代理）：25%
- 30年期国债（用 ETF/指数代理）：25%
- 黄金（国内黄金 ETF 或 GC=F + USDCNY）：25%
- 货币基金（511990 或低波动现金代理）：25%

说明：
中国市场可公开、可免费长期连续获得的“可交易标的”历史往往晚于 2000 年。
因此，当真实数据起始日晚于回测起点时，脚本会对缺口区间使用“代理合成序列”进行衔接，
并在终端报告中输出提示。

运行：
python permanent_portfolio_backtest.py

输出：
- 终端打印回测报告
- output/ 下生成 PNG 图表、CSV/Excel 数据导出
"""

from __future__ import annotations

import os
import math
import socket
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

# 无界面环境绘图
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns  # type: ignore
except Exception:  # pragma: no cover
    sns = None

try:
    from scipy import stats  # type: ignore
except Exception:  # pragma: no cover
    stats = None


# -----------------------------
# 全局配置
# -----------------------------

START_DATE = "2000-01-01"
END_DATE = "2024-12-31"
INITIAL_CAPITAL = 1_000_000

RF_ANNUAL = 0.02  # 夏普/索蒂诺：无风险利率（年化）
DEPOSIT_BENCH_ANNUAL = 0.03

TARGET_WEIGHTS: Dict[str, float] = {
    "stock": 0.25,
    "bond": 0.25,
    "gold": 0.25,
    "money": 0.25,
}

ASSET_CN_NAME = {
    "stock": "沪深300",
    "bond": "30年国债",
    "gold": "黄金",
    "money": "货币基金",
}


# -----------------------------
# 工具函数：中文字体
# -----------------------------

def setup_chinese_font() -> None:
    """尽量配置中文字体，确保中文标签可读。"""

    # 常见中文字体：SimHei / Microsoft YaHei / PingFang / Noto Sans CJK
    candidate_fonts = [
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]

    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for font in candidate_fonts:
        if font in available:
            plt.rcParams["font.sans-serif"] = [font]
            break

    plt.rcParams["axes.unicode_minus"] = False


# -----------------------------
# 数据获取模块
# -----------------------------

def _try_import_yfinance():
    try:
        import yfinance as yf  # type: ignore

        return yf
    except Exception:
        return None


def _try_import_tushare():
    try:
        import tushare as ts  # type: ignore

        return ts
    except Exception:
        return None


@lru_cache(maxsize=1)
def _yahoo_reachable(timeout: float = 2.0) -> bool:
    """快速网络探测：若无法连通 Yahoo Finance 域名，则跳过在线拉取。

    目的：避免在无网络环境下 yfinance 反复超时导致脚本卡住。
    """

    try:
        conn = socket.create_connection(("query1.finance.yahoo.com", 443), timeout=timeout)
        conn.close()
        return True
    except Exception:
        return False


def _download_yfinance_adj_close(
    ticker: str,
    start: str,
    end: str,
) -> Optional[pd.Series]:
    """从 yfinance 下载单一 ticker 的复权收盘价（Adj Close/Close）。"""

    yf = _try_import_yfinance()
    if yf is None:
        return None

    # 无网络时快速退出
    if not _yahoo_reachable():
        return None

    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            return None

        # yfinance 中国标的有时没有 Adj Close
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        s = df[col].dropna().copy()
        s.name = ticker
        s.index = pd.to_datetime(s.index)
        return s
    except Exception:
        return None


def _generate_synthetic_prices(
    dates: pd.DatetimeIndex,
    annual_mu: float,
    annual_sigma: float,
    seed: int,
    start_price: float = 100.0,
    shocks: Optional[Iterable[Tuple[str, str, float]]] = None,
) -> pd.Series:
    """生成可复现的合成价格序列。

    参数
    - annual_mu: 年化期望收益
    - annual_sigma: 年化波动率
    - shocks: 事件冲击列表 (start, end, daily_drift)，例如在危机期间加入负漂移
    """

    rng = np.random.default_rng(seed)
    n = len(dates)

    mu_d = annual_mu / 252.0
    sigma_d = annual_sigma / math.sqrt(252.0)
    rets = rng.normal(loc=mu_d, scale=sigma_d, size=n)

    if shocks:
        for s, e, drift in shocks:
            mask = (dates >= pd.Timestamp(s)) & (dates <= pd.Timestamp(e))
            rets[mask] += drift

    prices = start_price * np.exp(np.cumsum(rets))
    return pd.Series(prices, index=dates)


def _stitch_with_proxy(
    target_dates: pd.DatetimeIndex,
    real: pd.Series,
    proxy_mu: float,
    proxy_sigma: float,
    seed: int,
) -> Tuple[pd.Series, str]:
    """当真实数据起始日晚于回测起点时，用合成序列向前补齐并缩放衔接。"""

    note = ""
    real = real.sort_index()
    real_start = real.index.min()

    if real_start <= target_dates.min():
        stitched = real.reindex(target_dates).ffill()
        return stitched, note

    # 需要向前补齐
    proxy_dates = target_dates[target_dates < real_start]
    proxy = _generate_synthetic_prices(
        proxy_dates,
        annual_mu=proxy_mu,
        annual_sigma=proxy_sigma,
        seed=seed,
        start_price=100.0,
    )

    # 缩放使得 proxy 最后一天与 real 第一天对齐
    scale = float(real.iloc[0]) / float(proxy.iloc[-1]) if len(proxy) else 1.0
    proxy *= scale

    stitched = pd.concat([proxy, real]).reindex(target_dates).ffill()

    note = (
        f"真实数据起始日为 {real_start.date()}，"
        f"已用代理合成数据补齐 {target_dates.min().date()} ~ {(real_start - pd.Timedelta(days=1)).date()}。"
    )
    return stitched, note


def fetch_stock_data(start: str, end: str) -> Tuple[pd.Series, str]:
    """获取沪深300资产价格序列。"""

    notes = []

    # 优先 ETF：510300（华泰柏瑞沪深300ETF）
    candidates = ["510300.SS", "000300.SS", "399300.SZ", "^SSEC"]
    s = None
    used = None
    for t in candidates:
        s = _download_yfinance_adj_close(t, start=start, end=end)
        if s is not None and len(s) > 200:
            used = t
            break

    target_dates = pd.date_range(start=start, end=end, freq="B")

    if s is None:
        # 完全无法拉取：用合成数据
        s = _generate_synthetic_prices(
            target_dates,
            annual_mu=0.08,
            annual_sigma=0.20,
            seed=1,
            start_price=100.0,
            shocks=[
                ("2008-09-01", "2009-03-31", -0.0025),
                ("2015-06-01", "2015-09-30", -0.0018),
                ("2018-06-01", "2018-12-31", -0.0008),
            ],
        )
        notes.append("沪深300：未获取到真实数据，已使用合成数据（示例）代替。")
        return s.rename("stock"), "；".join(notes)

    s = s.rename("stock")
    stitched, note = _stitch_with_proxy(target_dates, s, proxy_mu=0.08, proxy_sigma=0.20, seed=11)
    if used:
        notes.append(f"沪深300：使用 yfinance 标的 {used}。")
    if note:
        notes.append(f"沪深300：{note}")

    return stitched, "；".join(notes)


def fetch_bond_data(start: str, end: str) -> Tuple[pd.Series, str]:
    """获取长期国债资产价格序列。

    现实中“30年期国债ETF/指数”长期连续数据较难免费获取。
    本脚本优先尝试 511090（国债ETF），否则使用合成代理。
    """

    notes = []
    candidates = ["511090.SS", "511010.SS"]
    s = None
    used = None
    for t in candidates:
        s = _download_yfinance_adj_close(t, start=start, end=end)
        if s is not None and len(s) > 200:
            used = t
            break

    target_dates = pd.date_range(start=start, end=end, freq="B")

    if s is None:
        s = _generate_synthetic_prices(
            target_dates,
            annual_mu=0.04,
            annual_sigma=0.06,
            seed=2,
            start_price=100.0,
            shocks=[
                ("2013-06-01", "2013-07-31", -0.0006),
                ("2022-11-01", "2022-12-31", -0.0005),
            ],
        )
        notes.append("国债：未获取到真实数据，已使用合成数据（示例）代替。")
        return s.rename("bond"), "；".join(notes)

    s = s.rename("bond")
    stitched, note = _stitch_with_proxy(target_dates, s, proxy_mu=0.04, proxy_sigma=0.06, seed=22)
    notes.append(f"国债：使用 yfinance 标的 {used}。")
    if note:
        notes.append(f"国债：{note}")

    # 额外提示：在真实场景可能用 10 年期代理
    notes.append("提示：若需严格30年期，可接入中债指数或收益率久期推导；本实现为工程化示例。")
    return stitched, "；".join(notes)


def fetch_gold_data(start: str, end: str) -> Tuple[pd.Series, str]:
    """获取黄金资产价格序列。

    优先：国内黄金ETF 518880
    备选：GC=F（美元黄金期货）* USDCNY=X（美元兑人民币）合成人民币金价
    """

    notes = []
    target_dates = pd.date_range(start=start, end=end, freq="B")

    etf = _download_yfinance_adj_close("518880.SS", start=start, end=end)
    if etf is not None and len(etf) > 200:
        etf = etf.rename("gold")
        stitched, note = _stitch_with_proxy(target_dates, etf, proxy_mu=0.05, proxy_sigma=0.15, seed=33)
        notes.append("黄金：使用 yfinance 标的 518880.SS（黄金ETF）。")
        if note:
            notes.append(f"黄金：{note}")
        return stitched, "；".join(notes)

    gold_usd = _download_yfinance_adj_close("GC=F", start=start, end=end)
    usdcny = _download_yfinance_adj_close("USDCNY=X", start=start, end=end)

    if gold_usd is not None and usdcny is not None and len(gold_usd) > 200 and len(usdcny) > 200:
        df = pd.concat([gold_usd.rename("gold_usd"), usdcny.rename("usdcny")], axis=1).dropna()
        cny = (df["gold_usd"] * df["usdcny"]).rename("gold")
        stitched, note = _stitch_with_proxy(target_dates, cny, proxy_mu=0.05, proxy_sigma=0.15, seed=44)
        notes.append("黄金：使用 GC=F * USDCNY=X 合成人民币金价。")
        if note:
            notes.append(f"黄金：{note}")
        return stitched, "；".join(notes)

    # 合成
    s = _generate_synthetic_prices(
        target_dates,
        annual_mu=0.05,
        annual_sigma=0.15,
        seed=3,
        start_price=100.0,
        shocks=[
            ("2008-09-01", "2009-03-31", 0.0012),
            ("2020-03-01", "2020-08-31", 0.0008),
        ],
    )
    notes.append("黄金：未获取到真实数据，已使用合成数据（示例）代替。")
    return s.rename("gold"), "；".join(notes)


def fetch_money_data(start: str, end: str) -> Tuple[pd.Series, str]:
    """获取货币基金/现金资产价格序列。

    优先：511990（华宝添益）
    备选：合成“低波动、接近无风险”的净值曲线
    """

    notes = []
    target_dates = pd.date_range(start=start, end=end, freq="B")

    s = _download_yfinance_adj_close("511990.SS", start=start, end=end)
    if s is not None and len(s) > 200:
        s = s.rename("money")
        stitched, note = _stitch_with_proxy(target_dates, s, proxy_mu=0.02, proxy_sigma=0.01, seed=55)
        notes.append("货币基金：使用 yfinance 标的 511990.SS。")
        if note:
            notes.append(f"货币基金：{note}")
        return stitched, "；".join(notes)

    # 合成：以 2% 左右年化，极低波动
    s = _generate_synthetic_prices(
        target_dates,
        annual_mu=0.02,
        annual_sigma=0.01,
        seed=4,
        start_price=100.0,
    )
    notes.append("货币基金：未获取到真实数据，已使用合成数据（示例）代替。")
    return s.rename("money"), "；".join(notes)


def fetch_all_data(start: str, end: str) -> Tuple[pd.DataFrame, str]:
    """获取并对齐四类资产价格数据。"""

    series = {}
    notes = []

    stock, n1 = fetch_stock_data(start, end)
    bond, n2 = fetch_bond_data(start, end)
    gold, n3 = fetch_gold_data(start, end)
    money, n4 = fetch_money_data(start, end)

    for n in [n1, n2, n3, n4]:
        if n:
            notes.append(n)

    series["stock"] = stock
    series["bond"] = bond
    series["gold"] = gold
    series["money"] = money

    prices = pd.concat(series.values(), axis=1)
    prices.columns = list(series.keys())

    # 清理：前向填充，确保没有空值
    prices = prices.sort_index().ffill().dropna()

    return prices, "\n".join(f"- {x}" for x in notes)


# -----------------------------
# 回测引擎模块
# -----------------------------


@dataclass
class BacktestResult:
    nav: pd.Series
    daily_returns: pd.Series
    weights: pd.DataFrame
    holdings_value: pd.DataFrame
    rebalance_log: pd.DataFrame


def _first_trading_day_flags(dates: pd.DatetimeIndex, freq: str) -> pd.Series:
    """生成“某频率下首个交易日”的布尔序列。

    freq: "M"(月) / "Q"(季) / "A"(年)
    """

    p = dates.to_period(freq)
    return pd.Series(p != p.shift(1), index=dates)


class Portfolio:
    """简化的日频回测组合。"""

    def __init__(
        self,
        prices: pd.DataFrame,
        target_weights: Dict[str, float],
        initial_capital: float = INITIAL_CAPITAL,
    ) -> None:
        self.prices = prices.copy()
        self.assets = list(target_weights.keys())
        self.target = pd.Series(target_weights).reindex(self.assets).astype(float)
        self.target /= self.target.sum()
        self.initial_capital = float(initial_capital)

        if set(self.assets) != set(self.prices.columns):
            raise ValueError("prices 列必须包含：" + ",".join(self.assets))

    def run(
        self,
        rebalance_mode: str = "annual",
        deviation_threshold: float = 0.10,
        periodic_freq: str = "A",
        event_drawdown_threshold: float = 0.15,
    ) -> BacktestResult:
        """运行回测。

        rebalance_mode:
        - annual: 每年首个交易日再平衡
        - deviation: 权重偏离目标超过阈值（绝对偏离）触发
        - event: 极端波动/回撤触发
        - periodic: 按 periodic_freq (M/Q/A) 定期再平衡
        - none: 不进行再平衡（买入并持有）
        """

        prices = self.prices
        dates = prices.index

        # 初始建仓
        p0 = prices.iloc[0]
        units = (self.initial_capital * self.target / p0).astype(float)

        nav_list = []
        weights_list = []
        holdings_value_list = []
        rebalance_records = []

        # 预计算辅助
        asset_returns = prices.pct_change().fillna(0.0)
        flags_annual = _first_trading_day_flags(dates, "A")
        flags_periodic = _first_trading_day_flags(dates, periodic_freq)

        running_peak = -np.inf

        for i, dt in enumerate(dates):
            px = prices.loc[dt]
            holding_value = units * px
            port_value = float(holding_value.sum())

            nav_list.append(port_value)
            w = (holding_value / port_value).astype(float)
            weights_list.append(w)
            holdings_value_list.append(holding_value)

            # 用当前净值更新峰值/回撤
            running_peak = max(running_peak, port_value)
            dd = (port_value / running_peak) - 1.0 if running_peak > 0 else 0.0

            # 判断是否需要再平衡（在收盘价执行）
            reason = None
            if i == 0:
                reason = None
            elif rebalance_mode == "none":
                reason = None
            elif rebalance_mode == "annual":
                if bool(flags_annual.loc[dt]):
                    reason = "年度再平衡"
            elif rebalance_mode == "periodic":
                if bool(flags_periodic.loc[dt]):
                    reason = f"定期再平衡({periodic_freq})"
            elif rebalance_mode == "deviation":
                if float((w - self.target).abs().max()) >= deviation_threshold:
                    reason = f"偏离度再平衡(阈值±{deviation_threshold:.0%})"
            elif rebalance_mode == "event":
                # 事件驱动：组合回撤超过阈值；或“风险资产”（优先股票）单日极端波动
                if "stock" in asset_returns.columns:
                    risk_r = float(asset_returns.loc[dt, "stock"])
                else:
                    risk_r = float(asset_returns.loc[dt].iloc[0])

                if dd <= -abs(event_drawdown_threshold):
                    reason = f"事件驱动(回撤≤-{abs(event_drawdown_threshold):.0%})"
                elif abs(risk_r) >= 0.07:
                    reason = "事件驱动(风险资产单日极端波动)"
            else:
                raise ValueError(f"未知 rebalance_mode: {rebalance_mode}")

            if reason:
                before_w = w
                before_units = units.copy()

                target_value = port_value * self.target
                new_units = (target_value / px).astype(float)
                trade_value = (new_units - before_units) * px

                turnover = float(trade_value.abs().sum() / port_value) if port_value > 0 else 0.0

                units = new_units
                after_value = units * px
                after_w = (after_value / float(after_value.sum())).astype(float)

                rec = {
                    "date": dt,
                    "reason": reason,
                    "portfolio_value_before": port_value,
                    "turnover": turnover,
                }
                for a in self.assets:
                    rec[f"trade_value_{a}"] = float(trade_value[a])
                    rec[f"trade_units_{a}"] = float(new_units[a] - before_units[a])
                    rec[f"weight_before_{a}"] = float(before_w[a])
                    rec[f"weight_after_{a}"] = float(after_w[a])

                rebalance_records.append(rec)

        nav = pd.Series(nav_list, index=dates, name="nav")
        daily_returns = nav.pct_change().fillna(0.0).rename("portfolio_return")
        weights = pd.DataFrame(weights_list, index=dates)
        holdings_value = pd.DataFrame(holdings_value_list, index=dates)
        rebalance_log = pd.DataFrame(rebalance_records)

        return BacktestResult(
            nav=nav,
            daily_returns=daily_returns,
            weights=weights,
            holdings_value=holdings_value,
            rebalance_log=rebalance_log,
        )


# -----------------------------
# 性能分析模块
# -----------------------------


def _annualized_return(nav: pd.Series) -> float:
    nav = nav.dropna()
    if len(nav) < 2:
        return 0.0
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return float((nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0)


def _max_drawdown(nav: pd.Series) -> Tuple[float, pd.Series]:
    nav = nav.dropna()
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min()), dd.rename("drawdown")


def _annualized_vol(daily_returns: pd.Series) -> float:
    return float(daily_returns.std(ddof=0) * math.sqrt(252.0))


def _sharpe_ratio(daily_returns: pd.Series, rf_annual: float = RF_ANNUAL) -> float:
    r = daily_returns.dropna()
    if r.std(ddof=0) == 0:
        return 0.0
    rf_daily = (1 + rf_annual) ** (1 / 252.0) - 1
    excess = r - rf_daily
    return float(excess.mean() / excess.std(ddof=0) * math.sqrt(252.0))


def _sortino_ratio(daily_returns: pd.Series, rf_annual: float = RF_ANNUAL) -> float:
    r = daily_returns.dropna()
    rf_daily = (1 + rf_annual) ** (1 / 252.0) - 1
    excess = r - rf_daily
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")
    downside_std = downside.std(ddof=0)
    if downside_std == 0:
        return float("inf")
    return float(excess.mean() / downside_std * math.sqrt(252.0))


def _calmar_ratio(nav: pd.Series) -> float:
    cagr = _annualized_return(nav)
    mdd, _ = _max_drawdown(nav)
    if mdd == 0:
        return float("inf")
    return float(cagr / abs(mdd))


def _win_rate_monthly(daily_returns: pd.Series) -> float:
    m = (1 + daily_returns).resample("M").prod() - 1
    if len(m) == 0:
        return 0.0
    return float((m > 0).mean())


def performance_summary(nav: pd.Series, daily_returns: pd.Series) -> Dict[str, float]:
    cagr = _annualized_return(nav)
    mdd, _ = _max_drawdown(nav)
    vol = _annualized_vol(daily_returns)
    sharpe = _sharpe_ratio(daily_returns)
    sortino = _sortino_ratio(daily_returns)
    calmar = _calmar_ratio(nav)
    win_rate = _win_rate_monthly(daily_returns)

    return {
        "cagr": cagr,
        "mdd": mdd,
        "vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "win_rate_month": win_rate,
    }


def segment_metrics(result: BacktestResult, start: str, end: str) -> Dict[str, float]:
    nav = result.nav.loc[start:end]
    r = result.daily_returns.loc[start:end]
    if nav.empty:
        return {"cagr": 0.0, "mdd": 0.0}
    cagr = _annualized_return(nav)
    mdd, _ = _max_drawdown(nav)
    vol = _annualized_vol(r)
    return {"cagr": cagr, "mdd": mdd, "vol": vol}


def asset_annualized_returns(prices: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for col in prices.columns:
        nav = prices[col].dropna()
        out[col] = _annualized_return(nav)
    return out


def attribution_contribution(prices: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """用“权重×资产收益”做近似贡献归因（用于报告展示）。"""

    asset_ret = prices.pct_change().fillna(0.0)
    w_prev = weights.shift(1).bfill()
    contrib_daily = w_prev * asset_ret
    per_asset = contrib_daily.sum()
    total = float(per_asset.sum())
    contrib = per_asset / total if total != 0 else per_asset * 0
    return contrib.rename("contribution")


# -----------------------------
# 报告与可视化
# -----------------------------


def format_pct(x: float, digits: int = 2) -> str:
    return f"{x * 100:.{digits}f}%"


def format_num(x: float, digits: int = 0) -> str:
    return f"{x:,.{digits}f}"


def generate_report(
    prices: pd.DataFrame,
    pp_result: BacktestResult,
    bench_stock: BacktestResult,
    bench_5050: BacktestResult,
    bench_deposit: BacktestResult,
    alt_portfolios: Dict[str, BacktestResult],
    data_notes: str,
    output_dir: str,
) -> str:
    """生成终端报告文本。"""

    summary = performance_summary(pp_result.nav, pp_result.daily_returns)
    final_value = float(pp_result.nav.iloc[-1])
    total_return = final_value / INITIAL_CAPITAL - 1.0

    seg1 = segment_metrics(pp_result, "2005-01-01", "2010-12-31")
    seg2 = segment_metrics(pp_result, "2011-01-01", "2018-12-31")
    seg3 = segment_metrics(pp_result, "2021-01-01", "2024-12-31")

    # 再平衡贡献：与“买入并持有”对比
    buy_hold = Portfolio(prices, TARGET_WEIGHTS, INITIAL_CAPITAL).run(rebalance_mode="none")
    rebalance_extra = float(pp_result.nav.iloc[-1] / buy_hold.nav.iloc[-1] - 1.0)

    reb_count = int(len(pp_result.rebalance_log))

    # 各资产表现/贡献
    asset_ret = asset_annualized_returns(prices)
    contrib = attribution_contribution(prices, pp_result.weights)

    # 对比基准
    stock_sum = performance_summary(bench_stock.nav, bench_stock.daily_returns)
    b5050_sum = performance_summary(bench_5050.nav, bench_5050.daily_returns)
    deposit_sum = performance_summary(bench_deposit.nav, bench_deposit.daily_returns)

    report = []
    report.append("========== 中国永久投资组合 回测报告 ==========")
    report.append(f"初始投资：{format_num(INITIAL_CAPITAL)} 元")
    report.append(
        f"投资周期：{pd.Timestamp(prices.index[0]).date()} - {pd.Timestamp(prices.index[-1]).date()}（~{(prices.index[-1]-prices.index[0]).days/365.25:.1f}年）"
    )
    report.append("")

    report.append("【整体表现】")
    report.append(f"期末总值：{format_num(final_value)} 元")
    report.append(f"累计收益：{format_num(final_value-INITIAL_CAPITAL)} 元（{format_pct(total_return)}）")
    report.append(f"年化收益率：{format_pct(summary['cagr'])}")
    report.append(f"最大回撤：{format_pct(summary['mdd'])}")
    report.append(f"夏普比率：{summary['sharpe']:.2f}（无风险利率 {RF_ANNUAL:.0%}）")
    report.append(f"卡尔玛比率：{summary['calmar']:.2f}")
    report.append(f"索蒂诺比率：{summary['sortino']:.2f}")
    report.append(f"月度胜率：{format_pct(summary['win_rate_month'])}")
    report.append("")

    report.append("【分段表现】")
    report.append(
        f"- 2005-2010（高速增长）：年化 {format_pct(seg1['cagr'])}，回撤 {format_pct(seg1['mdd'])}"
    )
    report.append(
        f"- 2011-2018（转型波动）：年化 {format_pct(seg2['cagr'])}，回撤 {format_pct(seg2['mdd'])}"
    )
    report.append(
        f"- 2021-2024（低利率）：年化 {format_pct(seg3['cagr'])}，回撤 {format_pct(seg3['mdd'])}"
    )
    report.append("")

    report.append("【再平衡贡献】")
    report.append(f"累计再平衡收益：{format_pct(rebalance_extra)}（相对买入并持有）")
    report.append(f"再平衡次数：{reb_count} 次")
    report.append("")

    report.append("【各资产表现】")
    for a in ["stock", "bond", "gold", "money"]:
        report.append(
            f"{ASSET_CN_NAME[a]}：年化 {format_pct(asset_ret.get(a, 0.0))}，贡献度 {format_pct(float(contrib.get(a, 0.0)))}"
        )
    report.append("")

    report.append("【对比基准】")
    report.append(
        f"单资产沪深300：年化 {format_pct(stock_sum['cagr'])}，回撤 {format_pct(stock_sum['mdd'])}"
    )
    report.append(
        f"50/50平衡组合：年化 {format_pct(b5050_sum['cagr'])}，回撤 {format_pct(b5050_sum['mdd'])}"
    )
    report.append(
        f"银行理财/定存基准（年化{DEPOSIT_BENCH_ANNUAL:.0%}）：年化 {format_pct(deposit_sum['cagr'])}，回撤 {format_pct(deposit_sum['mdd'])}"
    )

    if alt_portfolios:
        report.append("\n【其他配置方案（示例）】")
        for name, res in alt_portfolios.items():
            s = performance_summary(res.nav, res.daily_returns)
            report.append(f"{name}：年化 {format_pct(s['cagr'])}，回撤 {format_pct(s['mdd'])}，夏普 {s['sharpe']:.2f}")

    report.append(
        f"\n优胜幅度（永久组合 vs 50/50）：年化收益 +{format_pct(summary['cagr']-b5050_sum['cagr'])}，回撤改善 {format_pct(summary['mdd']-b5050_sum['mdd'])}"
    )

    report.append("\n【数据源/处理说明】")
    report.append(data_notes if data_notes else "- 无")

    report_text = "\n".join(report)

    # 保存报告
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


def plot_results(
    prices: pd.DataFrame,
    pp: BacktestResult,
    bench_stock: BacktestResult,
    bench_5050: BacktestResult,
    bench_deposit: BacktestResult,
    output_dir: str,
) -> None:
    """生成图表（PNG）。"""

    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    setup_chinese_font()

    # 1) 净值曲线对比
    plt.figure(figsize=(12, 6))
    (pp.nav / pp.nav.iloc[0]).plot(label="永久组合")
    (bench_stock.nav / bench_stock.nav.iloc[0]).plot(label="沪深300")
    (bench_5050.nav / bench_5050.nav.iloc[0]).plot(label="50/50(股债)")
    (bench_deposit.nav / bench_deposit.nav.iloc[0]).plot(label=f"定存基准({DEPOSIT_BENCH_ANNUAL:.0%})", linestyle="--")
    plt.title("净值曲线对比")
    plt.ylabel("净值（归一化）")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "01_nav_comparison.png"), dpi=150)
    plt.close()

    # 2) 子账户净值演进
    plt.figure(figsize=(12, 6))
    (pp.holdings_value.div(pp.holdings_value.iloc[0]).rename(columns=ASSET_CN_NAME)).plot()
    plt.title("各资产子账户净值演进（归一化）")
    plt.ylabel("归一化净值")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "02_sub_accounts.png"), dpi=150)
    plt.close()

    # 3) 年度收益率分布柱状图
    annual_ret = (1 + pp.daily_returns).resample("Y").prod() - 1
    plt.figure(figsize=(12, 5))
    annual_ret.index = annual_ret.index.year
    annual_ret.plot(kind="bar")
    plt.title("永久组合：年度收益率")
    plt.ylabel("年度收益率")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "03_annual_returns.png"), dpi=150)
    plt.close()

    # 4) 回撤曲线 + 事件标注
    mdd, dd = _max_drawdown(pp.nav)
    plt.figure(figsize=(12, 5))
    dd.plot(color="tab:red")
    plt.title(f"回撤曲线（最大回撤 {format_pct(mdd)}）")
    plt.ylabel("回撤")

    events = {
        "2008金融危机": "2008-09-15",
        "2015股灾": "2015-06-15",
        "2018贸易摩擦": "2018-07-06",
    }
    for name, d in events.items():
        dts = pd.Timestamp(d)
        if dd.index.min() <= dts <= dd.index.max():
            plt.axvline(dts, color="gray", linestyle="--", linewidth=1)
            plt.text(dts, 0, name, rotation=90, va="bottom", ha="right", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "04_drawdown_events.png"), dpi=150)
    plt.close()

    # 5) 资产配置占比变化
    plt.figure(figsize=(12, 6))
    pp.weights.rename(columns=ASSET_CN_NAME).plot.area(alpha=0.8)
    plt.title("资产配置占比变化（含漂移与再平衡）")
    plt.ylabel("权重")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "05_weights.png"), dpi=150)
    plt.close()

    # 6) 滚动年化收益率（1/3/5/10年）
    plt.figure(figsize=(12, 6))
    nav = pp.nav
    for years in [1, 3, 5, 10]:
        window = int(252 * years)
        rolling = nav / nav.shift(window) - 1
        rolling_ann = (1 + rolling) ** (1 / years) - 1
        rolling_ann.plot(label=f"{years}年滚动年化")
    plt.title("滚动年化收益率")
    plt.ylabel("年化收益")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "06_rolling_annualized.png"), dpi=150)
    plt.close()

    # 7) 月度收益分布直方图（含正态性检验 p 值）
    mret = (1 + pp.daily_returns).resample("M").prod() - 1
    plt.figure(figsize=(10, 5))
    plt.hist(mret.dropna(), bins=30, alpha=0.8)
    title = "月度收益分布"
    if stats is not None and len(mret.dropna()) >= 20:
        _, p = stats.normaltest(mret.dropna())
        title += f"（正态性检验 p={p:.3f}）"
    plt.title(title)
    plt.xlabel("月度收益")
    plt.ylabel("频数")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "07_monthly_return_hist.png"), dpi=150)
    plt.close()

    # 8) 相关性热力图
    asset_ret = prices.pct_change().dropna()
    corr = asset_ret.corr()
    plt.figure(figsize=(6, 5))
    if sns is not None:
        sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, fmt=".2f")
        plt.title("资产收益相关性（全样本）")
    else:
        plt.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
        plt.yticks(range(len(corr.index)), corr.index)
        plt.colorbar()
        plt.title("资产收益相关性（全样本）")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "08_correlation_heatmap.png"), dpi=150)
    plt.close()


def export_data(
    prices: pd.DataFrame,
    pp: BacktestResult,
    bench_stock: BacktestResult,
    bench_5050: BacktestResult,
    bench_deposit: BacktestResult,
    output_dir: str,
) -> None:
    """导出 CSV/Excel 数据。"""

    os.makedirs(output_dir, exist_ok=True)

    # 日度数据
    daily = pd.concat(
        [
            prices.add_prefix("price_"),
            pp.nav.rename("pp_nav"),
            pp.daily_returns.rename("pp_ret"),
            bench_stock.nav.rename("stock_nav"),
            bench_5050.nav.rename("bal5050_nav"),
            bench_deposit.nav.rename("deposit_nav"),
        ],
        axis=1,
    )
    daily.to_csv(os.path.join(output_dir, "daily_data.csv"), encoding="utf-8-sig")

    # 再平衡记录
    if not pp.rebalance_log.empty:
        pp.rebalance_log.to_csv(os.path.join(output_dir, "rebalance_log.csv"), index=False, encoding="utf-8-sig")

    # 月度/年度统计
    mret = (1 + pp.daily_returns).resample("M").prod() - 1
    yret = (1 + pp.daily_returns).resample("Y").prod() - 1

    monthly_stats = pd.DataFrame(
        {
            "monthly_return": mret,
        }
    )
    yearly_stats = pd.DataFrame({"yearly_return": yret})

    # 年度汇总表
    annual_summary = pd.DataFrame(
        {
            "pp_yearly_return": yret,
            "pp_yearly_vol(ann)": pp.daily_returns.resample("Y").std(ddof=0) * math.sqrt(252.0),
        }
    )

    # Excel（如环境无 openpyxl，则降级为 CSV）
    excel_path = os.path.join(output_dir, "summary.xlsx")
    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            annual_summary.to_excel(writer, sheet_name="annual_summary")
            monthly_stats.to_excel(writer, sheet_name="monthly")
            yearly_stats.to_excel(writer, sheet_name="yearly")
            pp.weights.to_excel(writer, sheet_name="weights")
    except Exception:
        annual_summary.to_csv(os.path.join(output_dir, "annual_summary.csv"), encoding="utf-8-sig")
        monthly_stats.to_csv(os.path.join(output_dir, "monthly_stats.csv"), encoding="utf-8-sig")
        yearly_stats.to_csv(os.path.join(output_dir, "yearly_stats.csv"), encoding="utf-8-sig")


# -----------------------------
# 基准组合构建
# -----------------------------


def build_deposit_benchmark(dates: pd.DatetimeIndex, annual_rate: float = DEPOSIT_BENCH_ANNUAL) -> BacktestResult:
    daily_r = (1 + annual_rate) ** (1 / 252.0) - 1
    nav = pd.Series((1 + daily_r) ** np.arange(len(dates)) * INITIAL_CAPITAL, index=dates, name="nav")
    ret = nav.pct_change().fillna(0.0)
    weights = pd.DataFrame(index=dates)
    holdings = pd.DataFrame(index=dates)
    log = pd.DataFrame()
    return BacktestResult(nav=nav, daily_returns=ret, weights=weights, holdings_value=holdings, rebalance_log=log)


# -----------------------------
# main
# -----------------------------


def main() -> None:
    warnings.filterwarnings("ignore")

    output_dir = os.path.join(os.path.dirname(__file__), "output")

    prices, data_notes = fetch_all_data(START_DATE, END_DATE)

    # 永久组合：默认年度再平衡
    pp = Portfolio(prices, TARGET_WEIGHTS, INITIAL_CAPITAL).run(rebalance_mode="annual")

    # 对比：单资产沪深300（100% stock，买入并持有）
    bench_stock = Portfolio(prices[["stock"]], {"stock": 1.0}, INITIAL_CAPITAL).run(rebalance_mode="none")

    # 对比：50/50（股+债）
    prices_5050 = prices[["stock", "bond"]].copy()
    bench_5050 = Portfolio(prices_5050, {"stock": 0.5, "bond": 0.5}, INITIAL_CAPITAL).run(rebalance_mode="annual")

    # 对比：银行理财/定存基准（年化3%）
    bench_deposit = build_deposit_benchmark(prices.index, annual_rate=DEPOSIT_BENCH_ANNUAL)

    # 其他配置方案（示例）
    alt_portfolios = {
        "权益增强型(40/20/20/20)": Portfolio(
            prices,
            {"stock": 0.40, "bond": 0.20, "gold": 0.20, "money": 0.20},
            INITIAL_CAPITAL,
        ).run(rebalance_mode="annual"),
        "债券稳健型(15/40/15/30)": Portfolio(
            prices,
            {"stock": 0.15, "bond": 0.40, "gold": 0.15, "money": 0.30},
            INITIAL_CAPITAL,
        ).run(rebalance_mode="annual"),
    }

    report_text = generate_report(
        prices=prices,
        pp_result=pp,
        bench_stock=bench_stock,
        bench_5050=bench_5050,
        bench_deposit=bench_deposit,
        alt_portfolios=alt_portfolios,
        data_notes=data_notes,
        output_dir=output_dir,
    )

    print(report_text)

    plot_results(prices, pp, bench_stock, bench_5050, bench_deposit, output_dir)
    export_data(prices, pp, bench_stock, bench_5050, bench_deposit, output_dir)

    # 额外：输出再平衡策略对比（定期：月/季/年）
    compare = {
        "月度": Portfolio(prices, TARGET_WEIGHTS, INITIAL_CAPITAL).run(rebalance_mode="periodic", periodic_freq="M"),
        "季度": Portfolio(prices, TARGET_WEIGHTS, INITIAL_CAPITAL).run(rebalance_mode="periodic", periodic_freq="Q"),
        "年度": pp,
        "偏离度(±10%)": Portfolio(prices, TARGET_WEIGHTS, INITIAL_CAPITAL).run(rebalance_mode="deviation", deviation_threshold=0.10),
        "事件驱动": Portfolio(prices, TARGET_WEIGHTS, INITIAL_CAPITAL).run(rebalance_mode="event"),
    }

    rows = []
    for name, res in compare.items():
        s = performance_summary(res.nav, res.daily_returns)
        rows.append(
            {
                "strategy": name,
                "cagr": s["cagr"],
                "mdd": s["mdd"],
                "sharpe": s["sharpe"],
                "rebalance_count": int(len(res.rebalance_log)),
            }
        )

    compare_df = pd.DataFrame(rows).set_index("strategy")
    compare_df.to_csv(os.path.join(output_dir, "rebalance_strategy_comparison.csv"), encoding="utf-8-sig")

    # 额外：输出“不同资产配置方案”对比
    portfolio_set: Dict[str, BacktestResult] = {
        "永久组合(25/25/25/25)": pp,
        "沪深300(单资产)": bench_stock,
        "50/50(股债)": bench_5050,
        f"定存基准({DEPOSIT_BENCH_ANNUAL:.0%})": bench_deposit,
        **alt_portfolios,
    }

    p_rows = []
    for name, res in portfolio_set.items():
        s = performance_summary(res.nav, res.daily_returns)
        p_rows.append(
            {
                "portfolio": name,
                "cagr": s["cagr"],
                "mdd": s["mdd"],
                "sharpe": s["sharpe"],
                "sortino": s["sortino"],
                "calmar": s["calmar"],
            }
        )

    pd.DataFrame(p_rows).set_index("portfolio").to_csv(
        os.path.join(output_dir, "portfolio_comparison.csv"), encoding="utf-8-sig"
    )


if __name__ == "__main__":
    main()
