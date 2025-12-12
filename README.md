# 中国永久投资组合回测分析系统

本仓库提供一个**单文件可运行**的 Python 回测脚本：`permanent_portfolio_backtest.py`。

## 功能

- 中国版永久投资组合（四等分：沪深300/长期国债/黄金/货币基金）
- 多种再平衡策略（年度/定期（月季年）/偏离度阈值/事件驱动）
- 指标：年化收益、最大回撤、夏普、索蒂诺、卡尔玛、月度胜率等
- 分段分析（2005-2010、2011-2018、2021-2024）
- 输出：
  - 终端报告 + `output/report.txt`
  - 图表 PNG：`output/plots/*.png`
  - 数据导出：`output/daily_data.csv`、`output/summary.xlsx`（若环境无 openpyxl 则降级为 CSV）

## 运行

```bash
python permanent_portfolio_backtest.py
```

## 数据说明

脚本会优先尝试通过 `yfinance` 拉取中国 ETF/指数数据；当网络不可用、依赖未安装或标的历史不足时，会自动使用**可复现的合成数据**补齐 2000-2024 的区间，以保证回测与分析流程完整可运行。
