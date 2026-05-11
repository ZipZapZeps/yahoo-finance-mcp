"""Yahoo Finance MCP Server — real-time and historical financial data via FastMCP."""

from __future__ import annotations

import io
import json
import sys
import textwrap
import traceback
from enum import Enum
from typing import Any, cast

import pandas as pd
import yfinance as yf
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations


class FinancialType(str, Enum):
    income_stmt = "income_stmt"
    quarterly_income_stmt = "quarterly_income_stmt"
    balance_sheet = "balance_sheet"
    quarterly_balance_sheet = "quarterly_balance_sheet"
    cashflow = "cashflow"
    quarterly_cashflow = "quarterly_cashflow"


class HolderType(str, Enum):
    major_holders = "major_holders"
    institutional_holders = "institutional_holders"
    mutualfund_holders = "mutualfund_holders"
    insider_transactions = "insider_transactions"
    insider_purchases = "insider_purchases"
    insider_roster_holders = "insider_roster_holders"


class RecommendationType(str, Enum):
    recommendations = "recommendations"
    upgrades_downgrades = "upgrades_downgrades"


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

yfinance_server = FastMCP(
    "yfinance",
    instructions="""# Yahoo Finance MCP Server

Provides real-time and historical financial data for any publicly traded ticker via Yahoo Finance.

## Available Tools
- **get_historical_stock_prices** — OHLCV price history with configurable period and interval
- **get_stock_info** — Full company profile, valuation metrics, and trading data
- **get_yahoo_finance_news** — Latest news articles for a ticker
- **get_stock_actions** — Dividend and split history
- **get_financial_statement** — Income statement, balance sheet, or cash flow (annual/quarterly)
- **get_holder_info** — Major, institutional, mutual fund holders and insider activity
- **get_option_expiration_dates** — Available options expiry dates
- **get_option_chain** — Full options chain (calls or puts) for a specific expiry
- **get_recommendations** — Analyst recommendations and broker upgrades/downgrades
- **run_python_code** — Execute Python with pandas/yfinance for custom analysis (code mode)

## Available Prompts
- **stock_analysis** — Guided comprehensive stock deep-dive
- **options_analysis** — Guided options chain analysis
- **compare_stocks** — Side-by-side fundamental comparison

## Usage Pattern
1. Start with `get_stock_info` for an overview.
2. Use `get_historical_stock_prices` or `get_financial_statement` for deeper analysis.
3. Use `run_python_code` for custom computations on the returned data.
""",
)

# Shared annotation presets
_ANN_READ_IDEMPOTENT = ToolAnnotations(
    readOnlyHint=True, idempotentHint=True, openWorldHint=True
)
_ANN_READ_LIVE = ToolAnnotations(
    readOnlyHint=True, idempotentHint=False, openWorldHint=True
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_ticker(ticker: str) -> tuple[yf.Ticker, str | None]:
    """Return (Ticker, error_message). error_message is None on success.

    Callers must guard with `if err: return err` before using the returned ticker.
    The ticker value is undefined (never None at runtime, but cast) when err is not None.
    """
    _company: yf.Ticker | None = None
    try:
        _company = yf.Ticker(ticker)
        if _company.isin is None:
            return _company, f"Company ticker '{ticker}' not found."
        return _company, None
    except Exception as exc:
        # _company is None only when construction itself raised; callers discard it on error.
        return cast(yf.Ticker, _company), f"Error resolving ticker '{ticker}': {exc}"


def _df_to_json(df: pd.DataFrame, **kwargs: Any) -> str:
    return df.to_json(orient="records", **kwargs)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@yfinance_server.tool(
    name="get_historical_stock_prices",
    title="Historical Stock Prices",
    description=(
        "Retrieve OHLCV (Open, High, Low, Close, Volume) price history for a stock ticker.\n\n"
        "Args:\n"
        "  ticker: Ticker symbol, e.g. 'AAPL'\n"
        "  period: One of 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max (default '1mo')\n"
        "  interval: One of 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo "
        "(default '1d')\n\n"
        "Returns JSON array of {Date, Open, High, Low, Close, Volume, Dividends, Stock Splits}."
    ),
    annotations=_ANN_READ_LIVE,
    meta={"tags": ["prices", "history", "ohlcv", "stock"]},
)
async def get_historical_stock_prices(
    ticker: str, period: str = "1mo", interval: str = "1d"
) -> str:
    company, err = _resolve_ticker(ticker)
    if err:
        return err
    hist = company.history(period=period, interval=interval)
    return _df_to_json(hist.reset_index(names="Date"), date_format="iso")


@yfinance_server.tool(
    name="get_stock_info",
    title="Stock Info",
    description=(
        "Fetch a comprehensive stock profile: current price, valuation multiples, "
        "company details, financial metrics, dividends, analyst coverage, and risk indicators.\n\n"
        "Args:\n"
        "  ticker: Ticker symbol, e.g. 'AAPL'\n\n"
        "Returns a flat JSON object with all available fields from Yahoo Finance."
    ),
    annotations=_ANN_READ_LIVE,
    meta={"tags": ["info", "profile", "fundamentals", "valuation", "stock"]},
)
async def get_stock_info(ticker: str) -> str:
    company, err = _resolve_ticker(ticker)
    if err:
        return err
    return json.dumps(company.info)


@yfinance_server.tool(
    name="get_yahoo_finance_news",
    title="Yahoo Finance News",
    description=(
        "Fetch the latest news articles for a ticker from Yahoo Finance.\n\n"
        "Args:\n"
        "  ticker: Ticker symbol, e.g. 'AAPL'\n\n"
        "Returns formatted news items with title, summary, description, and URL."
    ),
    annotations=_ANN_READ_LIVE,
    meta={"tags": ["news", "headlines", "sentiment", "stock"]},
)
async def get_yahoo_finance_news(ticker: str) -> str:
    company, err = _resolve_ticker(ticker)
    if err:
        return err
    try:
        raw_news = company.news
    except Exception as exc:
        return f"Error fetching news for '{ticker}': {exc}"

    items = [
        (
            f"Title: {n.get('content', {}).get('title', '')}\n"
            f"Summary: {n.get('content', {}).get('summary', '')}\n"
            f"Description: {n.get('content', {}).get('description', '')}\n"
            f"URL: {n.get('content', {}).get('canonicalUrl', {}).get('url', '')}"
        )
        for n in raw_news
        if n.get("content", {}).get("contentType") == "STORY"
    ]
    if not items:
        return f"No news found for ticker '{ticker}'."
    return "\n\n".join(items)


@yfinance_server.tool(
    name="get_stock_actions",
    title="Stock Actions (Dividends & Splits)",
    description=(
        "Retrieve the full dividend payment and stock split history for a ticker.\n\n"
        "Args:\n"
        "  ticker: Ticker symbol, e.g. 'AAPL'\n\n"
        "Returns JSON array of {Date, Dividends, Stock Splits}."
    ),
    annotations=_ANN_READ_IDEMPOTENT,
    meta={"tags": ["dividends", "splits", "corporate-actions", "stock"]},
)
async def get_stock_actions(ticker: str) -> str:
    try:
        company = yf.Ticker(ticker)
    except Exception as exc:
        return f"Error fetching actions for '{ticker}': {exc}"
    return _df_to_json(company.actions.reset_index(names="Date"), date_format="iso")


@yfinance_server.tool(
    name="get_financial_statement",
    title="Financial Statement",
    description=(
        "Retrieve an annual or quarterly financial statement for a ticker.\n\n"
        "Args:\n"
        "  ticker: Ticker symbol, e.g. 'AAPL'\n"
        "  financial_type: One of:\n"
        "    income_stmt | quarterly_income_stmt\n"
        "    balance_sheet | quarterly_balance_sheet\n"
        "    cashflow | quarterly_cashflow\n\n"
        "Returns JSON array of period objects where each key is a financial line item."
    ),
    annotations=_ANN_READ_IDEMPOTENT,
    meta={"tags": ["financials", "income", "balance-sheet", "cashflow", "fundamentals", "stock"]},
)
async def get_financial_statement(ticker: str, financial_type: str) -> str:
    company, err = _resolve_ticker(ticker)
    if err:
        return err

    match financial_type:
        case FinancialType.income_stmt:
            stmt = company.income_stmt
        case FinancialType.quarterly_income_stmt:
            stmt = company.quarterly_income_stmt
        case FinancialType.balance_sheet:
            stmt = company.balance_sheet
        case FinancialType.quarterly_balance_sheet:
            stmt = company.quarterly_balance_sheet
        case FinancialType.cashflow:
            stmt = company.cashflow
        case FinancialType.quarterly_cashflow:
            stmt = company.quarterly_cashflow
        case _:
            valid = ", ".join(t.value for t in FinancialType)
            return f"Invalid financial_type '{financial_type}'. Valid options: {valid}"

    result = [
        {
            "date": col.strftime("%Y-%m-%d") if isinstance(col, pd.Timestamp) else str(col),
            **{idx: (None if pd.isna(val) else val) for idx, val in stmt[col].items()},
        }
        for col in stmt.columns
    ]
    return json.dumps(result)


@yfinance_server.tool(
    name="get_holder_info",
    title="Holder Information",
    description=(
        "Retrieve ownership data for a ticker.\n\n"
        "Args:\n"
        "  ticker: Ticker symbol, e.g. 'AAPL'\n"
        "  holder_type: One of:\n"
        "    major_holders | institutional_holders | mutualfund_holders\n"
        "    insider_transactions | insider_purchases | insider_roster_holders\n\n"
        "Returns JSON array appropriate to the holder type selected."
    ),
    annotations=_ANN_READ_IDEMPOTENT,
    meta={"tags": ["holders", "ownership", "insiders", "institutional", "stock"]},
)
async def get_holder_info(ticker: str, holder_type: str) -> str:
    company, err = _resolve_ticker(ticker)
    if err:
        return err

    match holder_type:
        case HolderType.major_holders:
            return _df_to_json(company.major_holders.reset_index(names="metric"))
        case HolderType.institutional_holders:
            return _df_to_json(company.institutional_holders)
        case HolderType.mutualfund_holders:
            return _df_to_json(company.mutualfund_holders, date_format="iso")
        case HolderType.insider_transactions:
            return _df_to_json(company.insider_transactions, date_format="iso")
        case HolderType.insider_purchases:
            return _df_to_json(company.insider_purchases, date_format="iso")
        case HolderType.insider_roster_holders:
            return _df_to_json(company.insider_roster_holders, date_format="iso")
        case _:
            valid = ", ".join(t.value for t in HolderType)
            return f"Invalid holder_type '{holder_type}'. Valid options: {valid}"


@yfinance_server.tool(
    name="get_option_expiration_dates",
    title="Option Expiration Dates",
    description=(
        "List all available options expiration dates for a ticker.\n\n"
        "Args:\n"
        "  ticker: Ticker symbol, e.g. 'AAPL'\n\n"
        "Returns JSON array of date strings in YYYY-MM-DD format. "
        "Pass one of these dates to get_option_chain."
    ),
    annotations=_ANN_READ_LIVE,
    meta={"tags": ["options", "expiry", "derivatives", "stock"]},
)
async def get_option_expiration_dates(ticker: str) -> str:
    company, err = _resolve_ticker(ticker)
    if err:
        return err
    return json.dumps(company.options)


@yfinance_server.tool(
    name="get_option_chain",
    title="Option Chain",
    description=(
        "Fetch the full options chain for a specific ticker, expiration date, and side.\n\n"
        "Args:\n"
        "  ticker: Ticker symbol, e.g. 'AAPL'\n"
        "  expiration_date: A date from get_option_expiration_dates (format YYYY-MM-DD)\n"
        "  option_type: 'calls' or 'puts'\n\n"
        "Returns JSON array of option contracts with strike, bid, ask, IV, OI, greeks, etc."
    ),
    annotations=_ANN_READ_LIVE,
    meta={"tags": ["options", "calls", "puts", "chain", "derivatives", "greeks", "stock"]},
)
async def get_option_chain(ticker: str, expiration_date: str, option_type: str) -> str:
    company, err = _resolve_ticker(ticker)
    if err:
        return err
    if expiration_date not in company.options:
        return (
            f"No options available for '{expiration_date}'. "
            "Use get_option_expiration_dates to see valid dates."
        )
    if option_type not in ("calls", "puts"):
        return "Invalid option_type. Use 'calls' or 'puts'."
    chain = company.option_chain(expiration_date)
    df = chain.calls if option_type == "calls" else chain.puts
    return _df_to_json(df, date_format="iso")


@yfinance_server.tool(
    name="get_recommendations",
    title="Analyst Recommendations",
    description=(
        "Get analyst recommendations or broker upgrades/downgrades for a ticker.\n\n"
        "Args:\n"
        "  ticker: Ticker symbol, e.g. 'AAPL'\n"
        "  recommendation_type: 'recommendations' or 'upgrades_downgrades'\n"
        "  months_back: For upgrades_downgrades — how many months of history (default 12)\n\n"
        "Returns JSON array sorted by date descending, "
        "deduplicated to the most recent rating per firm."
    ),
    annotations=_ANN_READ_IDEMPOTENT,
    meta={"tags": ["analysts", "recommendations", "ratings", "upgrades", "stock"]},
)
async def get_recommendations(
    ticker: str, recommendation_type: str, months_back: int = 12
) -> str:
    company, err = _resolve_ticker(ticker)
    if err:
        return err
    try:
        match recommendation_type:
            case RecommendationType.recommendations:
                return _df_to_json(company.recommendations)
            case RecommendationType.upgrades_downgrades:
                df = company.upgrades_downgrades.reset_index()
                cutoff = pd.Timestamp.now() - pd.DateOffset(months=months_back)
                df = df[df["GradeDate"] >= cutoff].sort_values("GradeDate", ascending=False)
                return _df_to_json(df.drop_duplicates(subset=["Firm"]), date_format="iso")
            case _:
                valid = ", ".join(t.value for t in RecommendationType)
                return f"Invalid recommendation_type '{recommendation_type}'. Valid: {valid}"
    except Exception as exc:
        return f"Error fetching recommendations for '{ticker}': {exc}"


# ---------------------------------------------------------------------------
# Code mode — Python REPL tool
# ---------------------------------------------------------------------------

_CODE_GLOBALS: dict[str, Any] = {"pd": pd, "yf": yf, "json": json, "sys": sys}


@yfinance_server.tool(
    name="run_python_code",
    title="Run Python Code (Code Mode)",
    description=(
        "Execute a Python snippet in a sandboxed namespace with pandas and yfinance available.\n\n"
        "Pre-imported globals: pd (pandas), yf (yfinance), json, sys\n\n"
        "Args:\n"
        "  code: Python source to execute. Use print() to return results.\n\n"
        "Returns combined stdout and stderr. Ideal for custom calculations, "
        "data transformations, or exploratory analysis on financial data.\n\n"
        "Example:\n"
        "  ticker = yf.Ticker('AAPL')\n"
        "  hist = ticker.history(period='1mo')\n"
        "  print(hist['Close'].describe().to_json())"
    ),
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
    meta={"tags": ["code", "repl", "python", "analysis", "pandas"]},
)
async def run_python_code(code: str) -> str:
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    local_ns: dict[str, Any] = {}
    try:
        compiled = compile(textwrap.dedent(code), "<mcp-code>", "exec")
        with io.StringIO() as _out, io.StringIO() as _err:
            import contextlib

            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                exec(compiled, {**_CODE_GLOBALS}, local_ns)  # noqa: S102
    except Exception:
        stderr_buf.write(traceback.format_exc())

    parts: list[str] = []
    if out := stdout_buf.getvalue():
        parts.append(out)
    if err := stderr_buf.getvalue():
        parts.append(f"[stderr]\n{err}")
    return "\n".join(parts) if parts else "(no output)"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


@yfinance_server.prompt(
    name="stock_analysis",
    title="Comprehensive Stock Analysis",
    description="Guide a thorough analysis of a single stock using all available tools.",
)
def stock_analysis_prompt(ticker: str) -> str:
    t = ticker.upper()
    return textwrap.dedent(f"""
        Please perform a comprehensive analysis of **{t}** using the available tools.

        Steps:
        1. `get_stock_info('{t}')` — company overview and key metrics
        2. `get_historical_stock_prices('{t}', period='1y')` — 1-year price history
        3. `get_financial_statement('{t}', 'income_stmt')` — annual revenue and profit trends
        4. `get_financial_statement('{t}', 'balance_sheet')` — debt and equity position
        5. `get_holder_info('{t}', 'institutional_holders')` — institutional ownership
        6. `get_recommendations('{t}', 'upgrades_downgrades')` — recent analyst sentiment
        7. `get_yahoo_finance_news('{t}')` — latest news

        After gathering data, provide:
        - **Summary**: business model, market cap, sector
        - **Valuation**: P/E, P/B, EV/EBITDA vs. sector norms
        - **Financial Health**: revenue growth, margins, debt/equity ratio
        - **Momentum**: 52-week price trend, volume patterns
        - **Sentiment**: analyst ratings, insider activity, news tone
        - **Verdict**: Buy / Hold / Avoid with clear rationale
    """).strip()


@yfinance_server.prompt(
    name="options_analysis",
    title="Options Chain Analysis",
    description="Guide an options market analysis including IV skew, max pain, and positioning.",
)
def options_analysis_prompt(ticker: str) -> str:
    t = ticker.upper()
    return textwrap.dedent(f"""
        Perform an options market analysis for **{t}**.

        Steps:
        1. `get_option_expiration_dates('{t}')` — list available expirations
        2. Pick the nearest-term and one ~30-day expiration date
        3. For each date call both:
           - `get_option_chain('{t}', <date>, 'calls')`
           - `get_option_chain('{t}', <date>, 'puts')`
        4. `get_stock_info('{t}')` — get current price and implied volatility

        Analyze and report:
        - **Put/Call ratio** by open interest at each expiry
        - **IV skew** across strikes (calls vs puts, OTM vs ITM)
        - **Max pain** strike (where total option value to holders is minimized)
        - **Notable positioning**: unusually high OI or volume at specific strikes
        - **Interpretation**: what the options market implies about expected price range
    """).strip()


@yfinance_server.prompt(
    name="compare_stocks",
    title="Compare Stocks",
    description="Guide a side-by-side fundamental comparison of two or more comma-separated tickers.",
)
def compare_stocks_prompt(tickers: str) -> str:
    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    data_steps = "\n".join(
        f"   - `get_stock_info('{t}')`, "
        f"`get_financial_statement('{t}', 'income_stmt')`, "
        f"`get_recommendations('{t}', 'upgrades_downgrades')`"
        for t in ticker_list
    )
    col_header = " | ".join(ticker_list)
    sep = " | ".join("-" * max(len(t), 8) for t in ticker_list)
    return textwrap.dedent(f"""
        Compare the following stocks side-by-side: **{', '.join(ticker_list)}**

        For each ticker gather:
{data_steps}

        Produce a comparison table:

        | Metric | {col_header} |
        |--------|{sep}|
        | Market Cap | ... |
        | P/E Ratio | ... |
        | Revenue Growth (YoY) | ... |
        | Net Margin | ... |
        | Debt/Equity | ... |
        | Analyst Consensus | ... |

        Conclude with a recommendation on which offers better value/growth for a long-term investor.
    """).strip()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print("Starting Yahoo Finance MCP server...")
    yfinance_server.run(transport="stdio")


if __name__ == "__main__":
    main()
