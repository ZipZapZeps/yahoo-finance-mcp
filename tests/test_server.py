"""Pytest suite for Yahoo Finance MCP server."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

import server
from server import (
    FinancialType,
    HolderType,
    RecommendationType,
    compare_stocks_prompt,
    get_financial_statement,
    get_historical_stock_prices,
    get_holder_info,
    get_option_chain,
    get_option_expiration_dates,
    get_recommendations,
    get_stock_actions,
    get_stock_info,
    get_yahoo_finance_news,
    options_analysis_prompt,
    run_python_code,
    stock_analysis_prompt,
)


# ===========================================================================
# _resolve_ticker
# ===========================================================================


class TestResolveTickerHelper:
    def test_success(self, mocker: pytest.MonkeyPatch) -> None:
        inst = MagicMock()
        inst.isin = "US0378331005"
        mocker.patch("yfinance.Ticker", return_value=inst)
        company, err = server._resolve_ticker("AAPL")
        assert err is None
        assert company is inst

    def test_isin_none_returns_error(self, mocker: pytest.MonkeyPatch) -> None:
        inst = MagicMock()
        inst.isin = None
        mocker.patch("yfinance.Ticker", return_value=inst)
        _, err = server._resolve_ticker("INVALID")
        assert err is not None
        assert "not found" in err

    def test_isin_access_exception_returns_error(self, mocker: pytest.MonkeyPatch) -> None:
        inst = MagicMock()
        type(inst).isin = property(lambda self: (_ for _ in ()).throw(Exception("network error")))
        mocker.patch("yfinance.Ticker", return_value=inst)
        _, err = server._resolve_ticker("AAPL")
        assert err is not None
        assert "Error resolving" in err


# ===========================================================================
# get_historical_stock_prices
# ===========================================================================


class TestGetHistoricalStockPrices:
    @pytest.mark.asyncio
    async def test_returns_json_array(self, mock_ticker: MagicMock) -> None:
        result = await get_historical_stock_prices("AAPL")
        records = json.loads(result)
        assert isinstance(records, list)
        assert len(records) == 2
        assert "Close" in records[0]
        assert "Open" in records[0]

    @pytest.mark.asyncio
    async def test_passes_period_and_interval(self, mock_ticker: MagicMock) -> None:
        await get_historical_stock_prices("AAPL", period="1y", interval="1wk")
        mock_ticker.history.assert_called_once_with(period="1y", interval="1wk")

    @pytest.mark.asyncio
    async def test_invalid_ticker_returns_error(self, mocker: pytest.MonkeyPatch) -> None:
        inst = MagicMock()
        inst.isin = None
        mocker.patch("yfinance.Ticker", return_value=inst)
        result = await get_historical_stock_prices("FAKE")
        assert "not found" in result


# ===========================================================================
# get_stock_info
# ===========================================================================


class TestGetStockInfo:
    @pytest.mark.asyncio
    async def test_returns_json_object(self, mock_ticker: MagicMock) -> None:
        result = await get_stock_info("AAPL")
        data = json.loads(result)
        assert data["symbol"] == "AAPL"
        assert data["sector"] == "Technology"

    @pytest.mark.asyncio
    async def test_invalid_ticker_returns_error(self, mocker: pytest.MonkeyPatch) -> None:
        inst = MagicMock()
        inst.isin = None
        mocker.patch("yfinance.Ticker", return_value=inst)
        result = await get_stock_info("XXXX")
        assert "not found" in result


# ===========================================================================
# get_yahoo_finance_news
# ===========================================================================


class TestGetYahooFinanceNews:
    @pytest.mark.asyncio
    async def test_returns_story_articles_only(self, mock_ticker: MagicMock) -> None:
        result = await get_yahoo_finance_news("AAPL")
        assert "Apple Reports Record Q4 Earnings" in result
        assert "Video content" not in result

    @pytest.mark.asyncio
    async def test_formats_article_fields(self, mock_ticker: MagicMock) -> None:
        result = await get_yahoo_finance_news("AAPL")
        assert "Title:" in result
        assert "Summary:" in result
        assert "URL:" in result
        assert "finance.yahoo.com" in result

    @pytest.mark.asyncio
    async def test_no_stories_returns_not_found(self, mock_ticker: MagicMock) -> None:
        mock_ticker.news = [{"content": {"contentType": "VIDEO", "title": "vid"}}]
        result = await get_yahoo_finance_news("AAPL")
        assert "No news found" in result

    @pytest.mark.asyncio
    async def test_news_fetch_exception(self, mock_ticker: MagicMock) -> None:
        type(mock_ticker).news = property(lambda self: (_ for _ in ()).throw(Exception("timeout")))
        result = await get_yahoo_finance_news("AAPL")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_invalid_ticker_returns_error(self, mocker: pytest.MonkeyPatch) -> None:
        inst = MagicMock()
        inst.isin = None
        mocker.patch("yfinance.Ticker", return_value=inst)
        result = await get_yahoo_finance_news("FAKE")
        assert "not found" in result


# ===========================================================================
# get_stock_actions
# ===========================================================================


class TestGetStockActions:
    @pytest.mark.asyncio
    async def test_returns_json_array(self, mock_ticker: MagicMock) -> None:
        result = await get_stock_actions("AAPL")
        records = json.loads(result)
        assert isinstance(records, list)
        assert "Dividends" in records[0]

    @pytest.mark.asyncio
    async def test_ticker_exception_returns_error(self, mocker: pytest.MonkeyPatch) -> None:
        mocker.patch("yfinance.Ticker", side_effect=Exception("bad ticker"))
        result = await get_stock_actions("AAPL")
        assert "Error" in result


# ===========================================================================
# get_financial_statement
# ===========================================================================


class TestGetFinancialStatement:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "fin_type,attr",
        [
            (FinancialType.income_stmt, "income_stmt"),
            (FinancialType.quarterly_income_stmt, "quarterly_income_stmt"),
            (FinancialType.balance_sheet, "balance_sheet"),
            (FinancialType.quarterly_balance_sheet, "quarterly_balance_sheet"),
            (FinancialType.cashflow, "cashflow"),
            (FinancialType.quarterly_cashflow, "quarterly_cashflow"),
        ],
    )
    async def test_valid_types_return_json(
        self, mock_ticker: MagicMock, fin_type: FinancialType, attr: str
    ) -> None:
        result = await get_financial_statement("AAPL", fin_type)
        records = json.loads(result)
        assert isinstance(records, list)
        assert "date" in records[0]

    @pytest.mark.asyncio
    async def test_invalid_type_returns_error(self, mock_ticker: MagicMock) -> None:
        result = await get_financial_statement("AAPL", "unknown_stmt")
        assert "Invalid financial_type" in result
        assert "income_stmt" in result

    @pytest.mark.asyncio
    async def test_nan_values_serialized_as_null(self, mock_ticker: MagicMock) -> None:
        import math

        cols = pd.to_datetime(["2023-09-30"])
        df = pd.DataFrame({cols[0]: {"Revenue": float("nan"), "NetIncome": 1_000.0}})
        mock_ticker.income_stmt = df
        result = await get_financial_statement("AAPL", FinancialType.income_stmt)
        records = json.loads(result)
        assert records[0]["Revenue"] is None
        assert records[0]["NetIncome"] == 1_000.0

    @pytest.mark.asyncio
    async def test_invalid_ticker_returns_error(self, mocker: pytest.MonkeyPatch) -> None:
        inst = MagicMock()
        inst.isin = None
        mocker.patch("yfinance.Ticker", return_value=inst)
        result = await get_financial_statement("FAKE", FinancialType.income_stmt)
        assert "not found" in result


# ===========================================================================
# get_holder_info
# ===========================================================================


class TestGetHolderInfo:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "holder_type",
        [
            HolderType.institutional_holders,
            HolderType.mutualfund_holders,
            HolderType.insider_transactions,
            HolderType.insider_purchases,
            HolderType.insider_roster_holders,
        ],
    )
    async def test_valid_types_return_json(
        self, mock_ticker: MagicMock, holder_type: HolderType
    ) -> None:
        result = await get_holder_info("AAPL", holder_type)
        records = json.loads(result)
        assert isinstance(records, list)

    @pytest.mark.asyncio
    async def test_major_holders_adds_metric_column(self, mock_ticker: MagicMock) -> None:
        result = await get_holder_info("AAPL", HolderType.major_holders)
        records = json.loads(result)
        assert isinstance(records, list)

    @pytest.mark.asyncio
    async def test_invalid_type_returns_error(self, mock_ticker: MagicMock) -> None:
        result = await get_holder_info("AAPL", "unknown_holder")
        assert "Invalid holder_type" in result

    @pytest.mark.asyncio
    async def test_invalid_ticker_returns_error(self, mocker: pytest.MonkeyPatch) -> None:
        inst = MagicMock()
        inst.isin = None
        mocker.patch("yfinance.Ticker", return_value=inst)
        result = await get_holder_info("FAKE", HolderType.institutional_holders)
        assert "not found" in result


# ===========================================================================
# get_option_expiration_dates
# ===========================================================================


class TestGetOptionExpirationDates:
    @pytest.mark.asyncio
    async def test_returns_json_array_of_dates(self, mock_ticker: MagicMock) -> None:
        result = await get_option_expiration_dates("AAPL")
        dates = json.loads(result)
        assert isinstance(dates, list)
        assert "2024-01-19" in dates

    @pytest.mark.asyncio
    async def test_invalid_ticker_returns_error(self, mocker: pytest.MonkeyPatch) -> None:
        inst = MagicMock()
        inst.isin = None
        mocker.patch("yfinance.Ticker", return_value=inst)
        result = await get_option_expiration_dates("FAKE")
        assert "not found" in result


# ===========================================================================
# get_option_chain
# ===========================================================================


class TestGetOptionChain:
    @pytest.mark.asyncio
    async def test_calls_returns_json(self, mock_ticker: MagicMock) -> None:
        result = await get_option_chain("AAPL", "2024-01-19", "calls")
        records = json.loads(result)
        assert isinstance(records, list)
        assert "strike" in records[0]

    @pytest.mark.asyncio
    async def test_puts_returns_json(self, mock_ticker: MagicMock) -> None:
        result = await get_option_chain("AAPL", "2024-01-19", "puts")
        records = json.loads(result)
        assert isinstance(records, list)

    @pytest.mark.asyncio
    async def test_invalid_date_returns_error(self, mock_ticker: MagicMock) -> None:
        result = await get_option_chain("AAPL", "1999-01-01", "calls")
        assert "No options available" in result
        assert "get_option_expiration_dates" in result

    @pytest.mark.asyncio
    async def test_invalid_option_type_returns_error(self, mock_ticker: MagicMock) -> None:
        result = await get_option_chain("AAPL", "2024-01-19", "straddles")
        assert "Invalid option_type" in result

    @pytest.mark.asyncio
    async def test_invalid_ticker_returns_error(self, mocker: pytest.MonkeyPatch) -> None:
        inst = MagicMock()
        inst.isin = None
        mocker.patch("yfinance.Ticker", return_value=inst)
        result = await get_option_chain("FAKE", "2024-01-19", "calls")
        assert "not found" in result


# ===========================================================================
# get_recommendations
# ===========================================================================


class TestGetRecommendations:
    @pytest.mark.asyncio
    async def test_recommendations_returns_json(self, mock_ticker: MagicMock) -> None:
        result = await get_recommendations("AAPL", RecommendationType.recommendations)
        records = json.loads(result)
        assert isinstance(records, list)
        assert "strongBuy" in records[0]

    @pytest.mark.asyncio
    async def test_upgrades_downgrades_filters_by_months(
        self, mock_ticker: MagicMock, mocker: pytest.MonkeyPatch
    ) -> None:
        mocker.patch(
            "pandas.Timestamp.now",
            return_value=pd.Timestamp("2024-07-01"),
        )
        result = await get_recommendations(
            "AAPL", RecommendationType.upgrades_downgrades, months_back=12
        )
        records = json.loads(result)
        assert isinstance(records, list)

    @pytest.mark.asyncio
    async def test_upgrades_deduplicates_per_firm(self, mock_ticker: MagicMock) -> None:
        result = await get_recommendations(
            "AAPL", RecommendationType.upgrades_downgrades, months_back=24
        )
        records = json.loads(result)
        firms = [r["Firm"] for r in records]
        assert len(firms) == len(set(firms)), "duplicate firms found"

    @pytest.mark.asyncio
    async def test_invalid_type_returns_error(self, mock_ticker: MagicMock) -> None:
        result = await get_recommendations("AAPL", "bad_type")
        assert "Invalid recommendation_type" in result

    @pytest.mark.asyncio
    async def test_invalid_ticker_returns_error(self, mocker: pytest.MonkeyPatch) -> None:
        inst = MagicMock()
        inst.isin = None
        mocker.patch("yfinance.Ticker", return_value=inst)
        result = await get_recommendations("FAKE", RecommendationType.recommendations)
        assert "not found" in result


# ===========================================================================
# run_python_code
# ===========================================================================


class TestRunPythonCode:
    @pytest.mark.asyncio
    async def test_basic_print(self) -> None:
        result = await run_python_code("print('hello world')")
        assert "hello world" in result

    @pytest.mark.asyncio
    async def test_pandas_available(self) -> None:
        code = "import pandas as pd; print(type(pd.DataFrame()).__name__)"
        result = await run_python_code(code)
        assert "DataFrame" in result

    @pytest.mark.asyncio
    async def test_pd_global_available(self) -> None:
        result = await run_python_code("print(pd.__version__)")
        assert result.strip() != ""
        assert "Error" not in result

    @pytest.mark.asyncio
    async def test_multiline_code(self) -> None:
        code = "x = 2 + 2\nprint(f'result={x}')"
        result = await run_python_code(code)
        assert "result=4" in result

    @pytest.mark.asyncio
    async def test_exception_captured_in_stderr(self) -> None:
        result = await run_python_code("raise ValueError('test error')")
        assert "[stderr]" in result
        assert "ValueError" in result
        assert "test error" in result

    @pytest.mark.asyncio
    async def test_syntax_error_captured(self) -> None:
        result = await run_python_code("def broken(:\n    pass")
        assert "[stderr]" in result

    @pytest.mark.asyncio
    async def test_no_output_returns_sentinel(self) -> None:
        result = await run_python_code("x = 1 + 1")
        assert result == "(no output)"

    @pytest.mark.asyncio
    async def test_indented_code_dedented(self) -> None:
        code = """
            x = 10
            print(x * 2)
        """
        result = await run_python_code(code)
        assert "20" in result

    @pytest.mark.asyncio
    async def test_json_global_available(self) -> None:
        result = await run_python_code("print(json.dumps({'a': 1}))")
        data = json.loads(result.strip())
        assert data == {"a": 1}


# ===========================================================================
# Prompts
# ===========================================================================


class TestPrompts:
    def test_stock_analysis_contains_ticker(self) -> None:
        result = stock_analysis_prompt("aapl")
        assert "AAPL" in result
        assert "get_stock_info" in result
        assert "get_historical_stock_prices" in result
        assert "get_financial_statement" in result
        assert "get_recommendations" in result

    def test_stock_analysis_uppercase_ticker(self) -> None:
        result = stock_analysis_prompt("msft")
        assert "MSFT" in result

    def test_options_analysis_contains_ticker(self) -> None:
        result = options_analysis_prompt("aapl")
        assert "AAPL" in result
        assert "get_option_expiration_dates" in result
        assert "get_option_chain" in result

    def test_compare_stocks_single_ticker(self) -> None:
        result = compare_stocks_prompt("AAPL")
        assert "AAPL" in result
        assert "get_stock_info" in result

    def test_compare_stocks_multiple_tickers(self) -> None:
        result = compare_stocks_prompt("AAPL, MSFT, GOOGL")
        assert "AAPL" in result
        assert "MSFT" in result
        assert "GOOGL" in result

    def test_compare_stocks_comma_separated_with_spaces(self) -> None:
        result = compare_stocks_prompt(" aapl , msft ")
        assert "AAPL" in result
        assert "MSFT" in result


# ===========================================================================
# Enum values
# ===========================================================================


class TestEnums:
    def test_financial_type_values(self) -> None:
        assert FinancialType.income_stmt == "income_stmt"
        assert FinancialType.quarterly_cashflow == "quarterly_cashflow"

    def test_holder_type_values(self) -> None:
        assert HolderType.major_holders == "major_holders"
        assert HolderType.insider_roster_holders == "insider_roster_holders"

    def test_recommendation_type_values(self) -> None:
        assert RecommendationType.recommendations == "recommendations"
        assert RecommendationType.upgrades_downgrades == "upgrades_downgrades"

    def test_all_financial_types_covered(self) -> None:
        assert len(FinancialType) == 6

    def test_all_holder_types_covered(self) -> None:
        assert len(HolderType) == 6

    def test_all_recommendation_types_covered(self) -> None:
        assert len(RecommendationType) == 2
