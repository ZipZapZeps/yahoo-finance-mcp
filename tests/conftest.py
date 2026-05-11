"""Shared fixtures for Yahoo Finance MCP server tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# DataFrame builders
# ---------------------------------------------------------------------------


def make_price_df() -> pd.DataFrame:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    return pd.DataFrame(
        {
            "Open": [185.0, 186.5],
            "High": [187.0, 188.0],
            "Low": [184.0, 185.5],
            "Close": [186.0, 187.5],
            "Volume": [50_000_000, 55_000_000],
            "Dividends": [0.0, 0.0],
            "Stock Splits": [0.0, 0.0],
        },
        index=pd.DatetimeIndex(idx, name="Date"),
    )


def make_actions_df() -> pd.DataFrame:
    idx = pd.to_datetime(["2023-08-10", "2023-11-09"])
    return pd.DataFrame(
        {"Dividends": [0.24, 0.24], "Stock Splits": [0.0, 0.0]},
        index=pd.DatetimeIndex(idx, name="Date"),
    )


def make_financial_stmt() -> pd.DataFrame:
    cols = pd.to_datetime(["2023-09-30", "2022-09-24"])
    return pd.DataFrame(
        {
            cols[0]: {"Total Revenue": 383_285_000_000.0, "Net Income": 96_995_000_000.0},
            cols[1]: {"Total Revenue": 394_328_000_000.0, "Net Income": 99_803_000_000.0},
        }
    )


def make_recommendations_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "period": ["0m", "-1m", "-2m", "-3m"],
            "strongBuy": [12, 11, 10, 9],
            "buy": [20, 19, 18, 17],
            "hold": [8, 9, 10, 11],
            "sell": [2, 2, 3, 3],
            "strongSell": [1, 1, 1, 2],
        }
    )


def make_upgrades_df() -> pd.DataFrame:
    idx = pd.to_datetime(["2024-06-01", "2024-05-15", "2024-04-10"])
    return pd.DataFrame(
        {
            "Firm": ["Goldman Sachs", "Morgan Stanley", "Goldman Sachs"],
            "ToGrade": ["Buy", "Overweight", "Neutral"],
            "FromGrade": ["Neutral", "Equal-Weight", "Buy"],
            "Action": ["upgrade", "upgrade", "downgrade"],
        },
        index=pd.DatetimeIndex(idx, name="GradeDate"),
    )


def make_holders_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Holder": ["Vanguard Group", "BlackRock", "Berkshire Hathaway"],
            "Shares": [1_254_000_000, 1_020_000_000, 915_560_000],
            "pctHeld": [0.0791, 0.0643, 0.0577],
        }
    )


def make_major_holders_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Value": ["7.91%", "64.30%", "57.72%", "4,845"],
            "Breakdown": [
                "% of Shares Held by All Insider",
                "% of Shares Held by Institutions",
                "% of Float Held by Institutions",
                "Number of Institutions Holding Shares",
            ],
        }
    )


def make_options_chain_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "contractSymbol": ["AAPL240119C00150000", "AAPL240119C00160000"],
            "strike": [150.0, 160.0],
            "lastPrice": [38.5, 29.1],
            "bid": [38.4, 29.0],
            "ask": [38.6, 29.2],
            "volume": [100, 200],
            "openInterest": [5000, 8000],
            "impliedVolatility": [0.35, 0.30],
            "inTheMoney": [True, True],
        }
    )


# ---------------------------------------------------------------------------
# Ticker mock factory
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_ticker(mocker: pytest.MonkeyPatch) -> MagicMock:
    """Patch yfinance.Ticker and return a pre-configured mock instance."""
    instance = MagicMock()
    instance.isin = "US0378331005"
    instance.info = {
        "symbol": "AAPL",
        "longName": "Apple Inc.",
        "currentPrice": 187.5,
        "marketCap": 2_900_000_000_000,
        "trailingPE": 29.5,
        "dividendYield": 0.005,
        "sector": "Technology",
    }
    instance.history.return_value = make_price_df()
    instance.actions = make_actions_df()
    instance.income_stmt = make_financial_stmt()
    instance.quarterly_income_stmt = make_financial_stmt()
    instance.balance_sheet = make_financial_stmt()
    instance.quarterly_balance_sheet = make_financial_stmt()
    instance.cashflow = make_financial_stmt()
    instance.quarterly_cashflow = make_financial_stmt()
    instance.major_holders = make_major_holders_df()
    instance.institutional_holders = make_holders_df()
    instance.mutualfund_holders = make_holders_df()
    instance.insider_transactions = make_holders_df()
    instance.insider_purchases = make_holders_df()
    instance.insider_roster_holders = make_holders_df()
    instance.options = ("2024-01-19", "2024-02-16", "2024-03-15")
    chain_mock = MagicMock()
    chain_mock.calls = make_options_chain_df()
    chain_mock.puts = make_options_chain_df()
    instance.option_chain.return_value = chain_mock
    instance.recommendations = make_recommendations_df()
    instance.upgrades_downgrades = make_upgrades_df()
    instance.news = [
        {
            "content": {
                "contentType": "STORY",
                "title": "Apple Reports Record Q4 Earnings",
                "summary": "Revenue beat expectations",
                "description": "Detailed description here",
                "canonicalUrl": {"url": "https://finance.yahoo.com/news/apple-q4"},
            }
        },
        {
            "content": {
                "contentType": "VIDEO",  # should be filtered out
                "title": "Video content",
            }
        },
    ]
    mocker.patch("yfinance.Ticker", return_value=instance)
    return instance
