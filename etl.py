#!/usr/bin/env python3
"""
ETL: Notion Database  →  PostgreSQL (AWS RDS)

- Pulls all rows from a Notion database (handles pagination)
- Normalizes messy fields (budgets, clicks, dates, channel names, etc.)
- Loads into a PostgreSQL table on RDS

Environment variables (set via GitHub Secrets / .env):
  NOTION_TOKEN            # "secret_..." or "ntn_..." token
  NOTION_DATABASE_ID      # the Notion database (UUID-like) id
  NOTION_VERSION          # optional, default "2022-06-28"

  PG_HOST                 # your RDS endpoint, e.g. mydb.xxxxx.eu-west-3.rds.amazonaws.com
  PG_PORT                 # "5432"
  PG_DB                   # e.g. "finance" or "postgres"
  PG_USER                 # RDS master or app user
  PG_PASSWORD             # password
  PG_TABLE                # destination table name, default "campaigns"
  PG_IF_EXISTS            # replace | append | fail (default replace)
  PG_CHUNKSIZE            # e.g. "1000" (default 1000)

  LOG_LEVEL               # INFO (default), DEBUG, etc.
"""

from __future__ import annotations

import os
import re
import json
import math
import logging
from typing import Any, Dict, List, Optional

import requests
import pandas as pd
from dateutil import parser as dateparser
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

# ───────────────────────── Logging ─────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("etl_notion_to_rds")


# ──────────────────────── Config helpers ───────────────────
def env(key: str, default: Optional[str] = None) -> str:
    """Env with default; empty string counts as missing -> use default."""
    val = os.getenv(key)
    if (val is None or val == "") and default is not None:
        return default
    if val is None or val == "":
        raise RuntimeError(f"Missing required environment variable: {key}")
    return val


def build_pg_url(user: str, password: str, host: str, port: str, db: str) -> str:
    """
    Use SSL with RDS by default.
    """
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}?sslmode=require"


# ─────────────────────── Notion extraction ─────────────────
def notion_headers(token: str, version: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Notion-Version": version,
        "Content-Type": "application/json",
    }


def fetch_notion_pages(database_id: str, token: str, version: str = "2022-06-28") -> List[Dict[str, Any]]:
    """
    Fetch *all* pages from a Notion database via /query with cursor pagination.
    """
    url = f"https://api.notion.com/v1/databases/{database_id}/query"
    headers = notion_headers(token, version)
    payload: Dict[str, Any] = {}
    pages: List[Dict[str, Any]] = []

    while True:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if not resp.ok:
            raise RuntimeError(f"Notion API error {resp.status_code}: {resp.text}")
        data = resp.json()
        pages.extend(data.get("results", []))
        if data.get("has_more"):
            payload = {"start_cursor": data["next_cursor"]}
        else:
            break

    logger.info("Fetched %d Notion pages.", len(pages))
    return pages


def extract_property_value(prop: Dict[str, Any]) -> Any:
    """
    Generic extractor for common Notion property types -> simple Python scalars/strings.
    Extend as needed for your schema.
    """
    t = prop.get("type")
    if t == "title":
        arr = prop.get("title", [])
        return "".join([c.get("plain_text", "") for c in arr]) if arr else ""
    if t == "rich_text":
        arr = prop.get("rich_text", [])
        return "".join([c.get("plain_text", "") for c in arr]) if arr else ""
    if t == "date":
        d = prop.get("date") or {}
        return d.get("start") or ""
    if t == "select":
        s = prop.get("select") or {}
        return s.get("name") or ""
    if t == "status":
        s = prop.get("status") or {}
        return s.get("name") or ""
    if t == "multi_select":
        arr = prop.get("multi_select", [])
        return ", ".join([x.get("name", "") for x in arr])
    if t == "number":
        return prop.get("number")
    if t == "checkbox":
        return bool(prop.get("checkbox"))
    if t == "url":
        return prop.get("url") or ""
    if t == "people":
        arr = prop.get("people", [])
        return ", ".join([p.get("name") or p.get("id", "") for p in arr])
    # fallback to stringified dict
    return str(prop.get(t))


def pages_to_dataframe(pages: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for it in pages:
        props = it.get("properties", {})
        row = {k: extract_property_value(v) for k, v in props.items()}
        rows.append(row)
    df = pd.DataFrame(rows)
    logger.info("Constructed DataFrame with shape: %s", df.shape)
    return df


# ─────────────────────── Transform helpers ─────────────────
def parse_budget(cell) -> int:
    """
    Normalize messy budget values to integer euros.
    Handles: '€1,100', '1000 EUR', '3k', '2.5k', '4500', '€0', None, ''.
    Missing/invalid -> 0.
    """
    if cell is None:
        return 0
    if isinstance(cell, float) and math.isnan(cell):
        return 0
    s = str(cell).strip().lower()
    if s in {"", "na", "n/a", "none"}:
        return 0
    m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*k\s*$", s)
    if m:
        return int(round(float(m.group(1)) * 1000))
    s = (
        s.replace("€", "")
         .replace("eur", "")
         .replace(",", "")
         .replace(" ", "")
    )
    s = re.sub(r"[^0-9.]", "", s)
    if not s or s == ".":
        return 0
    try:
        return int(round(float(s)))
    except ValueError:
        return 0


def parse_clicks(cell) -> int:
    """Normalize messy click values to an int. Missing/invalid -> 0."""
    if cell is None:
        return 0
    if isinstance(cell, float) and math.isnan(cell):
        return 0
    s = str(cell).strip().lower()
    if s in {"", "na", "n/a", "none", "[]", "--"}:
        return 0
    s = (s.replace("about", "")
           .replace("approx", "")
           .replace("~", "")
           .replace("+", "")
           .strip())
    m = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*k\s*$", s)
    if m:
        return int(round(float(m.group(1)) * 1000))
    s = s.replace(",", "").replace(" ", "")
    s = re.sub(r"[^0-9.]", "", s)
    if not s or s == ".":
        return 0
    try:
        return int(round(float(s)))
    except ValueError:
        return 0


def parse_date_any(cell):
    """Try to parse any date format; return pd.NaT if invalid."""
    if pd.isna(cell) or str(cell).strip() in {"[]", "None", ""}:
        return pd.NaT
    try:
        return pd.to_datetime(dateparser.parse(str(cell), dayfirst=False, fuzzy=True))
    except Exception:
        return pd.NaT


def parse_channel_name(cell: str) -> str:
    if cell is None:
        return ""
    cell = (str(cell)
            .replace("You tube", "YouTube")
            .replace("You Tube", "YouTube"))
    return cell.lower().strip()


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all light transforms if columns exist (idempotent)."""
    out = df.copy()

    # Manager
    if "Manager" in out.columns:
        out["Manager"] = out["Manager"].apply(lambda x: "Unknown" if str(x) in {"[]", ""} else x)

    # Clicks
    if "Clicks" in out.columns:
        out["Clicks"] = out["Clicks"].apply(parse_clicks).astype("Int64")

    # Budget (€)
    if "Budget (€)" in out.columns:
        out["Budget (€)"] = out["Budget (€)"].apply(parse_budget).astype("Int64")

    # Launch Date columns -> single "Launch Date"
    if {"Launch Date (Date)", "Launch Date (Text)"} <= set(out.columns):
        clean_dates: List[pd.Timestamp] = []
        for dval, tval in zip(out["Launch Date (Date)"], out["Launch Date (Text)"]):
            parsed = parse_date_any(dval) or pd.NaT
            if pd.isna(parsed):
                parsed = parse_date_any(tval)
            clean_dates.append(parsed)
        out["Launch Date"] = clean_dates
        out.drop(columns=["Launch Date (Date)", "Launch Date (Text)"], inplace=True, errors="ignore")

    # Channel
    if "Channel" in out.columns:
        out["Channel"] = out["Channel"].apply(parse_channel_name)

    return out


# ────────────────────────── Load to RDS ────────────────────
def get_engine(pg_url: str) -> Engine:
    logger.info("Creating SQLAlchemy engine.")
    return create_engine(pg_url, pool_pre_ping=True, pool_recycle=300)


def load_dataframe(
    df: pd.DataFrame,
    engine: Engine,
    table: str,
    if_exists: str = "replace",
    chunksize: Optional[int] = 1000,
) -> None:
    if df.empty:
        logger.warning("DataFrame is empty. Nothing to load.")
        return
    logger.info("Loading DataFrame into '%s' (if_exists=%s, chunksize=%s)…", table, if_exists, chunksize)
    df.to_sql(
        table,
        engine,
        if_exists=if_exists,
        index=False,
        chunksize=chunksize,
        method="multi",
    )
    logger.info("Load completed.")


# ──────────────────────────── Main ─────────────────────────
def main() -> None:
    # Notion
    notion_token = env("NOTION_TOKEN")
    notion_db_id = env("NOTION_DATABASE_ID")
    notion_version = env("NOTION_VERSION", "2022-06-28")

    # Postgres / RDS
    pg_host = env("PG_HOST")
    pg_port = env("PG_PORT", "5432")
    pg_db = env("PG_DB")            # e.g., "finance" (or "postgres")
    pg_user = env("PG_USER")
    pg_password = env("PG_PASSWORD")
    pg_table = env("PG_TABLE", "campaigns")
    pg_if_exists = env("PG_IF_EXISTS", "replace")  # replace | append | fail
    pg_chunksize = int(env("PG_CHUNKSIZE", "1000"))

    # Extract
    pages = fetch_notion_pages(notion_db_id, notion_token, notion_version)
    raw_df = pages_to_dataframe(pages)

    # Transform (safe guards if columns don't exist)
    df = transform(raw_df)
    logger.info("Final DF shape before load: %s", df.shape)

    # Load
    engine = get_engine(build_pg_url(pg_user, pg_password, pg_host, pg_port, pg_db))
    load_dataframe(df, engine, table=pg_table, if_exists=pg_if_exists, chunksize=pg_chunksize)


if __name__ == "__main__":
    main()
