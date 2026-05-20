from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import gspread
import pandas as pd
import pytz
from gspread.exceptions import WorksheetNotFound
from oauth2client.service_account import ServiceAccountCredentials


def _load_google_credentials() -> dict[str, Any]:
    """
    Load Google service account credentials from env or a local file.
    """
    env_json = os.environ.get('GOOGLE')
    if env_json:
        return json.loads(env_json)

    credentials_path = os.environ.get('GOOGLE_CREDENTIALS_PATH')
    if credentials_path:
        path = Path(credentials_path).expanduser()
    else:
        path = Path(__file__).resolve().parent.parent / 'credentials.json'

    if not path.exists():
        raise FileNotFoundError(
            'Google credentials not found. Set the GOOGLE env var with the service '
            'account JSON, set GOOGLE_CREDENTIALS_PATH, or place credentials.json '
            'at the repository root.'
        )

    return json.loads(path.read_text(encoding='utf-8'))


def push_to_sheet(df: pd.DataFrame, sheet_name: str) -> None:
    """
    Append sentiment data to the named sheet tab.
    """

    scope = [
        'https://spreadsheets.google.com/feeds',
        'https://www.googleapis.com/auth/drive',
    ]

    creds_dict = _load_google_credentials()
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)

    sheet = client.open_by_key('1EQAhrCWmOzDD6VhVig4f3AffWMVZmrsrZKkgUc6h6WQ')
    try:
        worksheet = sheet.worksheet(sheet_name)
    except WorksheetNotFound:
        rows = max(len(df) + 1, 1000)
        worksheet = sheet.add_worksheet(title=sheet_name, rows=rows, cols=4)

    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')

    df = df.copy()
    df['ticker'] = df['ticker'].astype(str)

    existing_data = worksheet.get_all_values()

    rows = []

    # ✅ header only once
    if not existing_data:
        rows.append(['Stock Name', 'Sentiment Score', '', 'Date & Time'])

    # ✅ data
    for _, row in df.iterrows():
        rows.append([str(row['ticker']), float(row['sentiment_score']), '', now])

    # 🔥 FIX: force append from column A
    start_row = len(existing_data) + 1
    worksheet.update(rows, f'A{start_row}')
