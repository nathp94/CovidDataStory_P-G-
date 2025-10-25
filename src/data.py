import io, pandas as pd
from typing import Tuple
import streamlit as st

def _read_csv_auto(b: bytes) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin-1"]:
        try:
            return pd.read_csv(
                io.BytesIO(b), sep=",", encoding=enc, low_memory=False,
                dtype={"dep": "string", "lib_dep": "string", "lib_reg": "string"},
                parse_dates=["date"], dayfirst=False
            )
        except Exception:
            pass
    raise ValueError("Encodage non reconnu.")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = ['cv_dose1', 'R']
    df2 = df.drop(columns=columns_to_drop, errors='ignore')
    df2_clean = df2.drop_duplicates(keep='first')
    data_clean = df2_clean.dropna(axis=0, how='any')
    data_clean = data_clean.reset_index(drop=True)
    data_clean['date'] = pd.to_datetime(data_clean['date'])
    return data_clean

def national_data(df: pd.DataFrame) -> pd.DataFrame:
    cols_sum = [
        "hosp", "rea", "rad", "dchosp",
        "incid_hosp", "incid_rea", "incid_rad", "incid_dchosp",
        "pos", "pos_7j"
    ]
    nat = (df.groupby("date", as_index=False)[cols_sum].sum())
    return nat

def df_info(df: pd.DataFrame) -> Tuple[int, int, int]:
    n_rows, n_cols = df.shape
    n_missing = int(df.isna().sum().sum())
    return n_rows, n_cols, n_missing

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

@st.cache_data(show_spinner=True)
def load_and_prepare(csv_bytes: bytes):
    """
    Parse the uploaded CSV bytes, clean the data, and compute national aggregates.
    Cached so repeated interactions don't re-do the heavy work.
    """
    df_ini = _read_csv_auto(csv_bytes)
    df = clean_data(df_ini)
    nat = national_data(df)
    return df_ini, df, nat