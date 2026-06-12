"""Descarga de series financieras colombianas desde APIs públicas.

Fuentes:
- TRM:    API Socrata de datos.gov.co (dato oficial de la Superfinanciera).
- IBR:    Web service SDMX del Banco de la República (totoro.banrep.gov.co).
- COLCAP: Yahoo Finance vía yfinance (ETF ICOLCAP.CL, proxy líquido del índice).

Cada fetcher devuelve un DataFrame con DatetimeIndex y una columna `valor`.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

import pandas as pd
import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

TRM_URL = "https://www.datos.gov.co/resource/32sa-8pi3.json"
IBR_URL = (
    "https://totoro.banrep.gov.co/nsi-jax-ws/rest/data/"
    "ESTAT,DF_IBR_DAILY_HIST,1.0/all/ALL/"
)


def _default_start(years: int = 10) -> date:
    return date.today() - timedelta(days=365 * years)


def fetch_trm(start: date | None = None) -> pd.DataFrame:
    """TRM oficial (COP/USD) desde datos.gov.co (dataset 32sa-8pi3)."""
    start = start or _default_start()
    params = {
        "$select": "vigenciadesde,valor",
        "$where": f"vigenciadesde >= '{start.isoformat()}'",
        "$order": "vigenciadesde",
        "$limit": "20000",
    }
    resp = requests.get(TRM_URL, params=params, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    if df.empty:
        raise ValueError("La API de TRM no devolvió datos")
    df["fecha"] = pd.to_datetime(df["vigenciadesde"]).dt.normalize()
    df["valor"] = df["valor"].astype(float)
    # La TRM rige también fines de semana/festivos con el mismo valor; nos
    # quedamos con un dato por fecha de inicio de vigencia (serie de días hábiles).
    df = df.drop_duplicates(subset="fecha", keep="last")
    return df.set_index("fecha")[["valor"]].sort_index()


def fetch_ibr(start: date | None = None) -> pd.DataFrame:
    """IBR overnight, tasa efectiva anual (%), desde el web service SDMX de banrep.

    Filtra la serie SUBJECT=IRIBRM00 (plazo overnight) con UNIT_MEASURE=ER
    (tasa efectiva). El servicio responde SDMX-ML 2.1; se parsea con regex
    para no depender de librerías XML adicionales.
    """
    start = start or _default_start()
    params = {
        "startPeriod": str(start.year),
        "endPeriod": str(date.today().year + 1),
        "dimensionAtObservation": "TIME_PERIOD",
        "detail": "full",
    }
    resp = requests.get(IBR_URL, params=params, headers=HEADERS, timeout=120)
    resp.raise_for_status()
    xml = resp.text

    series_blocks = re.findall(r"<generic:Series>.*?</generic:Series>", xml, re.S)
    target = None
    for block in series_blocks:
        if 'id="SUBJECT" value="IRIBRM00"' in block and 'id="UNIT_MEASURE" value="ER"' in block:
            target = block
            break
    if target is None:
        raise ValueError("No se encontró la serie IBR overnight (IRIBRM00/ER) en la respuesta SDMX")

    obs = re.findall(
        r'<generic:ObsDimension value="(\d{8})" /><generic:ObsValue value="([\d.]+)"',
        target,
    )
    if not obs:
        raise ValueError("La serie IBR no contiene observaciones")
    df = pd.DataFrame(obs, columns=["fecha", "valor"])
    df["fecha"] = pd.to_datetime(df["fecha"], format="%Y%m%d")
    df["valor"] = df["valor"].astype(float)
    df = df.set_index("fecha")[["valor"]].sort_index()
    return df[df.index >= pd.Timestamp(start)]


def fetch_colcap(start: date | None = None) -> pd.DataFrame:
    """COLCAP vía Yahoo Finance usando el ETF ICOLCAP.CL (BVC) como proxy."""
    import yfinance as yf

    start = start or _default_start()
    df = yf.download(
        "ICOLCAP.CL", start=start.isoformat(), progress=False, auto_adjust=True
    )
    if df is None or df.empty:
        raise ValueError("Yahoo Finance no devolvió datos para ICOLCAP.CL")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance>=0.2.50 usa columnas MultiIndex
        close = close.iloc[:, 0]
    out = close.to_frame("valor")
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out.index.name = "fecha"
    return out.dropna().sort_index()


FETCHERS = {
    "TRM": fetch_trm,
    "IBR": fetch_ibr,
    "COLCAP": fetch_colcap,
}

ASSET_INFO = {
    "TRM": {
        "nombre": "TRM (COP/USD)",
        "unidad": "COP por USD",
        "fuente": "Superfinanciera vía datos.gov.co",
        "es_tasa": False,
    },
    "IBR": {
        "nombre": "IBR overnight (E.A.)",
        "unidad": "% efectivo anual",
        "fuente": "Banco de la República (SDMX)",
        "es_tasa": True,
    },
    "COLCAP": {
        "nombre": "COLCAP (ETF ICOLCAP)",
        "unidad": "COP",
        "fuente": "Yahoo Finance (ICOLCAP.CL)",
        "es_tasa": False,
    },
}
