"""Figuras Plotly para el dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

COLOR = "#1f77b4"
COLOR_ROJO = "#d62728"
COLOR_GRIS = "rgba(120,120,120,0.5)"


def _layout(fig: go.Figure, titulo: str, y_titulo: str = "") -> go.Figure:
    fig.update_layout(
        title=titulo,
        yaxis_title=y_titulo,
        template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=40),
        hovermode="x unified",
    )
    return fig


def price_chart(serie: pd.Series, titulo: str, unidad: str) -> go.Figure:
    fig = go.Figure(go.Scatter(x=serie.index, y=serie.values, mode="lines",
                               line=dict(color=COLOR, width=1.5), name=titulo))
    fig.update_xaxes(rangeslider_visible=True,
                     rangeselector=dict(buttons=[
                         dict(count=6, label="6m", step="month", stepmode="backward"),
                         dict(count=1, label="1a", step="year", stepmode="backward"),
                         dict(count=5, label="5a", step="year", stepmode="backward"),
                         dict(step="all", label="Todo"),
                     ]))
    return _layout(fig, titulo, unidad)


def returns_chart(retornos: pd.Series, titulo: str, unidad_ret: str) -> go.Figure:
    fig = go.Figure(go.Scatter(x=retornos.index, y=retornos.values, mode="lines",
                               line=dict(color=COLOR, width=0.8), name="Retornos"))
    return _layout(fig, titulo, unidad_ret)


def histogram_chart(retornos: pd.Series, unidad_ret: str) -> go.Figure:
    x = np.linspace(retornos.min(), retornos.max(), 400)
    mu, sigma = retornos.mean(), retornos.std()
    nu, loc, scale = stats.t.fit(retornos)

    fig = go.Figure()
    fig.add_histogram(x=retornos, histnorm="probability density", nbinsx=80,
                      name="Retornos", marker_color=COLOR, opacity=0.6)
    fig.add_scatter(x=x, y=stats.norm.pdf(x, mu, sigma), mode="lines",
                    name="Normal", line=dict(color=COLOR_ROJO, dash="dash"))
    fig.add_scatter(x=x, y=stats.t.pdf(x, nu, loc, scale), mode="lines",
                    name=f"t-Student (ν={nu:.1f})", line=dict(color="green"))
    fig.update_layout(barmode="overlay")
    return _layout(fig, "Distribución de retornos vs. Normal y t-Student", "Densidad")


def qq_chart(retornos: pd.Series) -> go.Figure:
    std = (retornos - retornos.mean()) / retornos.std()
    osm, osr = stats.probplot(std, dist="norm", fit=False)
    fig = go.Figure()
    fig.add_scatter(x=osm, y=osr, mode="markers", name="Cuantiles",
                    marker=dict(color=COLOR, size=4))
    lims = [min(osm.min(), osr.min()), max(osm.max(), osr.max())]
    fig.add_scatter(x=lims, y=lims, mode="lines", name="45°",
                    line=dict(color=COLOR_ROJO, dash="dash"))
    fig.update_xaxes(title="Cuantiles teóricos (Normal)")
    return _layout(fig, "QQ-plot vs. Normal", "Cuantiles muestrales")


def volatility_chart(retornos: pd.Series, vol_condicional: pd.Series,
                     unidad_ret: str, ventana: int = 21) -> go.Figure:
    vol_realizada = retornos.rolling(ventana).std()
    fig = go.Figure()
    fig.add_scatter(x=vol_realizada.index, y=vol_realizada.values, mode="lines",
                    name=f"Vol. realizada ({ventana}d)", line=dict(color=COLOR_GRIS, width=1))
    fig.add_scatter(x=vol_condicional.index, y=vol_condicional.values, mode="lines",
                    name="Vol. condicional GARCH", line=dict(color=COLOR_ROJO, width=1.5))
    return _layout(fig, "Volatilidad diaria: GARCH vs. realizada", unidad_ret)


def vol_forecast_chart(vol_fc: pd.Series, vol_incond: float, unidad_ret: str) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=vol_fc.index, y=vol_fc.values, mode="lines+markers",
                    name="Forecast GARCH", line=dict(color=COLOR))
    if not np.isnan(vol_incond):
        fig.add_hline(y=vol_incond, line_dash="dash", line_color=COLOR_ROJO,
                      annotation_text="Vol. incondicional (largo plazo)")
    fig.update_xaxes(title="Días hacia adelante")
    return _layout(fig, "Forecast de volatilidad diaria", unidad_ret)


def var_comparison_chart(tabla: pd.DataFrame, unidad_ret: str) -> go.Figure:
    fig = go.Figure()
    for col in tabla.columns:
        fig.add_bar(name=col, x=tabla.index, y=tabla[col])
    fig.update_layout(barmode="group")
    return _layout(fig, "VaR y ES a 1 día por método", f"Pérdida ({unidad_ret})")


def backtest_chart(retornos: pd.Series, var_serie: pd.Series,
                   violaciones: pd.Series, conf: float, unidad_ret: str) -> go.Figure:
    r = retornos.loc[var_serie.index]
    fig = go.Figure()
    fig.add_scatter(x=r.index, y=r.values, mode="lines", name="Retornos",
                    line=dict(color=COLOR_GRIS, width=0.8))
    fig.add_scatter(x=var_serie.index, y=-var_serie.values, mode="lines",
                    name=f"-VaR {conf:.0%} (histórico rodante)",
                    line=dict(color=COLOR_ROJO, width=1.2))
    viol = r[violaciones]
    fig.add_scatter(x=viol.index, y=viol.values, mode="markers", name="Violaciones",
                    marker=dict(color="black", size=6, symbol="x"))
    return _layout(fig, "Backtesting del VaR", unidad_ret)


def fan_chart(percentiles: pd.DataFrame, historico: pd.Series,
              titulo: str, unidad: str, dias_historia: int = 120) -> go.Figure:
    """Fan chart: histórico reciente + abanico de percentiles simulados."""
    hist = historico.iloc[-dias_historia:]
    fechas_fut = pd.bdate_range(historico.index[-1], periods=len(percentiles), freq="B")

    fig = go.Figure()
    fig.add_scatter(x=hist.index, y=hist.values, mode="lines", name="Histórico",
                    line=dict(color=COLOR, width=1.5))
    bandas = [("p5", "p95", "rgba(214,39,40,0.15)", "5–95%"),
              ("p25", "p75", "rgba(214,39,40,0.30)", "25–75%")]
    for lo, hi, color, nombre in bandas:
        fig.add_scatter(x=fechas_fut, y=percentiles[hi], mode="lines",
                        line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=fechas_fut, y=percentiles[lo], mode="lines",
                        line=dict(width=0), fill="tonexty", fillcolor=color,
                        name=f"Banda {nombre}")
    fig.add_scatter(x=fechas_fut, y=percentiles["p50"], mode="lines",
                    name="Mediana", line=dict(color=COLOR_ROJO, dash="dash"))
    return _layout(fig, titulo, unidad)


def terminal_dist_chart(valores_terminales: np.ndarray, umbral: float | None,
                        unidad: str) -> go.Figure:
    fig = go.Figure(go.Histogram(x=valores_terminales, nbinsx=80,
                                 marker_color=COLOR, opacity=0.7, name="Valor terminal"))
    if umbral is not None:
        fig.add_vline(x=umbral, line_dash="dash", line_color=COLOR_ROJO,
                      annotation_text=f"Umbral: {umbral:,.2f}")
    fig.update_xaxes(title=unidad)
    return _layout(fig, "Distribución del valor terminal simulado", "Frecuencia")
