"""Dashboard financiero Colombia: TRM, IBR y COLCAP.

Descarga datos de APIs públicas (datos.gov.co, Banco de la República, Yahoo
Finance), aplica modelos de riesgo (log-returns, GARCH, VaR, ES) y permite
explorar escenarios Monte Carlo.

Ejecutar con: streamlit run app.py
"""

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from src import charts, models
from src.data_fetchers import ASSET_INFO, fetch_colcap, fetch_ibr, fetch_trm

st.set_page_config(page_title="Dashboard Financiero Colombia", page_icon="📈",
                   layout="wide")


# ---------------------------------------------------------------------------
# Datos con caché (1 hora)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Descargando TRM (datos.gov.co)...")
def get_trm(start: date) -> pd.DataFrame:
    return fetch_trm(start)


@st.cache_data(ttl=3600, show_spinner="Descargando IBR (Banco de la República)...")
def get_ibr(start: date) -> pd.DataFrame:
    return fetch_ibr(start)


@st.cache_data(ttl=3600, show_spinner="Descargando COLCAP (Yahoo Finance)...")
def get_colcap(start: date) -> pd.DataFrame:
    return fetch_colcap(start)


GETTERS = {"TRM": get_trm, "IBR": get_ibr, "COLCAP": get_colcap}


@st.cache_resource(ttl=3600, show_spinner="Ajustando GARCH(1,1)...")
def fit_garch_cached(activo: str, start: date, n_obs: int, _retornos: pd.Series):
    # activo/start/n_obs forman la clave de caché; la serie va con _ para no hashearla
    return models.fit_garch(_retornos)


def load_series(start: date) -> dict[str, pd.DataFrame | None]:
    datos = {}
    for nombre, getter in GETTERS.items():
        try:
            datos[nombre] = getter(start)
        except Exception as exc:
            datos[nombre] = None
            st.warning(f"No se pudo descargar {nombre}: {exc}. "
                       "El dashboard sigue funcionando con las demás series.")
    return datos


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("📈 Configuración")

activo = st.sidebar.selectbox("Activo para análisis detallado", list(GETTERS),
                              format_func=lambda a: ASSET_INFO[a]["nombre"])

anios = st.sidebar.slider("Años de histórico", 1, 15, 10)
start = date.today() - timedelta(days=365 * anios)

conf = st.sidebar.radio("Nivel de confianza VaR/ES", [0.95, 0.99],
                        format_func=lambda c: f"{c:.0%}", horizontal=True)

st.sidebar.subheader("Escenarios Monte Carlo")
horizonte = st.sidebar.slider("Horizonte (días hábiles)", 10, 250, 90, step=10)
n_sims = st.sidebar.select_slider("Número de simulaciones",
                                  [500, 1000, 2500, 5000, 10000], value=2500)

if st.sidebar.button("🔄 Refrescar datos"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

st.title("Dashboard Financiero Colombia")
st.caption("TRM · IBR · COLCAP — log-returns, GARCH(1,1), VaR, ES y escenarios Monte Carlo")

datos = load_series(start)

if datos.get(activo) is None:
    st.error(f"No hay datos disponibles para {activo}. Elige otro activo en la barra lateral.")
    st.stop()

info = ASSET_INFO[activo]
serie = datos[activo]["valor"]
es_tasa = info["es_tasa"]
unidad = info["unidad"]
unidad_ret = "pb" if es_tasa else "%"

retornos = models.compute_returns(serie, es_tasa)
res_garch = fit_garch_cached(activo, start, len(retornos), retornos)

tab_resumen, tab_serie, tab_vol, tab_riesgo, tab_esc = st.tabs(
    ["📋 Resumen", "📈 Serie y retornos", "🌪️ Volatilidad (GARCH)",
     "⚠️ Riesgo (VaR/ES)", "🔮 Escenarios"])

# ---------------------------------------------------------------------------
# Tab 1: Resumen
# ---------------------------------------------------------------------------

with tab_resumen:
    cols = st.columns(3)
    for col, nombre in zip(cols, GETTERS):
        df = datos.get(nombre)
        inf = ASSET_INFO[nombre]
        with col:
            st.subheader(inf["nombre"])
            if df is None:
                st.error("Sin datos")
                continue
            s = df["valor"]
            ultimo, previo = s.iloc[-1], s.iloc[-2]
            ini_anio = s[s.index < pd.Timestamp(date.today().year, 1, 1)]
            if inf["es_tasa"]:
                delta_d = f"{(ultimo - previo) * 100:+.1f} pb"
                delta_ytd = (f"{(ultimo - ini_anio.iloc[-1]) * 100:+.0f} pb YTD"
                             if len(ini_anio) else "—")
                valor_fmt = f"{ultimo:.3f} %"
            else:
                delta_d = f"{(ultimo / previo - 1) * 100:+.2f} %"
                delta_ytd = (f"{(ultimo / ini_anio.iloc[-1] - 1) * 100:+.1f} % YTD"
                             if len(ini_anio) else "—")
                valor_fmt = f"{ultimo:,.2f}"
            st.metric(f"Último ({s.index[-1]:%Y-%m-%d})", valor_fmt, delta_d,
                      delta_color="off" if inf["es_tasa"] else "normal")
            st.caption(f"{delta_ytd} · Fuente: {inf['fuente']}")
            r = models.compute_returns(s, inf["es_tasa"])
            u = "pb" if inf["es_tasa"] else "%"
            st.caption(f"Vol. anualizada: {r.std() * (252 ** 0.5):,.1f} {u} · "
                       f"VaR 95% 1d (hist.): {-r.quantile(0.05):,.2f} {u}")
            st.line_chart(s.iloc[-252:], height=160)

# ---------------------------------------------------------------------------
# Tab 2: Serie y retornos
# ---------------------------------------------------------------------------

with tab_serie:
    st.plotly_chart(charts.price_chart(serie, info["nombre"], unidad),
                    use_container_width=True)
    st.plotly_chart(charts.returns_chart(
        retornos,
        f"{'Cambios diarios' if es_tasa else 'Log-returns diarios'} ({unidad_ret})",
        unidad_ret), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.histogram_chart(retornos, unidad_ret),
                        use_container_width=True)
    with c2:
        st.plotly_chart(charts.qq_chart(retornos), use_container_width=True)

    stats_d = models.descriptive_stats(retornos)
    st.subheader("Estadísticas de los retornos")
    tabla = pd.DataFrame({
        "Observaciones": [f"{stats_d['n']:,}"],
        f"Media diaria ({unidad_ret})": [f"{stats_d['media_diaria']:.4f}"],
        f"Vol. diaria ({unidad_ret})": [f"{stats_d['vol_diaria']:.3f}"],
        f"Vol. anualizada ({unidad_ret})": [f"{stats_d['vol_anualizada']:.2f}"],
        "Skewness": [f"{stats_d['skewness']:.3f}"],
        "Curtosis (exceso)": [f"{stats_d['kurtosis_exceso']:.2f}"],
        "Jarque-Bera p-value": [f"{stats_d['jarque_bera_pvalue']:.2e}"],
    }, index=[info["nombre"]])
    st.dataframe(tabla, use_container_width=True)
    if stats_d["jarque_bera_pvalue"] < 0.05:
        st.info("El test Jarque-Bera rechaza normalidad (p < 0.05): colas pesadas — "
                "el VaR paramétrico normal tiende a subestimar el riesgo extremo.")

# ---------------------------------------------------------------------------
# Tab 3: Volatilidad GARCH
# ---------------------------------------------------------------------------

with tab_vol:
    g = models.garch_summary(res_garch)
    c = st.columns(5)
    c[0].metric("ω (omega)", f"{g['omega']:.4f}")
    c[1].metric("α (ARCH)", f"{g['alpha']:.3f}")
    c[2].metric("β (GARCH)", f"{g['beta']:.3f}")
    c[3].metric("Persistencia α+β", f"{g['persistencia']:.3f}")
    c[4].metric("ν (g.l. t-Student)", f"{g['nu']:.1f}")
    if g["persistencia"] >= 1:
        st.warning("Persistencia ≥ 1: la varianza no es estacionaria; el forecast "
                   "de largo plazo no converge.")
    else:
        st.caption(f"Vol. incondicional: {g['vol_incond_diaria']:.3f} {unidad_ret} diaria "
                   f"({g['vol_incond_anualizada']:.1f} {unidad_ret} anualizada) · "
                   f"AIC: {g['aic']:,.1f} · BIC: {g['bic']:,.1f}")

    st.plotly_chart(charts.volatility_chart(
        retornos, res_garch.conditional_volatility, unidad_ret),
        use_container_width=True)

    dias_fc = st.slider("Días de forecast de volatilidad", 5, 120, 30, key="vol_fc")
    vol_fc = models.garch_vol_forecast(res_garch, dias_fc)
    st.plotly_chart(charts.vol_forecast_chart(vol_fc, g["vol_incond_diaria"], unidad_ret),
                    use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 4: Riesgo VaR/ES
# ---------------------------------------------------------------------------

with tab_riesgo:
    tabla_var = models.var_es_table(retornos, res_garch, conf)
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.subheader(f"VaR y ES a 1 día ({unidad_ret})")
        st.dataframe(tabla_var.style.format("{:.3f}"), use_container_width=True)
        ultimo = serie.iloc[-1]
        var_g = tabla_var.loc["Condicional (GARCH)"].iloc[0]
        if es_tasa:
            st.caption(f"Lectura: con {conf:.0%} de confianza, el IBR no debería "
                       f"subir/bajar más de {var_g:.1f} pb en un día.")
        else:
            st.caption(f"Lectura: con {conf:.0%} de confianza, la pérdida diaria no "
                       f"debería exceder {var_g:.2f}% "
                       f"(≈ {ultimo * var_g / 100:,.1f} {unidad} sobre el nivel actual).")
    with c2:
        st.plotly_chart(charts.var_comparison_chart(tabla_var, unidad_ret),
                        use_container_width=True)

    st.subheader("Backtesting del VaR histórico (ventana rodante de 250 días)")
    bt = models.backtest_var(retornos, conf)
    c = st.columns(4)
    c[0].metric("Observaciones evaluadas", f"{bt['n_obs']:,}")
    c[1].metric("Violaciones", bt["n_violaciones"])
    c[2].metric("Esperadas", f"{bt['esperadas']:.1f}")
    kupiec = bt["kupiec_pvalue"]
    c[3].metric("Kupiec p-value", "—" if pd.isna(kupiec) else f"{kupiec:.3f}")
    if not pd.isna(kupiec):
        if kupiec < 0.05:
            st.warning("El test de Kupiec rechaza el modelo (p < 0.05): el número de "
                       "violaciones difiere significativamente de lo esperado.")
        else:
            st.success("El test de Kupiec no rechaza el modelo: las violaciones son "
                       "consistentes con el nivel de confianza.")
    st.plotly_chart(charts.backtest_chart(retornos, bt["var_serie"], bt["violaciones"],
                                          conf, unidad_ret), use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 5: Escenarios Monte Carlo
# ---------------------------------------------------------------------------

with tab_esc:
    ultimo = float(serie.iloc[-1])
    paths = models.monte_carlo_paths(ultimo, res_garch, horizonte, n_sims, es_tasa)
    pct = models.fan_chart_percentiles(paths)

    st.plotly_chart(charts.fan_chart(
        pct, serie, f"Proyección Monte Carlo a {horizonte} días "
        f"({n_sims:,} simulaciones GARCH)", unidad), use_container_width=True)

    c = st.columns(4)
    c[0].metric("Nivel actual", f"{ultimo:,.2f}")
    c[1].metric(f"Mediana a {horizonte}d", f"{pct['p50'].iloc[-1]:,.2f}")
    c[2].metric("Percentil 5", f"{pct['p5'].iloc[-1]:,.2f}")
    c[3].metric("Percentil 95", f"{pct['p95'].iloc[-1]:,.2f}")

    st.subheader("Probabilidad de cruzar un umbral")
    c1, c2 = st.columns([1, 2])
    with c1:
        umbral = st.number_input(f"Umbral ({unidad})", value=round(ultimo * 1.05, 2),
                                 step=0.25 if es_tasa else 50.0)
        direccion = st.radio("Dirección", ["Por encima", "Por debajo"], horizontal=True)
        por_encima = direccion == "Por encima"
        probs = models.prob_threshold(paths, umbral, por_encima)
        st.metric(f"P(terminar {direccion.lower()} de {umbral:,.2f})",
                  f"{probs['p_terminal']:.1%}")
        st.metric(f"P(tocar el umbral en {horizonte}d)", f"{probs['p_toca']:.1%}")
    with c2:
        st.plotly_chart(charts.terminal_dist_chart(paths[:, -1], umbral, unidad),
                        use_container_width=True)

st.divider()
st.caption("Fuentes: TRM — Superfinanciera vía datos.gov.co · IBR — Banco de la "
           "República (web service SDMX) · COLCAP — Yahoo Finance (ICOLCAP.CL). "
           "Modelos: GARCH(1,1)-t (librería arch). Uso académico/informativo; "
           "no constituye asesoría de inversión.")
