"""Modelos estadísticos: log-returns, GARCH, VaR, ES, backtesting y Monte Carlo."""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Retornos y estadísticas descriptivas
# ---------------------------------------------------------------------------

def compute_returns(precios: pd.Series, es_tasa: bool = False) -> pd.Series:
    """Log-returns en % para precios; primeras diferencias en pb para tasas.

    Para una tasa como el IBR el log-return no tiene sentido económico, por lo
    que se modela el cambio diario en puntos básicos (pb). La escala en pb
    además evita problemas numéricos al estimar el GARCH (varianzas ínfimas).
    """
    if es_tasa:
        r = precios.diff() * 100  # puntos porcentuales -> puntos básicos
    else:
        r = np.log(precios / precios.shift(1)) * 100
    return r.dropna()


def descriptive_stats(retornos: pd.Series) -> dict:
    jb_stat, jb_p = stats.jarque_bera(retornos)
    return {
        "n": len(retornos),
        "media_diaria": retornos.mean(),
        "vol_diaria": retornos.std(),
        "vol_anualizada": retornos.std() * np.sqrt(TRADING_DAYS),
        "skewness": stats.skew(retornos),
        "kurtosis_exceso": stats.kurtosis(retornos),
        "min": retornos.min(),
        "max": retornos.max(),
        "jarque_bera": jb_stat,
        "jarque_bera_pvalue": jb_p,
    }


# ---------------------------------------------------------------------------
# GARCH(1,1)
# ---------------------------------------------------------------------------

def fit_garch(retornos: pd.Series):
    """Ajusta un GARCH(1,1) con distribución t de Student sobre los retornos.

    Devuelve el resultado de `arch` (ARCHModelResult).
    """
    model = arch_model(retornos, vol="GARCH", p=1, q=1, dist="t", mean="Constant")
    return model.fit(disp="off")


def garch_summary(res) -> dict:
    p = res.params
    alpha, beta = p["alpha[1]"], p["beta[1]"]
    persistencia = alpha + beta
    vol_lp = np.sqrt(p["omega"] / (1 - persistencia)) if persistencia < 1 else np.nan
    return {
        "omega": p["omega"],
        "alpha": alpha,
        "beta": beta,
        "nu": p.get("nu", np.nan),
        "persistencia": persistencia,
        "vol_incond_diaria": vol_lp,
        "vol_incond_anualizada": vol_lp * np.sqrt(TRADING_DAYS) if not np.isnan(vol_lp) else np.nan,
        "loglik": res.loglikelihood,
        "aic": res.aic,
        "bic": res.bic,
    }


def garch_vol_forecast(res, horizonte: int = 30) -> pd.Series:
    """Forecast de volatilidad diaria (mismas unidades de los retornos) a N días."""
    fc = res.forecast(horizon=horizonte, reindex=False)
    vol = np.sqrt(fc.variance.iloc[-1].values)
    return pd.Series(vol, index=np.arange(1, horizonte + 1), name="vol_diaria")


# ---------------------------------------------------------------------------
# VaR y Expected Shortfall
# ---------------------------------------------------------------------------

def var_es_table(retornos: pd.Series, res_garch, conf: float = 0.95) -> pd.DataFrame:
    """VaR y ES a 1 día (en las unidades de los retornos: % o pp) por método.

    Convención: se reportan como pérdidas positivas (cuantil izquierdo con
    signo invertido).
    """
    alpha = 1 - conf
    mu, sigma = retornos.mean(), retornos.std()
    filas = {}

    # Paramétrico normal
    z = stats.norm.ppf(alpha)
    var_n = -(mu + sigma * z)
    es_n = -(mu - sigma * stats.norm.pdf(z) / alpha)
    filas["Paramétrico (Normal)"] = (var_n, es_n)

    # Paramétrico t de Student (ajustada a los retornos)
    nu, loc, scale = stats.t.fit(retornos)
    tq = stats.t.ppf(alpha, nu)
    var_t = -(loc + scale * tq)
    if nu > 1:
        # ES analítico de la t: E[X | X < q] para la t estandarizada
        es_t_std = -(stats.t.pdf(tq, nu) / alpha) * (nu + tq**2) / (nu - 1)
        es_t = -(loc + scale * es_t_std)
    else:
        # Con nu <= 1 la media (y por tanto el ES) de la t no existe
        es_t = np.nan
    filas["Paramétrico (t-Student)"] = (var_t, es_t)

    # Histórico
    q_h = retornos.quantile(alpha)
    var_h = -q_h
    es_h = -retornos[retornos <= q_h].mean()
    filas["Histórico"] = (var_h, es_h)

    # Condicional GARCH: sigma del próximo día + cuantil t estandarizada del modelo
    fc = res_garch.forecast(horizon=1, reindex=False)
    sigma_g = float(np.sqrt(fc.variance.iloc[-1, 0]))
    mu_g = res_garch.params.get("mu", mu)
    nu_g = res_garch.params.get("nu", np.nan)
    if np.isnan(nu_g):
        qz, es_z = z, -stats.norm.pdf(z) / alpha
    else:
        # cuantil de la t estandarizada (varianza 1)
        t_std = np.sqrt((nu_g - 2) / nu_g)
        qz = stats.t.ppf(alpha, nu_g) * t_std
        es_z = -(stats.t.pdf(stats.t.ppf(alpha, nu_g), nu_g) / alpha) * \
            (nu_g + stats.t.ppf(alpha, nu_g) ** 2) / (nu_g - 1) * t_std
    filas["Condicional (GARCH)"] = (-(mu_g + sigma_g * qz), -(mu_g + sigma_g * es_z))

    df = pd.DataFrame(filas, index=[f"VaR {conf:.0%}", f"ES {conf:.0%}"]).T
    return df


def backtest_var(retornos: pd.Series, conf: float = 0.95, ventana: int = 250) -> dict:
    """Backtest de VaR histórico con ventana rodante + test de Kupiec (POF)."""
    alpha = 1 - conf
    var_serie = -retornos.rolling(ventana).quantile(alpha).shift(1)
    var_serie = var_serie.dropna()
    realizados = retornos.loc[var_serie.index]
    violaciones = realizados < -var_serie
    n, x = len(violaciones), int(violaciones.sum())
    esperadas = alpha * n

    # Test de Kupiec (proportion of failures)
    if 0 < x < n:
        p_hat = x / n
        lr = -2 * (
            (n - x) * np.log(1 - alpha) + x * np.log(alpha)
            - (n - x) * np.log(1 - p_hat) - x * np.log(p_hat)
        )
        p_value = 1 - stats.chi2.cdf(lr, df=1)
    else:
        lr, p_value = np.nan, np.nan

    return {
        "var_serie": var_serie,
        "violaciones": violaciones,
        "n_obs": n,
        "n_violaciones": x,
        "esperadas": esperadas,
        "kupiec_lr": lr,
        "kupiec_pvalue": p_value,
    }


# ---------------------------------------------------------------------------
# Simulación Monte Carlo (escenarios)
# ---------------------------------------------------------------------------

def monte_carlo_paths(
    ultimo_precio: float,
    res_garch,
    horizonte: int = 90,
    n_sims: int = 1000,
    es_tasa: bool = False,
    seed: int | None = 42,
) -> np.ndarray:
    """Simula trayectorias del activo usando el GARCH(1,1) ajustado.

    Recursión: sigma2_t = omega + alpha*e_{t-1}^2 + beta*sigma2_{t-1}, con
    innovaciones t de Student estandarizadas. Para precios los retornos
    simulados son log-returns en %, para tasas son cambios en pb.

    Devuelve un array (n_sims, horizonte + 1) que incluye el valor inicial.
    """
    rng = np.random.default_rng(seed)
    p = res_garch.params
    mu = p.get("mu", 0.0)
    omega, alpha, beta = p["omega"], p["alpha[1]"], p["beta[1]"]
    nu = p.get("nu", np.nan)

    # Estado inicial: última varianza condicional y último residuo del ajuste
    sigma2 = np.full(n_sims, res_garch.conditional_volatility.iloc[-1] ** 2)
    e_prev = np.full(n_sims, res_garch.resid.iloc[-1])

    if np.isnan(nu):
        innov = rng.standard_normal((horizonte, n_sims))
    else:
        innov = rng.standard_t(nu, (horizonte, n_sims)) * np.sqrt((nu - 2) / nu)

    retornos = np.empty((horizonte, n_sims))
    for t in range(horizonte):
        sigma2 = omega + alpha * e_prev**2 + beta * sigma2
        e_prev = np.sqrt(sigma2) * innov[t]
        retornos[t] = mu + e_prev

    paths = np.empty((n_sims, horizonte + 1))
    paths[:, 0] = ultimo_precio
    if es_tasa:
        # retornos en pb -> nivel de la tasa en puntos porcentuales
        paths[:, 1:] = ultimo_precio + np.cumsum(retornos.T / 100, axis=1)
    else:
        paths[:, 1:] = ultimo_precio * np.exp(np.cumsum(retornos.T / 100, axis=1))
    return paths


def fan_chart_percentiles(paths: np.ndarray, percentiles=(5, 25, 50, 75, 95)) -> pd.DataFrame:
    """Percentiles por paso de tiempo para el fan chart."""
    data = {f"p{p}": np.percentile(paths, p, axis=0) for p in percentiles}
    return pd.DataFrame(data, index=np.arange(paths.shape[1]))


def prob_threshold(paths: np.ndarray, umbral: float, por_encima: bool = True) -> dict:
    """Probabilidad de cruzar/terminar sobre un umbral en el horizonte simulado."""
    terminal = paths[:, -1]
    if por_encima:
        p_terminal = float((terminal > umbral).mean())
        p_toca = float((paths.max(axis=1) > umbral).mean())
    else:
        p_terminal = float((terminal < umbral).mean())
        p_toca = float((paths.min(axis=1) < umbral).mean())
    return {"p_terminal": p_terminal, "p_toca": p_toca}
