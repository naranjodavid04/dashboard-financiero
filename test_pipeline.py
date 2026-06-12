"""Smoke test del pipeline completo: datos -> modelos. Ejecutar con el venv."""

from datetime import date, timedelta

from src import models
from src.data_fetchers import ASSET_INFO, FETCHERS

start = date.today() - timedelta(days=365 * 10)

for nombre, fetcher in FETCHERS.items():
    info = ASSET_INFO[nombre]
    print(f"\n=== {nombre} ({info['nombre']}) ===")
    df = fetcher(start)
    print(f"  filas={len(df)}  rango={df.index.min().date()} -> {df.index.max().date()}")
    assert len(df) > 1000, f"{nombre}: muy pocos datos"

    serie = df["valor"]
    r = models.compute_returns(serie, info["es_tasa"])
    stats_d = models.descriptive_stats(r)
    print(f"  retornos n={stats_d['n']}  vol_diaria={stats_d['vol_diaria']:.3f}  "
          f"kurtosis={stats_d['kurtosis_exceso']:.1f}")

    res = models.fit_garch(r)
    g = models.garch_summary(res)
    print(f"  GARCH: alpha={g['alpha']:.3f} beta={g['beta']:.3f} "
          f"persistencia={g['persistencia']:.3f} nu={g['nu']:.1f}")
    assert g["persistencia"] < 1.05, f"{nombre}: persistencia anómala"

    for conf in (0.95, 0.99):
        t = models.var_es_table(r, res, conf)
        print(f"  VaR/ES {conf:.0%}:")
        print(t.round(3).to_string(line_width=120))
        # sanity: ES >= VaR para cada método (el ES de la t puede ser NaN si nu<=1)
        ok = t.dropna()
        assert (ok.iloc[:, 1] >= ok.iloc[:, 0] - 1e-9).all(), f"{nombre}: ES < VaR"

    t95 = models.var_es_table(r, res, 0.95)
    t99 = models.var_es_table(r, res, 0.99)
    assert (t99.iloc[:, 0].values >= t95.iloc[:, 0].values).all(), "VaR99 < VaR95"

    bt = models.backtest_var(r, 0.95)
    print(f"  Backtest 95%: {bt['n_violaciones']} violaciones de {bt['n_obs']} "
          f"(esperadas {bt['esperadas']:.1f}, Kupiec p={bt['kupiec_pvalue']:.3f})")

    paths = models.monte_carlo_paths(float(serie.iloc[-1]), res, horizonte=90,
                                     n_sims=1000, es_tasa=info["es_tasa"])
    pct = models.fan_chart_percentiles(paths)
    print(f"  MC 90d: mediana={pct['p50'].iloc[-1]:,.2f}  "
          f"p5={pct['p5'].iloc[-1]:,.2f}  p95={pct['p95'].iloc[-1]:,.2f}  "
          f"(actual {serie.iloc[-1]:,.2f})")
    assert paths.shape == (1000, 91)
    assert (pct["p5"] <= pct["p95"]).all()

print("\nTODO OK")
