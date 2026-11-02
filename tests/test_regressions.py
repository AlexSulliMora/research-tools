import polars as pl

from research_tools import regressions


def test_winsorize_clips_at_quantiles():
    df = pl.DataFrame({'x': [float(i) for i in range(101)], 'y': [float(-i) for i in range(101)]})
    out = df.with_columns(pl.col('x', 'y').pipe(regressions.winsorize))
    assert out['x'].min() == 1.0 and out['x'].max() == 99.0
    assert out['y'].min() == -99.0 and out['y'].max() == -1.0
    out = df.with_columns(pl.col('x').pipe(regressions.winsorize, 0.05))
    assert out['x'].min() == 5.0 and out['x'].max() == 95.0

