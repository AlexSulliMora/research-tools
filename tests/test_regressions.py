import polars as pl

from research_tools import regressions


def test_winsorize_clips_at_quantiles():
    df = pl.DataFrame({'x': [float(i) for i in range(101)]})
    out = regressions.winsorize(df, ['x'])
    assert out['x'].min() == 1.0 and out['x'].max() == 99.0


def test_paper_layout_rows():
    model = regressions.ModelResult(
        name='(1)', dependent='y', coefficients={'x': -0.1234}, standard_errors={'x': 0.05},
        t_statistics={'x': -2.468}, p_values={'x': 0.02}, adjusted_r2=0.1234, nobs=1500,
    )
    table = regressions.paper_layout([model], {'x': 'X'}, ['(1)'])
    assert table['Model'].to_list() == ['X', '', 'Adj R2', 'Nobs']
    assert table['(1)'].to_list() == ['−0.123**', '−2.47', '12.34%', '1,500']
