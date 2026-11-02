import polars as pl
import pyfixest as pf

from research_tools import regressions


def test_winsorize_clips_at_quantiles():
    df = pl.DataFrame({'x': [float(i) for i in range(101)], 'y': [float(-i) for i in range(101)]})
    out = df.with_columns(pl.col('x', 'y').pipe(regressions.winsorize))
    assert out['x'].min() == 1.0 and out['x'].max() == 99.0
    assert out['y'].min() == -99.0 and out['y'].max() == -1.0
    out = df.with_columns(pl.col('x').pipe(regressions.winsorize, 0.05))
    assert out['x'].min() == 5.0 and out['x'].max() == 95.0


def test_paper_layout_reads_pyfixest_fit():
    fit = pf.feols('Y ~ X1', data=pf.get_data(), vcov='hetero')
    tidy = fit.tidy()
    table = regressions.paper_layout([fit], {'X1': 'X', 'X2': 'Missing'}, ['(1)'], {'Y': 'Outcome'})
    assert table['Model'].to_list() == ['Dependent variable', 'X', '', 'Missing', '', 'Adj R2', 'Nobs']
    cells = table['(1)'].to_list()
    assert cells[0] == 'Outcome'
    assert cells[1] == regressions.printed_number(tidy.at['X1', 'Estimate'], 3) + '***'
    assert cells[2] == regressions.printed_number(tidy.at['X1', 't value'], 2)
    assert cells[3:5] == ['', '']
    assert cells[6] == f'{fit._N:,}'
