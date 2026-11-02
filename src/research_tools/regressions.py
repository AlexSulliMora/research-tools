"""
Winsorizing. Estimate with pyfixest directly (`pf.feols(...)`) and make regression tables with
`pf.etable(...)`; see the README for the paper layout.

    winsorize(column, tail)        an expression clipped at its `tail` and 1 - `tail` quantiles
"""

import polars as pl


def winsorize(column: pl.Expr, tail: float = 0.01) -> pl.Expr:
    """`column` clipped at its `tail` and 1 - `tail` quantiles (linear interpolation), so that values beyond
    them are set to them. Use it in a context: `df.with_columns(pl.col('a', 'b').pipe(winsorize, 0.05))`."""
    if not 0 <= tail < 0.5:
        raise ValueError(f'tail must be in [0, 0.5), got {tail}')
    return column.clip(
        column.quantile(tail, interpolation='linear'),
        column.quantile(1 - tail, interpolation='linear'),
    )
