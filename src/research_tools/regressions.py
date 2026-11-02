"""
Winsorizing, and pyfixest models as a table in the layout of finance and accounting papers. This module
estimates nothing: fit each model with pyfixest yourself (`pf.feols(...)`) and pass the fitted models here.

    winsorize(column, tail)        an expression clipped at its `tail` and 1 - `tail` quantiles
    paper_layout(fits, ...)        fitted pyfixest models as papers print them: a coefficient with stars,
                                   the t-statistic on the row below, then adjusted R2 and Nobs
"""

import polars as pl

MINUS = '−'   # negative numbers are printed with a minus sign, not a hyphen

# Significance stars: *** 1%, ** 5%, * 10% (two-sided)
STARS = ((0.01, '***'), (0.05, '**'), (0.10, '*'))


def winsorize(column: pl.Expr, tail: float = 0.01) -> pl.Expr:
    """`column` clipped at its `tail` and 1 - `tail` quantiles (linear interpolation), so that values beyond
    them are set to them. Use it in a context: `df.with_columns(pl.col('a', 'b').pipe(winsorize, 0.05))`."""
    if not 0 <= tail < 0.5:
        raise ValueError(f'tail must be in [0, 0.5), got {tail}')
    return column.clip(
        column.quantile(tail, interpolation='linear'),
        column.quantile(1 - tail, interpolation='linear'),
    )


def printed_number(value: float, decimals: int) -> str:
    """A number as a paper prints it: fixed decimals, with a minus sign for negative numbers."""
    return f'{value:.{decimals}f}'.replace('-', MINUS)


def stars(p_value: float) -> str:
    """Significance stars for a two-sided p-value."""
    return next((mark for threshold, mark in STARS if p_value < threshold), '')


def paper_layout(
    fits: list,
    row_labels: dict[str, str],
    column_labels: list[str],
    dependent_labels: dict[str, str] | None = None,
) -> pl.DataFrame:
    """
    Fitted pyfixest models (`pf.feols(...)`) as papers print regression tables: one column per model
    (`column_labels`), and for each regressor in `row_labels` (pyfixest's name -> printed label, in the order
    to print) a row with the coefficient (three decimals and stars) and a row below it, with an empty label,
    with the t-statistic (two decimals); then adjusted R2 (a percentage, two decimals) and Nobs. A regressor
    not in a model leaves its cells empty. With `dependent_labels` (dependent variable -> label), a first row
    names each model's dependent variable. Every number comes from pyfixest's `tidy()` and the fitted model.
    """
    # pyfixest 0.60 has no public accessor for the dependent variable, the number of observations or
    # the adjusted R2; they are read from its attributes _depvar, _N and _adj_r2
    tidies = [fit.tidy() for fit in fits]
    rows = []
    if dependent_labels:
        rows.append(('Dependent variable', [dependent_labels[fit._depvar] for fit in fits]))
    for variable, label in row_labels.items():
        coefficients, t_statistics = [], []
        for tidy in tidies:
            if variable in tidy.index:
                coefficients.append(
                    printed_number(tidy.at[variable, 'Estimate'], 3) + stars(tidy.at[variable, 'Pr(>|t|)'])
                )
                t_statistics.append(printed_number(tidy.at[variable, 't value'], 2))
            else:
                coefficients.append('')
                t_statistics.append('')
        rows.append((label, coefficients))
        rows.append(('', t_statistics))
    rows.append(('Adj R2', [printed_number(100 * fit._adj_r2, 2) + '%' for fit in fits]))
    rows.append(('Nobs', [f'{fit._N:,}' for fit in fits]))
    return rows_to_table(rows, column_labels)


def rows_to_table(rows: list[tuple[str, list[str]]], column_labels: list[str]) -> pl.DataFrame:
    """A table of text from (row label, cells) pairs, with a first column `Model` holding the row labels."""
    return pl.DataFrame(
        {
            'Model': [label for label, _ in rows],
            **{column: [cells[i] for _, cells in rows] for i, column in enumerate(column_labels)},
        },
    )
