"""
OLS regressions with pyfixest, and their results as a table in the layout of finance and accounting papers.

    winsorize(sample, columns)     each column clipped at its 1st and 99th percentiles in the sample
    ModelResult                    one model's coefficients, standard errors, t-statistics, p-values,
                                   adjusted R2 and number of observations
    estimate(sample, ...)          one OLS regression with pyfixest, standard errors clustered two ways
    paper_layout(models, ...)      the models as papers print them: a coefficient with stars, the
                                   t-statistic on the row below, then adjusted R2 and Nobs

pyfixest returns its results as pandas objects; they are read only to fill ModelResult with Python
numbers, and no data pass through pandas.
"""

from dataclasses import dataclass

import polars as pl
import pyfixest as pf

MINUS = '−'   # negative numbers are printed with a minus sign, not a hyphen

# Significance stars: *** 1%, ** 5%, * 10% (two-sided)
STARS = ((0.01, '***'), (0.05, '**'), (0.10, '*'))


def winsorize(
    sample: pl.DataFrame,
    columns: list[str],
    lower: float = 0.01,
    upper: float = 0.99,
) -> pl.DataFrame:
    """Each of `columns` clipped at its `lower` and `upper` quantiles in `sample` (linear
    interpolation), so that values beyond them are set to them."""
    return sample.with_columns(
        pl.col(column).clip(
            pl.col(column).quantile(lower, interpolation='linear'),
            pl.col(column).quantile(upper, interpolation='linear'),
        )
        for column in columns
    )


@dataclass
class ModelResult:
    """One estimated model, with the numbers a regression table prints."""
    name: str
    dependent: str
    coefficients: dict[str, float]
    standard_errors: dict[str, float]
    t_statistics: dict[str, float]
    p_values: dict[str, float]
    adjusted_r2: float
    nobs: int


def estimate(
    sample: pl.DataFrame,
    dependent: str,
    regressors: list[str],
    name: str,
    clusters: tuple[str, ...] = ('gvkey', 'year'),
) -> ModelResult:
    """
    OLS of `dependent` on an intercept and `regressors`, with standard errors clustered by each of
    `clusters` (by default two-way, by firm and by calendar year, following Gow et al. 2012) and
    pyfixest's default small-sample adjustment. The estimation sample is decided here, in Polars: the rows
    of `sample` with every model variable finite. The model's number of observations is checked against it.
    """
    variables = [dependent, *regressors]
    model_sample = sample.filter(
        pl.all_horizontal(
            pl.col(variable).cast(pl.Float64).is_finite().fill_null(False) for variable in variables
        ),
        pl.all_horizontal(pl.col(cluster).is_not_null() for cluster in clusters),
    )
    fit = pf.feols(
        f'{dependent} ~ {" + ".join(regressors)}',
        data=model_sample,
        vcov={'CRV1': ' + '.join(clusters)},
    )
    # pyfixest 0.60 has no public accessor for the number of observations or the adjusted R2; they are read
    # from its attributes _N and _adj_r2
    if fit._N != model_sample.height:
        raise ValueError(
            f'{name}: the model used {fit._N:,} observations of the {model_sample.height:,} passed'
        )

    def as_dict(series) -> dict[str, float]:
        return {str(key): float(value) for key, value in series.items()}

    return ModelResult(
        name=name,
        dependent=dependent,
        coefficients=as_dict(fit.coef()),
        standard_errors=as_dict(fit.se()),
        t_statistics=as_dict(fit.tstat()),
        p_values=as_dict(fit.pvalue()),
        adjusted_r2=float(fit._adj_r2),
        nobs=int(fit._N),
    )


def printed_number(value: float, decimals: int) -> str:
    """A number as a paper prints it: fixed decimals, with a minus sign for negative numbers."""
    return f'{value:.{decimals}f}'.replace('-', MINUS)


def stars(p_value: float) -> str:
    """Significance stars for a two-sided p-value."""
    return next((mark for threshold, mark in STARS if p_value < threshold), '')


def paper_layout(
    models: list[ModelResult],
    row_labels: dict[str, str],
    column_labels: list[str],
    dependent_labels: dict[str, str] | None = None,
) -> pl.DataFrame:
    """
    The models as papers print regression tables: one column per model (`column_labels`), and for
    each regressor in `row_labels` (pyfixest's name -> printed label, in the order to print) a row with the
    coefficient (three decimals and stars) and a row below it, with an empty label, with the t-statistic
    (two decimals); then adjusted R2 (a percentage, two decimals) and Nobs. A regressor not in a model
    leaves its cells empty. With `dependent_labels` (dependent variable -> label), a first row names each
    model's dependent variable.
    """
    rows = []
    if dependent_labels:
        rows.append(('Dependent variable', [dependent_labels[model.dependent] for model in models]))
    for variable, label in row_labels.items():
        coefficients, t_statistics = [], []
        for model in models:
            if variable in model.coefficients:
                coefficients.append(
                    printed_number(model.coefficients[variable], 3) + stars(model.p_values[variable])
                )
                t_statistics.append(printed_number(model.t_statistics[variable], 2))
            else:
                coefficients.append('')
                t_statistics.append('')
        rows.append((label, coefficients))
        rows.append(('', t_statistics))
    rows.append(('Adj R2', [printed_number(100 * model.adjusted_r2, 2) + '%' for model in models]))
    rows.append(('Nobs', [f'{model.nobs:,}' for model in models]))
    return rows_to_table(rows, column_labels)


def rows_to_table(rows: list[tuple[str, list[str]]], column_labels: list[str]) -> pl.DataFrame:
    """A table of text from (row label, cells) pairs, with a first column `Model` holding the row labels."""
    return pl.DataFrame(
        {
            'Model': [label for label, _ in rows],
            **{column: [cells[i] for _, cells in rows] for i, column in enumerate(column_labels)},
        },
    )
