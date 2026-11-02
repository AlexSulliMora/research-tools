import polars as pl

from research_tools import tables


def test_number_reads_printed_numbers():
    assert tables.number('0.153***') == (0.153, 0.001)
    assert tables.number('(−3.81)') == (-3.81, 0.01)
    assert tables.number('1,211') == (1211.0, 1.0)
    assert tables.number('Yes') is None


def test_difference_level_bins():
    assert tables.difference_level('1.00', '1.004') is None   # within rounding
    assert tables.difference_level('1.00', '1.15') == 1
    assert tables.difference_level('1.00', '1.40') == 2
    assert tables.difference_level('1.00', '−1.00') == 3      # opposite signs


def test_row_keys_key_unlabelled_rows_by_the_row_above():
    df = pl.DataFrame({'Model': ['SIZE', '', 'BM', ''], 'A': ['1', '2', '3', '4']})
    assert tables.row_keys(df, 1) == [('SIZE',), ('SIZE', 1), ('BM',), ('BM', 1)]


def test_table_html_spanners_and_definitions():
    tables.DEFINITIONS['SIZE'] = 'Log market value'
    df = pl.DataFrame({'Model': ['SIZE'], 'Main | (1)': ['0.1'], 'Main | (2)': ['0.2']})
    out = tables.table_html(df)
    assert 'table-scroll' in out and 'Main' in out and 'title="Log market value"' in out


def test_assets_are_included_once():
    tables._assets_shown = False
    first, second = tables._with_assets('x'), tables._with_assets('y')
    assert first.startswith('<style>') and '.table-scroll' in first and 'compare-control' in first
    assert second == 'y'
