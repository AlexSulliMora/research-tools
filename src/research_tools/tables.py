"""
Tables and charts for Quarto documents (reports and the rendered pages of pipeline scripts), in HTML
and in PDF (Typst).

    show(content)    display Markdown text, or a DataFrame as a great_tables table
    table_html(df)   a DataFrame as a great_tables table (HTML), with spanners, frozen headers and
                     frozen label columns
    table_typst(df)  the same table for the PDF, as a Typst table
    md_table(df)     a DataFrame as a Markdown pipe table
    show_comparison(replication, original)  tabs: the replication with the original's numbers next to
                     each column, shaded by how much they differ (the default); each table on its own
    show_tabs(tabs)  tables in tabs, with a "Compare with" menu that puts another tab's numbers next to
                     each column and shades the cells by how much they differ
    output_format()  the format being rendered ('html' or 'typst', the PDF)
    chart(c)         an Altair chart sized for the page: interactive in HTML, a static PNG in the PDF

In the PDF (Typst), which cannot show HTML, show(), show_comparison() and show_tabs() print Typst tables
instead, and chart() a PNG drawn with vl-convert.

The styles and the script the tables need (frozen headers, the compare menu, hover definitions) are
included in the page with the first table, so a project needs no CSS or JavaScript file of its own for
them.

Charts use one Altair theme, registered here ('report'): white background, no frame, muted axis labels
and a light grid, the legend on top, font size 11, and the colorblind-checked palette CHART_COLORS.

Column names of the form 'Spanner | Label' put Label under a spanning header; 'Top | Spanner | Label'
gives two levels of spanners. Adjacent columns with the same spanner share one header.

DEFINITIONS (row label -> definition) starts empty; a document fills it, and table rows with those labels
then show the definition when the mouse rests on the label.
"""

import html
import itertools
import json
import os
import re
from importlib.resources import files

import altair as alt
import polars as pl
import vl_convert
from great_tables import GT, loc, style
from IPython.display import HTML, Image, Markdown, display

SPAN = ' | '   # separates spanner and column labels in column names


def output_format() -> str:
    """The base format of the document Quarto is rendering ('html', 'typst', ...), from the file Quarto
    names in QUARTO_EXECUTE_INFO; 'html' outside a Quarto render (for example in VS Code)."""
    info_file = os.environ.get('QUARTO_EXECUTE_INFO')
    if not info_file:
        return 'html'
    with open(info_file, encoding='utf-8') as file:
        return json.load(file)['format']['identifier']['base-format']


def is_pdf() -> bool:
    """True when rendering the PDF (Typst)."""
    return output_format() == 'typst'


# Charts: one theme for every chart, colors in this order everywhere
CHART_COLORS = ['#2a78d6', '#eb6834', '#1baf7a']   # blue, orange, aqua (checked for colorblind readers)
TEXT_COLOR = '#0b0b0b'
MUTED_COLOR = '#52514e'
GRID_COLOR = '#e4e3df'
CHART_HEIGHT = 260
PDF_CHART_WIDTH = 620   # the PDF has no container to fill


@alt.theme.register('report', enable=True)
def report_theme() -> alt.theme.ThemeConfig:
    """White background, no frame, muted axis labels, light grid, legend on top, font size 11."""
    axis = {
        'labelColor': MUTED_COLOR,
        'titleColor': MUTED_COLOR,
        'labelFontSize': 11,
        'titleFontSize': 11,
        'titleFontWeight': 'normal',
        'gridColor': GRID_COLOR,
        'domainColor': GRID_COLOR,
        'tickColor': GRID_COLOR,
    }
    return {
        'config': {
            'background': 'white',
            'font': 'sans-serif',
            'view': {'stroke': None},
            'axis': axis,
            'legend': {
                'orient': 'top',
                'labelColor': TEXT_COLOR,
                'labelFontSize': 11,
                'title': None,
            },
            'range': {'category': CHART_COLORS},
            'text': {'color': TEXT_COLOR, 'fontSize': 11},
        },
    }


_charts = itertools.count(1)


def chart(c: alt.Chart, height: int = CHART_HEIGHT):
    """The chart as the cell's output: in HTML, interactive and as wide as the page; in the PDF, a PNG
    image drawn by vl-convert at a fixed width. The cell's `fig-cap` is its title.

    In HTML the chart needs no internet connection: the first chart of the page carries the Vega, Vega-Lite
    and Vega-Embed libraries, bundled by vl-convert (about 0.9 MB), and every chart draws itself with
    them. Both run as JavaScript modules, which the browser runs in page order."""
    if is_pdf():
        fixed_size = c.properties(width=PDF_CHART_WIDTH, height=height)
        png = vl_convert.vegalite_to_png(fixed_size.to_json(), scale=2)
        return Image(data=png, format='png')
    number = next(_charts)
    libraries = f'<script type="module">{vl_convert.javascript_bundle()}</script>' if number == 1 else ''
    spec = c.properties(width='container', height=height).to_json(indent=None)
    return HTML(
        f'{libraries}<div id="chart-{number}" style="width: 100%"></div>'
        f'<script type="module">vegaEmbed("#chart-{number}", {spec}, {{"actions": false}});</script>'
    )

# Row label -> definition, shown on hover. Empty here: each document fills it, for example
# tables.DEFINITIONS.update({'SIZE': 'Log of market value of equity.'})
DEFINITIONS: dict[str, str] = {}
LABEL_CELL = re.compile(r'(<t[hd][^>]*class="gt_row gt_left gt_stub"[^>]*>)(.*?)(</t[hd]>)', re.S)

def with_definitions(table: str) -> str:
    """Table HTML with each row label found in DEFINITIONS marked, and its definition shown on hover."""
    def mark(m: re.Match) -> str:
        definition = DEFINITIONS.get(html.unescape(m.group(2)).strip())
        if not definition:
            return m.group(0)
        return f'{m.group(1)}<span class="has-def" title="{html.escape(definition)}">{m.group(2)}</span>{m.group(3)}'
    return LABEL_CELL.sub(mark, table)

# Shading of compared cells, by how much the two numbers differ (see difference_level)
DIFF_COLORS = {1: '#fbf0c0', 2: '#f8cf8c', 3: '#f2a0a0'}
DIFF_LEGEND = ('<div class="diff-legend">Shading shows how far the two numbers differ, relative to the larger of '
               f'the two: <span style="background:{DIFF_COLORS[1]}">10-25%</span> '
               f'<span style="background:{DIFF_COLORS[2]}">25-50%</span> '
               f'<span style="background:{DIFF_COLORS[3]}">50% or more, or opposite signs</span>. Differences '
               'within the rounding of the printed numbers are not shaded.</div>')
DIFF_LEGEND_TYPST = (
    'Shading shows how far the two numbers differ, relative to the larger of the two: '
    f'#box(fill: rgb("{DIFF_COLORS[1]}"), outset: (y: 1.5pt))[10-25%] '
    f'#box(fill: rgb("{DIFF_COLORS[2]}"), outset: (y: 1.5pt))[25-50%] '
    f'#box(fill: rgb("{DIFF_COLORS[3]}"), outset: (y: 1.5pt))[50% or more, or opposite signs]. '
    'Differences within the rounding of the printed numbers are not shaded.'
)

def typst_string(text) -> str:
    """Text as a Typst string literal, which Typst prints as is (stars, #, $ and brackets included)."""
    text = '' if text is None else str(text)
    return '"' + text.replace('\\', '\\\\').replace('"', '\\"') + '"'

def spanner_groups(names: list[str], left: int, level: int) -> list[tuple[tuple[str, ...] | None, list[str]]]:
    """The value columns (after the first `left`) in runs of adjacent columns with the same path of spanners
    at `level` (0 is the spanner just above the column labels), each with that path; None where a column
    has no spanner at that level."""
    def path(c: str) -> tuple[str, ...] | None:
        parts = c.split(SPAN)
        return tuple(parts[:len(parts) - 1 - level]) if len(parts) - 1 > level else None
    return [(p, list(cols)) for p, cols in itertools.groupby(names[left:], key=path)]

def table_typst(df: pl.DataFrame, left: int = 1, fills: list[tuple[str, list[int], str]] | None = None,
                note: str = '') -> str:
    """The table_html table for the PDF, as a Typst table: labels left-aligned, values right-aligned in
    tabular figures, spanners as in table_html, horizontal rules only. fills: (column, row numbers, color)
    to shade; note: Typst markup under the table."""
    names = df.columns
    parts = {c: c.split(SPAN) for c in names}
    levels = max(len(p) for p in parts.values()) - 1
    header = []
    # Spanner rows, from the top level down; adjacent columns with the same path of spanners share a cell.
    # Each spanner's rule is its cell's bottom border, so neighbouring spanners' rules stay apart.
    for level in reversed(range(levels)):
        header.append(f'table.cell(colspan: {left}, [])')
        for p, cols in spanner_groups(names, left, level):
            if p:
                header.append(f'table.cell(colspan: {len(cols)}, align: center, stroke: (bottom: 0.5pt), '
                              f'{typst_string(p[-1])})')
            else:
                header.append(f'table.cell(colspan: {len(cols)}, [])')
    header += [typst_string(parts[c][-1]) for c in names]

    fill_of = {(c, i): color for c, rows, color in fills or [] for i in rows}
    body = []
    for i, row in enumerate(df.iter_rows()):
        for c, value in zip(names, row):
            cell = typst_string(value)
            body.append(f'table.cell(fill: rgb("{fill_of[c, i]}"), {cell})' if (c, i) in fill_of else cell)

    alignment = ', '.join(['left'] * left + ['right'] * (len(names) - left))
    # A table wider than the page gets a proportionally smaller font; the gaps and cell margins are in em, so
    # they shrink with it.
    return '\n'.join([
        '```{=typst}',
        '#block[',
        '#set text(size: 7.5pt, hyphenate: false, number-width: "tabular")',
        '#set par(justify: false)',
        '#let tbl = table(',
        f'  columns: {len(names)},',
        f'  align: ({alignment},),',
        '  stroke: none,',
        '  column-gutter: 0.5em,',
        '  inset: (x: 0.25em, y: 0.25em),',
        '  table.hline(stroke: 1pt),',
        '  table.header(' + ', '.join(header) + '),',
        '  table.hline(stroke: 0.5pt),',
        *[f'  {cell},' for cell in body],
        '  table.hline(stroke: 1pt),',
        ')',
        '#layout(page => {',
        '  let width = measure(tbl).width',
        '  set text(size: 7.5pt * calc.min(1, page.width / width))',
        '  tbl',
        '})',
        note,
        ']',
        '```',
    ])

def md_table(df: pl.DataFrame, left: int = 1) -> str:
    """Pipe table; the first `left` columns hold labels (left-aligned), the rest numbers."""
    fmt = lambda v: '' if v is None else str(v).replace('|', '\\|')
    lines = ['| ' + ' | '.join(df.columns) + ' |',
             '|' + '|'.join(':---' if i < left else '---:' for i in range(len(df.columns))) + '|']
    lines += ['| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in df.iter_rows()]
    return '\n'.join(lines)

def table_html(df: pl.DataFrame, left: int = 1, fills: list[tuple[str, list[int], str]] | None = None,
               band: list[int] | None = None, note: str = '') -> str:
    """A great_tables table as HTML: the first `left` columns hold labels (left-aligned and frozen when the
    table scrolls sideways), the rest values (right-aligned). Column names 'A | B' put B under spanner A.
    fills: (column, row numbers, color) to shade; band: rows to highlight as a block (shaded, with heavy
    rules above and below); note: HTML shown under the table."""
    names = df.columns
    ids = {c: f'c{i}' for i, c in enumerate(names)}
    data = df.with_columns(pl.all().cast(pl.String).fill_null('')).rename(ids)
    stub = ids[names[0]] if left == 1 else None
    g = GT(data, rowname_col=stub)
    if stub:
        g = g.tab_stubhead(label=names[0])

    parts = {c: c.split(SPAN) for c in names}
    g = g.cols_label(**{ids[c]: parts[c][-1] for c in names if ids[c] != stub})
    # Spanners, from the one just above the column labels (level 0) upwards
    values = names[left:]
    for level in range(max(len(p) for p in parts.values()) - 1):
        for k, (p, cols) in enumerate(spanner_groups(names, left, level)):
            if p:
                g = g.tab_spanner(label=p[-1], columns=[ids[c] for c in cols], level=level, id=f's{level}_{k}')

    g = (g.cols_align('right', columns=[ids[c] for c in values])
         .cols_align('left', columns=[ids[c] for c in names[:left]])
         .tab_options(table_font_size='0.82rem', data_row_padding='2px', column_labels_font_weight='bold',
                      table_margin_left='0', column_labels_padding='3px', table_border_top_style='hidden', container_overflow_x='visible',
                      container_overflow_y='visible', container_padding_y='0px'))
    if band:
        edge = '2px solid #333'
        g = (g.tab_style(style.fill('#e8f0fa'), [loc.body(rows=band)] + ([loc.stub(rows=band)] if stub else []))
             .tab_style(style.css(f'border-top: {edge} !important'),
                        [loc.body(rows=[band[0]])] + ([loc.stub(rows=[band[0]])] if stub else []))
             .tab_style(style.css(f'border-bottom: {edge} !important'),
                        [loc.body(rows=[band[-1]])] + ([loc.stub(rows=[band[-1]])] if stub else [])))
    for col, rows, color in fills or []:
        g = g.tab_style(style.fill(color), loc.body(columns=ids[col], rows=rows))
    return f'<div class="table-scroll" data-left="{left}">{with_definitions(g.as_raw_html())}</div>{note}'


# The styles (tables.css) and the compare menu's script, included in the page once, with the first table
_ASSETS = (
    f'<style>{files(__package__).joinpath("tables.css").read_text(encoding="utf-8")}</style>'
    f'<script>{files(__package__).joinpath("compare.js").read_text(encoding="utf-8")}</script>'
)
_assets_shown = False


def _with_assets(content: str) -> str:
    """content, preceded by the tables' styles and script the first time a table is shown on the page."""
    global _assets_shown
    if _assets_shown:
        return content
    _assets_shown = True
    return _ASSETS + content


def show(content: str | pl.DataFrame, left: int = 1, **options) -> None:
    """Show Markdown text, or a DataFrame as a table (see table_html; table_typst in the PDF)."""
    if isinstance(content, pl.DataFrame) and is_pdf():
        display(Markdown(table_typst(content, left)))
    elif isinstance(content, pl.DataFrame):
        display(HTML(_with_assets(table_html(content, left, **options))))
    else:
        display(Markdown(content))

# Comparing two tables of the same layout

NUMBER = re.compile(r'^\s*\(?\$?([-−]?)\$?(\d[\d,]*)(?:\.(\d+))?')

def number(cell: str | None) -> tuple[float, float] | None:
    """The number at the start of a cell ('0.153***', '1,211', '10.59%', '(-3.81)', '$1,234.50') and the
    rounding unit of its printed decimals; None for cells that do not start with a number ('Yes', '')."""
    m = NUMBER.match(cell or '')
    if not m:
        return None
    sign, whole, decimals = m.groups()
    value = float(whole.replace(',', '') + ('.' + decimals if decimals else ''))
    return (-value if sign else value), 10.0 ** -len(decimals or '')

def difference_level(a: str, b: str) -> int | None:
    """How much two printed numbers differ: None within the rounding of the printed numbers or under 10%
    of the larger absolute value; 1 for 10-25%, 2 for 25-50%, 3 for 50% or more or opposite signs."""
    x, y = number(a), number(b)
    if x is None or y is None:
        return None
    (x, ux), (y, uy) = x, y
    if abs(x - y) <= max(ux, uy):
        return None
    if x * y < 0:
        return 3
    rel = abs(x - y) / max(abs(x), abs(y))
    return 3 if rel >= 0.5 else 2 if rel >= 0.25 else 1 if rel >= 0.10 else None

def row_keys(df: pl.DataFrame, left: int) -> list[tuple]:
    """A key for each row, from its label columns. A row with empty labels (such as the t-statistics printed
    beneath a coefficient) is keyed by the last labelled row above it and its position after that row."""
    keys, last, k = [], (), 0
    for row in df.select(df.columns[:left]).iter_rows():
        if any(row):
            last, k = row, 0
            keys.append(row)
        else:
            k += 1
            keys.append((*last, k))
    return keys

def show_comparison(replication: pl.DataFrame, original: pl.DataFrame, note: str = '', left: int = 1) -> None:
    """Tabs Comparison (the default), Replication and Original, then `note`. The comparison is the
    replication with the original's number for the same row and column next to each of its columns
    (labelled Rep and Orig), the Orig column shaded by difference_level; the PDF shows only it. Rows are matched
    on their labels (row_keys); columns by position when the two tables have the same number of columns,
    else by name."""
    df, ref = (d.with_columns(pl.all().cast(pl.String)) for d in (replication, original))
    ref_col = (dict(zip(df.columns, ref.columns)) if df.width == ref.width else {c: c for c in ref.columns})
    ref_rows = dict(zip(row_keys(ref, left), ref.iter_rows(named=True)))
    out, fills = {c: df[c].to_list() for c in df.columns[:left]}, {}
    for i, (key, row) in enumerate(zip(row_keys(df, left), df.iter_rows(named=True))):
        other = ref_rows.get(key, {})
        for c in df.columns[left:]:
            here, there = f'{c}{SPAN}Rep', f'{c}{SPAN}Orig'
            v, w = row[c], other.get(ref_col.get(c))
            out.setdefault(here, []).append(v)
            out.setdefault(there, []).append(w)
            if (level := difference_level(v, w)) is not None:
                fills.setdefault((there, level), []).append(i)
    compared = pl.DataFrame(out)
    shading = [(c, r, DIFF_COLORS[level]) for (c, level), r in fills.items()]
    if is_pdf():
        display(Markdown(table_typst(compared, left, shading, note=DIFF_LEGEND_TYPST)
                         + (f'\n\n{note}' if note else '')))
        return
    # Tabs, the comparison first (shown by default)
    tabs = {
        'Comparison': table_html(compared, left, fills=shading, note=DIFF_LEGEND),
        'Replication': table_html(replication, left),
        'Original': table_html(original, left),
    }
    tabs['Comparison'] = _with_assets(tabs['Comparison'])
    panes = ''.join(f'#### {label}\n\n```{{=html}}\n{table}\n```\n\n' for label, table in tabs.items())
    display(Markdown(f'::: {{.panel-tabset}}\n\n{panes}:::\n\n{note}'))


def compare_html(df: pl.DataFrame, ref: pl.DataFrame, name: str, ref_name: str, left: int = 1) -> str:
    """df (the tab called name) with ref's number for the same row and column next to each of its columns,
    the ref column shaded by difference_level. Rows are matched on their labels (row_keys); columns by
    position when the two tables have the same number of columns, else by name."""
    df, ref = (d.with_columns(pl.all().cast(pl.String)) for d in (df, ref))
    ref_col = (dict(zip(df.columns, ref.columns)) if df.width == ref.width else {c: c for c in ref.columns})
    ref_rows = dict(zip(row_keys(ref, left), ref.iter_rows(named=True)))
    out, fills = {c: df[c].to_list() for c in df.columns[:left]}, {}
    for i, (key, row) in enumerate(zip(row_keys(df, left), df.iter_rows(named=True))):
        other = ref_rows.get(key, {})
        for c in df.columns[left:]:
            here, there = f'{c}{SPAN}{name}', f'{c}{SPAN}{ref_name}'
            v, w = row[c], other.get(ref_col.get(c))
            out.setdefault(here, []).append(v)
            out.setdefault(there, []).append(w)
            if (level := difference_level(v, w)) is not None:
                fills.setdefault((there, level), []).append(i)
    return table_html(pl.DataFrame(out), left,
                      fills=[(c, r, DIFF_COLORS[level]) for (c, level), r in fills.items()], note=DIFF_LEGEND)

_groups = itertools.count(1)

def show_tabs(tabs: list[tuple[str, pl.DataFrame | str | None, str]], left: int = 1) -> None:
    """Print tables in tabs (label, table, note under the table), with a "Compare with" menu above them:
    choosing a tab puts that tab's numbers next to each column of the others, shaded by how much they
    differ; the tab labels head the two columns, so they should be short. A tab whose content is text
    (Markdown) is shown as it is and takes no part in comparisons. The cell must have `output: asis`, so
    that the printed Markdown and HTML become part of the report."""
    tabs = [t for t in tabs if t[1] is not None]
    if is_pdf():   # no menus in the PDF: each tab's table under its label
        for label, tbl, note in tabs:
            body = table_typst(tbl, left) if isinstance(tbl, pl.DataFrame) else tbl
            print(f'\n\n**{label}**\n\n{body}\n\n{note}\n\n')
        return
    tables = {j: (label, tbl) for j, (label, tbl, _) in enumerate(tabs) if isinstance(tbl, pl.DataFrame)}
    group = f'compare-{next(_groups)}'
    options = ''.join(f'<option value="{j}">{label}</option>' for j, (label, _) in tables.items())
    control = (f'```{{=html}}\n<div class="compare-control"><label for="{group}">Compare with</label> '
               f'<select id="{group}"><option value="none">nothing</option>{options}</select></div>\n```\n\n'
               if len(tables) > 1 else '')
    panes = []
    for i, (label, tbl, note) in enumerate(tabs):
        if not isinstance(tbl, pl.DataFrame):
            panes.append(f'#### {label}\n\n{tbl}\n\n{note}')
            continue
        variants = [f'<div class="cmp-variant" data-cmp="none">{table_html(tbl, left)}</div>']
        variants += [f'<div class="cmp-variant" data-cmp="{j}" hidden>'
                     f'{compare_html(tbl, ref, label, ref_label, left)}</div>'
                     for j, (ref_label, ref) in tables.items() if j != i]
        panes.append(f'#### {label}\n\n```{{=html}}\n{"".join(variants)}\n```\n\n{note}')
    control = f'```{{=html}}\n{_with_assets("")}\n```\n\n' + control
    print(f'\n\n::: {{.compare-group}}\n\n{control}::: {{.panel-tabset}}\n\n'
          + '\n\n'.join(panes) + '\n\n:::\n\n:::\n\n')
