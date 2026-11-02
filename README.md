# research-tools

Helpers shared across research projects. Each project installs this package with pixi and imports the module
it needs:

| Module | What it does |
|---|---|
| `research_tools.wrds_db` | Reads WRDS's PostgreSQL database into Polars over one connection (one two-factor push), with types from the database, in batches written straight to Parquet. `pull_tables(queries, folder)` pulls every missing file, and only connects when one is missing. |
| `research_tools.tables` | Tables and charts for Quarto documents, in HTML and PDF (Typst): `show(df)`, spanning headers (`'Spanner \| Label'` column names), frozen headers and labels, hover definitions (`DEFINITIONS`), tabs with a "Compare with" menu (`show_tabs`), replication-versus-original comparisons (`show_comparison`), and an Altair theme (`chart`). |
| `research_tools.regressions` | OLS with pyfixest and clustered standard errors (`estimate`), winsorizing (`winsorize`), and regression tables in a paper's layout (`paper_layout`). |

## Use it in a project

In the project's `pixi.toml`, pin a commit or a tag:

```toml
[pypi-dependencies]
research-tools = { git = "https://github.com/AlexSulliMora/research-tools", rev = "<commit or tag>" }
```

then run `pixi install`. In code:

```python
from research_tools import tables, wrds_db
from research_tools.tables import show

tables.DEFINITIONS.update({'SIZE': 'Log of market value of equity.'})
```

The tables include their own styles and script in the page (with the first table), so a project needs no CSS
or JavaScript file for them.

While changing this package alongside a project, point the project at a local copy instead, so edits take effect
without a new commit:

```toml
research-tools = { path = "../research-tools", editable = true }
```

The repository is private, so each machine that installs it needs git signed in to GitHub (`gh auth login`).

## Releasing a version

1. Change the code, run the tests (`pip install -e '.[test]'` then `pytest`, or in a pixi project
   `pixi run python -m pytest`).
2. Raise `version` in `pyproject.toml` and `__version__` in `src/research_tools/__init__.py`.
3. Commit, then tag: `git tag v0.2.0 && git push --tags`.
4. In each project, change `rev` to the new tag when you want the new version, and run `pixi install`.

## WRDS access

Every new connection can send a two-factor push to the account holder's phone, and repeated attempts can lock
the account. `wrds_db.connect()` makes one attempt and never retries; credentials come from `~/.pgpass`, and the
password is never printed.
