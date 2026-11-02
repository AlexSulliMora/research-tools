"""
Helpers shared across research projects:

    research_tools.wrds_db       WRDS's PostgreSQL database read into Polars (one connection, batches,
                                 types from the database)
    research_tools.tables        tables and charts for Quarto documents (HTML and PDF)
    research_tools.regressions   OLS with pyfixest, and regression tables in a paper's layout

Import the module you need, e.g. `from research_tools import wrds_db, tables`.
"""

__version__ = '0.1.0'
