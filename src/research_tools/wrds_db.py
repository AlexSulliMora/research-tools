"""
WRDS's PostgreSQL database, read into Polars through the ADBC driver: a replacement for the
`wrds` package (which reads through pandas and loads a whole result into memory at once).

    connect()                        one connection, from the credentials in ~/.pgpass
    close()                          close every connection connect() opened (and their queries)
    list_libraries(conn)             the libraries (database schemas), e.g. crsp, comp
    list_tables(conn, library)       the tables and views in a library
    describe_table(conn, lib, tbl)   each column's database type and the Polars type it is read as
    read(conn, query)                a query's result as one DataFrame, or with iter_batches=True,
                                     as batches of DataFrames as they arrive
    pull_to_parquet(conn, query, f)  a query's result written to one Parquet file, batch by batch
    pull_tables({name: query}, dir)  several queries saved as <dir>/<name>.parquet, over one
                                     connection, opened only if some file is missing (or is
                                     named in overwrite); with keep_going=True, a failed query
                                     is recorded and the others still run; session_sql runs
                                     settings (e.g. a query time limit) before the pulls

For example:

    from research_tools import wrds_db
    conn = wrds_db.connect()
    funda = wrds_db.read(conn, "select gvkey, datadate, at from comp.funda where fyear = 2020")
    wrds_db.close()

WRDS asks that connections be closed: an open connection stays open on their server until
the program ends. close() closes every connection connect() opened, after ending any query
still running on it, and it runs on its own when Python exits. A `with wrds_db.connect() as
conn:` block also closes its connection at the end of the block. Each query's cursor is closed
when its result has been read, when the query fails, and when a loop over its batches
stops early.

Every connection can send a two-factor push to the account holder's phone, and repeated
attempts can lock the account. So connect() makes one attempt and never retries, and
pull_tables() connects only when a table is missing. The password is never printed: it is
removed from any error message.

Data arrive as the database stores them, and every change is left to the user:
- Types come from the database itself (each column's type travels with the data), never from
  guessing at the values. Integers keep their width (int2, int4, int8 become Int16, Int32,
  Int64), so widen them before arithmetic that could exceed the width.
- Missing values stay null. Float columns can also hold NaN and infinity, which stay NaN and
  infinity, distinct from null.
- `numeric` columns become Float64 (about 15-16 significant digits); NaN and infinity in them
  stay NaN and infinity.
- Types the driver cannot convert (e.g. geometric types) stop the read before any data are
  fetched; with convert_dtypes=True, they are read as the database's text form instead.
- Dates and timestamps of 'infinity' or '-infinity' (PostgreSQL allows them) arrive as wrong
  dates millions of years in the past, and Polars can fail to print them. They are not checked.
- Rows arrive in the order the query sets: add `order by` for a reproducible file.
"""

import atexit
import os
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from urllib.parse import quote

import adbc_driver_postgresql.dbapi as adbc
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from adbc_driver_postgresql import StatementOptions

WRDS_HOST = 'wrds-pgdata.wharton.upenn.edu'
BATCH_BYTES = 512_000_000       # target size of each batch from the driver (its estimate is rough)
ROW_GROUP_SIZE = 1_000_000      # rows per Parquet row group

_connections: list[adbc.Connection] = []   # opened by connect(), closed by close()
_cursors: set[adbc.Cursor] = set()          # queries whose results are still being read


#%% Connecting

def _pgpass_file() -> Path:
    """Where PostgreSQL keeps saved passwords: PGPASSFILE if set, else ~/.pgpass (on Windows,
    %APPDATA%/postgresql/pgpass.conf)."""
    if 'PGPASSFILE' in os.environ:
        return Path(os.environ['PGPASSFILE'])
    if os.name == 'nt':
        return Path(os.environ['APPDATA']) / 'postgresql' / 'pgpass.conf'
    return Path.home() / '.pgpass'


def _credentials(host: str) -> tuple[str, str, str, str]:
    """Port, database, username and password from the password file's line for this host.
    Fields are separated by colons; a colon or backslash inside a field is written \\: or \\\\."""
    path = _pgpass_file()
    for line in path.read_text(encoding='utf-8').splitlines():
        fields = re.split(r'(?<!\\):', line, maxsplit=4)
        if len(fields) == 5 and fields[0] == host:
            port, database, username, password = (re.sub(r'\\(.)', r'\1', f) for f in fields[1:])
            return port, database, username, password
    raise SystemExit(f'No line for {host} in {path}. Add one: {host}:9737:wrds:<username>:<password>')


def connect(host: str = WRDS_HOST, sslmode: str = 'require') -> adbc.Connection:
    """One connection to WRDS; close it with close(). If the connection fails, the error is
    raised: there is no second attempt."""
    port, database, username, password = _credentials(host)
    uri = f'postgresql://{username}:{quote(password, safe="")}@{host}:{port}/{database}?sslmode={sslmode}'
    print(f'Connecting to {host} as {username} (this can send a two-factor push).', flush=True)
    try:
        conn = adbc.connect(uri, autocommit=True)   # each query on its own, as the wrds package does
    except Exception as err:
        message = str(err).replace(quote(password, safe=''), '***').replace(password, '***')
        raise ConnectionError(f'Connection to {host} failed; not retrying. {message}') from None
    _connections.append(conn)
    return conn


def close() -> None:
    """Close every connection connect() opened, first ending any query whose result is still
    being read. Closing an already closed connection does nothing."""
    for cursor in list(_cursors):
        _close_cursor(cursor)
    while _connections:
        _connections.pop().close()


atexit.register(close)


def _close_cursor(cursor: adbc.Cursor) -> None:
    """Close a query's cursor (ending the query if it is still running) and stop tracking it."""
    _cursors.discard(cursor)
    cursor.close()


#%% Types

def _unconvertible(schema: pa.Schema) -> list[str]:
    """Columns of types the driver passes on as raw bytes, because it cannot convert them."""
    return [
        field.name
        for field in schema
        if isinstance(field.type, pa.BaseExtensionType) and pa.types.is_binary(field.type.storage_type)
    ]


def _numeric(schema: pa.Schema) -> list[str]:
    """Columns of PostgreSQL's numeric type, which the driver sends as text."""
    return [
        field.name
        for field in schema
        if field.metadata and field.metadata.get(b'POSTGRESQL:type') == b'numeric'
    ]


def _storage_type(arrow_type: pa.DataType) -> pa.DataType:
    """An extension type (numeric as text, json) as its plain storage type."""
    return arrow_type.storage_type if isinstance(arrow_type, pa.BaseExtensionType) else arrow_type


def _storage(column: pa.Array) -> pa.Array:
    """An extension column (numeric as text, json) as its plain storage column."""
    return column.storage if isinstance(column, pa.ExtensionArray) else column


def _polars_schema(schema: pa.Schema) -> pl.Schema:
    """The Polars schema a result converts to: the driver's types, numeric columns as Float64."""
    empty = pl.from_arrow(pa.table({
        field.name: pa.array([], type=_storage_type(field.type))
        for field in schema
    }))
    numeric = _numeric(schema)
    return pl.Schema({
        name: pl.Float64 if name in numeric else dtype
        for name, dtype in empty.schema.items()
    })


def _to_polars(batch: pa.RecordBatch, numeric: list[str]) -> pl.DataFrame:
    """One batch from the driver as a DataFrame, numeric columns as Float64."""
    df = pl.from_arrow(pa.table({
        name: _storage(column)
        for name, column in zip(batch.schema.names, batch.columns)
    }))
    return df.with_columns(pl.col(numeric).cast(pl.Float64))


def _as_text(query: str, names: list[str], to_text: list[str]) -> str:
    """The query, with the columns in to_text converted to text by the database."""
    quoted = {name: '"' + name.replace('"', '""') + '"' for name in names}
    columns = ', '.join(
        f'q.{quoted[name]}::text as {quoted[name]}' if name in to_text else f'q.{quoted[name]}'
        for name in names
    )
    return f'select {columns} from ({query.strip().rstrip(";")}) as q'


#%% Reading

def _execute(
    conn: adbc.Connection,
    query: str,
    convert_dtypes: bool,
    batch_bytes: int,
) -> tuple[pl.Schema, Iterator[pl.DataFrame]]:
    """Run a query: the Polars schema of its result (known before any rows arrive), and its batches.
    The database first describes the result without running the query, so that columns of types the
    driver cannot convert stop the read, or with convert_dtypes are converted to text, before any
    data are fetched."""
    cursor = conn.cursor()
    _cursors.add(cursor)
    try:
        cursor.adbc_statement.set_options(**{StatementOptions.BATCH_SIZE_HINT_BYTES.value: str(batch_bytes)})
        described = cursor.adbc_execute_schema(query)
        if unconvertible := _unconvertible(described):
            types = ', '.join(f'{name} ({described.field(name).metadata[b"POSTGRESQL:type"].decode()})'
                              for name in unconvertible)
            if not convert_dtypes:
                raise ValueError(f'Columns of types the driver cannot convert: {types}. Pass '
                                 'convert_dtypes=True to read them as text.')
            print(f'Reading as text: {types}', flush=True)
            query = _as_text(query, described.names, unconvertible)
        cursor.execute(query)
        reader = cursor.fetch_record_batch()
        schema, numeric = _polars_schema(reader.schema), _numeric(reader.schema)
    except BaseException:
        _close_cursor(cursor)
        raise

    def batches() -> Iterator[pl.DataFrame]:
        # The cursor closes when the batches run out, when converting one fails, and when the
        # loop reading them stops early (Python then closes this generator)
        try:
            for batch in reader:
                yield _to_polars(batch, numeric)
        finally:
            _close_cursor(cursor)

    return schema, batches()


def read(
    conn: adbc.Connection,
    query: str,
    *,
    iter_batches: bool = False,
    convert_dtypes: bool = False,
    batch_bytes: int = BATCH_BYTES,
) -> pl.DataFrame | Iterator[pl.DataFrame]:
    """The result of a query as one DataFrame. With iter_batches=True, an iterator of DataFrames
    of about batch_bytes each, converted as they arrive, so that memory holds one at a time (to
    write a large result to a file, use pull_to_parquet). convert_dtypes=True reads columns of
    types the driver cannot convert as text, rather than stopping."""
    schema, batches = _execute(conn, query, convert_dtypes, batch_bytes)
    if iter_batches:
        return batches
    return pl.concat([pl.DataFrame(schema=schema), *batches])


def pull_to_parquet(
    conn: adbc.Connection,
    query: str,
    out_file: Path,
    *,
    convert_dtypes: bool = False,
    batch_bytes: int = BATCH_BYTES,
    row_group_size: int = ROW_GROUP_SIZE,
) -> None:
    """Write the result of a query to one Parquet file (zstd), batch by batch, so that memory holds
    one batch at a time. The file is written under a temporary name and renamed at the end, so an
    interrupted pull leaves no file that looks complete. convert_dtypes: as in read()."""
    out_file = Path(out_file)
    part_file = out_file.with_name(out_file.name + '.part')
    schema, batches = _execute(conn, query, convert_dtypes, batch_bytes)
    rows = 0
    try:
        arrow_schema = pl.DataFrame(schema=schema).to_arrow().schema
        with pq.ParquetWriter(part_file, arrow_schema, compression='zstd') as writer:
            for df in batches:
                writer.write_table(df.to_arrow(), row_group_size=row_group_size)
                rows += df.height
                print(f'  {out_file.name}: {rows:_} rows', flush=True)
    except BaseException:
        part_file.unlink(missing_ok=True)
        raise
    finally:
        batches.close()   # ends the query now if writing failed partway
    part_file.replace(out_file)
    print(f'Saved {out_file} ({rows:_} rows)', flush=True)


def pull_tables(
    queries: dict[str, str],
    out_dir: Path,
    *,
    overwrite: bool | str | Iterable[str] = False,
    keep_going: bool = False,
    session_sql: Iterable[str] = (),
    **options,
) -> dict[str, str]:
    """Save each query's result as <out_dir>/<name>.parquet, skipping files already saved. One
    connection serves all the pulls, and there is none if nothing needs pulling.
    overwrite: which saved files to pull again: False (none), True (all), or a query's name or a
    list of names. A name that matches no query stops before connecting. A file pulled again is
    replaced only once the new one is complete, so a failed pull keeps the old file.
    keep_going: if False, the first failed query raises its error and the later queries do not
    run. If True, a failed query's error is printed and recorded, and the later queries still
    run over the same connection (no file is saved for the failed one, and nothing is retried).
    A failed connection, and an interruption (Ctrl+C), stop the pulls either way.
    session_sql: statements run on the connection, in order, before the pulls, for example
    "select current_setting('statement_timeout')" and "set statement_timeout = '4h'" (a limit on each query's run
    time, including sending its rows). Each one's result, if it has one, is printed. A failed
    statement is printed and the pulls still run (it is not in the returned errors).
    options go to pull_to_parquet (convert_dtypes, batch_bytes, row_group_size).
    Returns the error message of each failed query, by name (empty if none failed)."""
    out_dir = Path(out_dir)
    if isinstance(overwrite, bool):
        again = set(queries) if overwrite else set()
    else:
        again = {overwrite} if isinstance(overwrite, str) else set(overwrite)
        if unknown := again - set(queries):
            not_found = ', '.join(sorted(unknown))
            raise ValueError(f'overwrite values must be included in queries. Not found: {not_found}')
    to_pull = {
        name: query
        for name, query in queries.items()
        if name in again or not (out_dir / f'{name}.parquet').exists()
    }
    if not to_pull:
        print(f'All {len(queries)} tables already saved in {out_dir}; not connecting to WRDS.')
        return {}
    print(f'To pull: {", ".join(to_pull)}', flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    failed: dict[str, str] = {}
    with connect() as conn:
        _run_session_sql(conn, session_sql)
        for name, query in to_pull.items():
            try:
                pull_to_parquet(conn, query, out_dir / f'{name}.parquet', **options)
            except Exception as err:   # not KeyboardInterrupt, which stops the pulls
                if not keep_going:
                    raise
                failed[name] = f'{type(err).__name__}: {err}'
                print(f'Failed: {name}. {failed[name]}', flush=True)
    if failed:
        print(f'{len(failed)} of {len(to_pull)} pulls failed: {", ".join(failed)}', flush=True)
    return failed


def _run_session_sql(conn: adbc.Connection, statements: Iterable[str]) -> None:
    """Run each statement on the connection, printing its result rows (if any) or its error.
    A failed statement does not stop the ones after it."""
    for sql in statements:
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall() if cursor.description else []
            print(f'{sql}: {rows}' if rows else f'{sql}: done', flush=True)
        except Exception as err:
            print(f'Failed (the pulls go on): {sql}. {type(err).__name__}: {err}', flush=True)


#%% Exploring the database

def list_libraries(conn: adbc.Connection) -> pl.DataFrame:
    """The libraries (database schemas) visible to this account."""
    return read(conn, 'select schema_name as library from information_schema.schemata order by 1')


def list_tables(conn: adbc.Connection, library: str) -> pl.DataFrame:
    """The tables and views in a library."""
    library = library.replace("'", "''")
    return read(
        conn,
        f"""
        select table_name, table_type
        from information_schema.tables
        where table_schema = '{library}'
        order by table_name
        """,
    )


def describe_table(
    conn: adbc.Connection,
    library: str,
    table: str,
) -> pl.DataFrame:
    """Each column's database type and the Polars type read() gives it, from the database's
    description of the table (no rows are read). Columns of types the driver cannot convert show
    'String (with convert_dtypes=True)'."""
    with conn.cursor() as cursor:
        schema = cursor.adbc_execute_schema(f'select * from {library}.{table}')
    unconvertible = _unconvertible(schema)
    return pl.DataFrame({
        'column': schema.names,
        'database_type': [field.metadata[b'POSTGRESQL:type'].decode() for field in schema],
        'polars_type': [
            'String (with convert_dtypes=True)' if name in unconvertible else str(dtype)
            for name, dtype in _polars_schema(schema).items()
        ],
    })
