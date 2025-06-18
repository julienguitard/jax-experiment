from typing import List
from sources.common.collections import lmap
from sources.common.collections import lfilter
from sources.common.collections import dmap
from sources.common.collections import dfilter
from sources.common.strings import format_partially
from sources.common.decorators import repeat_decorator
from sources.common.decorators import logger_decorator
from sources.database.connectors.connexions import Buffer
from sources.database.sql.renderers import renderers


def list_main_tables_from_sql(sql: str) -> List[str]:
    """
    Returns a list of main tables from a template.

    Args:
        template (str): The template used to identify the tables.

    Returns:
        List[str]: A list of strings representing the main tables.
    """
    return lmap(
        lambda s: s.split(" ")[0],
        lfilter(lambda s: "(" in s, sql.split("CREATE TABLE ")),
    )


def transform(
    buffer: Buffer,
    affix: str,
    source: str,
    epoch: int,
    overwrite_tables: bool,
    verbose=False,
) -> None:
    actual_tables = buffer.list_tables()
    expected_tables = list_main_tables_from_sql(
        renderers["create_main_tables"].render(affix)
    )
    existing = not (False in [x in actual_tables for x in expected_tables])
    sqls = [
        ("drop_create_buffer_tables", {"args": [affix], "condition": True}),
        (
            "drop_main_tables",
            {"args": [affix], "condition": (overwrite_tables or not existing)},
        ),
        (
            "create_main_tables",
            {"args": [affix], "condition": (overwrite_tables or not existing)},
        ),
        (
            "merge_into_tables",
            {"args": [affix, epoch, source], "condition": True},
        ),
        ("update_new_", {"args": [affix], "condition": True}),
    ]

    ddl_queries = lmap(
        lambda s: renderers[s[0]].render(*s[1]["args"]),
        lfilter(lambda s: s[1]["condition"], sqls),
    )
    try:
        _ = buffer.query_many(ddl_queries, [], verbose=verbose)
    except Exception as e:
        print(e)
