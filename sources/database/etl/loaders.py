from jax import numpy as jnp
from jax import jit
from jax import Array
from typing import Any
from typing import Callable
from typing import Tuple
from typing import List
from typing import Iterable

from sources.common.constants import FLOAT_PRECISION
from sources.common.types import PipelinedQueries
from sources.common.types import PipelinedResults
from sources.common.decorators import logger_decorator
from sources.common.decorators import time_decorator
from sources.common.collections import lmap

from sources.common.constants import TOKEN_TEMPLATE
from sources.database.connectors.connexions import Buffer
from sources.database.connectors.connexions_decorators import buffer_decorator
from sources.database.sql.renderers import Renderer
from sources.database.sql.renderers import renderers


def load(
    buffer: Buffer,
    cluster_sqls: PipelinedQueries,
    transform_cluster_sqls: Callable[[PipelinedResults], PipelinedQueries],
    transform_load_results: Callable[[PipelinedResults], List[Array]],
    verbose: bool = False,
) -> Callable[[str], List[Array]]:
    pqueries = transform_cluster_sqls(
        buffer.query_many(cluster_sqls[0], cluster_sqls[1])
    )
    loads = map(
        lambda p: transform_load_results(buffer.query_many(p[0], p[1])),
        pqueries,
    )
    return loads


def generate_culster_sqls(
    renderers: Renderer, affix: str, batch_size: int
) -> PipelinedQueries:
    cluster_sqls = (
        [renderers["drop_create_iterable_clusters"].render(affix, batch_size)],
        [renderers["select_iterand"].render(affix)],
    )
    return cluster_sqls


def generate_transform_cluster_sqls(
    renderers: Renderer, affix: str
) -> Callable[[PipelinedResults], PipelinedQueries]:
    def transform_cluster_sqls(results: PipelinedResults) -> PipelinedQueries:
        return lmap(
            lambda row: (
                [],
                [renderers["load_iterably"].render(affix, row[0])],
            ),
            results[0],
        )

    return transform_cluster_sqls


def generate_transform_load_results() -> (
    Callable[[PipelinedResults], List[Array]]
):
    def transform_load_results(results: PipelinedResults) -> List[Array]:
        return lmap(jnp.array, results)

    return transform_load_results
