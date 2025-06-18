import argparse
import os
import time
import json
from typing import Any
from typing import Callable
from typing import List
from typing import Dict
from typing import Tuple
from typing import TypeVar
from typing import Union

from sources.common.types import ComplexValue
from sources.common.constants import TOKEN_TEMPLATE
from sources.common.constants import DATA_SOURCE
from sources.common.constants import PERSISTENT_DATA_PATH
from sources.common.collections import lmap
from sources.common.collections import lmapzip as lmz
from sources.common.collections import lfilter
from sources.common.collections import dmap
from sources.common.collections import dfilter
from sources.common.strings import format_partially
from sources.common.decorators import repeat_decorator
from sources.common.decorators import logger_decorator
from sources.database.connectors.connexions import Buffer
from sources.database.connectors.connexions_decorators import buffer_decorator

from sources.database.etl.extractors import download_data
from sources.database.etl.transformers import transform
from sources.database.sql.renderers import renderers
from sources.database.sql.renderers import merge_predicates


def download_(affix: str, epoch: int) -> None:
    """
    Downloads data from a specified data source and saves it in a persistent data path.

    Parameters:
        affix (str): A string representing the affix used in the file name.
        epoch (int): An integer representing the epoch time.

    Returns:
        None
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(epoch))
    print("Data sources:", DATA_SOURCE, timestamp)
    download_data(
        DATA_SOURCE, PERSISTENT_DATA_PATH, "{}_{}.csv".format(affix, epoch)
    )


def transform_(affix: str, sources: List[str], overwrite_file: bool) -> None:
    sources_epochs_overwrites = lmz(
        lambda i, s: (
            s,
            int(s.split(".csv")[0].split("_")[-1]),
            (i == 0) and overwrite_file,
        ),
        enumerate(sources),
    )

    @buffer_decorator
    def execute(b):
        _ = lmz(
            lambda e, s, ou: logger_decorator(transform)(
                b, affix, e, s, ou, verbose=False
            ),
            sources_epochs_overwrites,
        )

    _ = execute()


@repeat_decorator(times=288, seconds=300)
def download_repeatedly_(affix: str) -> None:
    epoch = int(time.time())
    download_(affix, epoch)


if __name__ == "_main__":
    parser = argparse.ArgumentParser(description="do some transforms")
    arg_types = {"affix": str, "overwrite_file": str}
    _ = lmz(lambda k, v: parser.add_argument(k, type=v), arg_types.items())
    args = parser.parse_args()
    affix = args.affix
    overwrite_file = args.overwrite_file == "t"
    sources = lmap(
        lambda s: "{}{}".format(PERSISTENT_DATA_PATH, s),
        lfilter(
            lambda f: ("{}_{}".format(affix, "buffer") in f and ".csv" in f),
            os.listdir(PERSISTENT_DATA_PATH),
        ),
    )
    _ = transform_(affix, sources, overwrite_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="do some transforms")
    arg_types = {"affix": str}
    _ = lmz(lambda k, v: parser.add_argument(k, type=v), arg_types.items())
    args = parser.parse_args()
    affix = args.affix
    _ = download_repeatedly_(affix)
