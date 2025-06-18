import json
import re
import copy
from jax import config
from jax import numpy as jnp
from jax import random
from jax import Array
from typing import Callable, List, Dict, Any, Tuple
from sources.common.constants import PERSISTENT_DATA_PATH
from sources.common.decorators import time_decorator
from sources.common.decorators import logger_decorator
from sources.common.decorators import nested_decorator
from sources.common.collections import lmap
from sources.common.collections import dfilter
from sources.ml_server.data.transformers import batchifier
from sources.ml_server.data.transformers import dispatcher
from sources.ml_server.models_.architectures import compute_weight_metadata
from sources.ml_server.models_.architectures import compute_layer_metadata
from sources.ml_server.models_.architectures import generate_architecture
from sources.ml_server.models_.models import generate_model
from sources.ml_server.models_.models import generate_optimizer
from sources.ml_server.trainers.iterands import next_state_generator
from sources.ml_server.trainers.initializers import initialize_state
from sources.database.connectors.connexions_decorators import buffer_decorator
from sources.database.sql.renderers import renderers
from sources.database.etl.loaders import load
from sources.database.etl.loaders import generate_culster_sqls
from sources.database.etl.loaders import generate_transform_cluster_sqls
from sources.database.etl.loaders import generate_transform_load_results

# TO DO
# config = dotenv_values("../.env")

_ = config.update("jax_enable_x64", True)

affix = "velib"
batch_size = 1024


key = random.PRNGKey(0)
n_epochs = 300
mini_batch_size = 32

architecture_seed = {
    "affix": "velib",
    "inputs": [
        "normalized_pos_0",
        "normalized_pos_1",
        "normalized_dow",
        "normalized_hour",
        "normalized_hour",
        "normalized_duf",
    ],
    "outputs": [
        "normalized_dad",
        "normalalized_dm",
        "normalalized_de",
    ],
    "architecture_type": "relu_feedfoward",
    "loss_type": "mse",
    "hidden_layer_type": "relu",
    "final_layer_type": "linear",
    "batch_normed": True,
    "hidden_layers_sizes": [12],
    "penalization": {"penalization_type": 1.0, "penalization_l": 1e-4},
    "optimizer": {
        "optimizer_type": "sgd_with_momentum",
        "learning_rate": 1e-3,
        "b_momentum": jnp.array([0.1]),
        "b_batch_norm": jnp.array([0.9]),
        "lps": jnp.zeros(2),
        "lds": jnp.zeros(2),
    },
}


architecture = generate_architecture(
    compute_weight_metadata, compute_layer_metadata, architecture_seed
)


(
    model,
    loss,
    penalized_loss,
    penalized_loss_grad,
    compute_grad_like,
) = generate_model(architecture)

optimize_split = generate_optimizer(architecture)


cluster_sqls = generate_culster_sqls(
    renderers, architecture["affix"], batch_size
)
transform_cluster_sqls = generate_transform_cluster_sqls(
    renderers, architecture["affix"]
)
transform_load_sqls = generate_transform_load_results()
batchify = batchifier(mini_batch_size)
dispatch_tensor = dispatcher("column_splitter", len(architecture["inputs"]))


generate_next_step = next_state_generator(
    dispatch_tensor,
    model,
    loss,
    penalized_loss,
    compute_grad_like,
    optimize_split,
)


@nested_decorator
def serialize_(x: Any) -> Any:
    return x


@buffer_decorator
def execute(buffer):
    initialize_weights, initialize_moments = initialize_state(
        architecture, key
    )
    history = []
    weights = initialize_weights()
    moments = initialize_moments(weights)
    state = {"weights": weights, "moments": moments}
    # state["weights"][-1] = lmap(lambda x: 0.0 * x, state["weights"][-1])
    for e in range(0, n_epochs):
        print("epoch={}".format(e))
        loads = load(
            buffer, cluster_sqls, transform_cluster_sqls, transform_load_sqls
        )
        for ili, li in enumerate(loads):
            for ilii, lii in enumerate(li):
                b = batchify(lii)
                for ibb, bb in enumerate(b):
                    state = generate_next_step(state, e, ili, ilii, ibb, bb)
        history.extend([copy.deepcopy(state)])
        print(dfilter(lambda k, v: ("loss" in k, True), state))
    with open(PERSISTENT_DATA_PATH + "fits/logs.json", "w") as f:
        d = re.sub("NaN", "null", json.dumps(serialize_(history)))
        d = lmap(
            lambda st: re.sub("NaN", "null", json.dumps(serialize_(st)))
            + "\n",
            history,
        )
        f.writelines(d[:-1])


_ = time_decorator(execute)()

print(4)