import copy
from jax import numpy as jnp
from jax import jit
from jax import Array
from typing import Callable
from typing import Tuple
from typing import Dict
from typing import Any
from sources.common.collections import lmap


def next_state_generator(
    dispatch_tensor: Callable,
    model: Callable,
    loss: Callable,
    penalized_loss: Callable,
    compute_grad_like: Callable,
    optimize_split: Callable,
) -> Callable:
    @jit
    def generate_next_state(
        state: Dict[str, Any], e: int, ili: int, ilii: int, ibb: int, bb: Array
    ) -> Dict[str, Any]:
        new_state = copy.deepcopy(state)
        new_state["epoch"] = e
        new_state["load_0"] = ili
        new_state["load_1"] = ilii
        new_state["batch"] = ibb
        xy = dispatch_tensor(bb)
        new_state["x"], new_state["y"] = xy["x"], xy["y"]
        new_state["model"] = model(new_state["x"], new_state["weights"])
        new_state["loss"] = loss(new_state["model"][-1][0][0], new_state["y"])
        new_state["penalized_loss"] = penalized_loss(
            new_state["x"], new_state["weights"], new_state["y"]
        )
        new_state["gradients"] = compute_grad_like(
            new_state["x"], new_state["weights"], new_state["y"]
        )
        new_state["max_gradients"] = lmap(
            lambda g: lmap(lambda g_: jnp.max(jnp.abs(g_)), g),
            new_state["gradients"],
        )
        new_state["max_weights"] = lmap(
            lambda g: lmap(lambda g_: jnp.max(jnp.abs(g_)), g),
            new_state["weights"],
        )
        opts = optimize_split(
            new_state["weights"],
            new_state["moments"],
            new_state["gradients"],
        )
        new_state["weights"] = opts[0]
        new_state["moments"] = opts[1]

        return new_state

    return generate_next_state


def next_state_generator_(
    dispatch_tensor: Callable,
    model: Callable,
    loss: Callable,
    penalized_loss: Callable,
    compute_grad_like: Callable,
    optimize_split: Callable,
) -> Callable:
    @jit
    def generate_next_state(
        state: Dict[str, Any], e: int, ili: int, ilii: int, ibb: int, bb: Array
    ) -> Dict[str, Any]:
        new_state = copy.deepcopy(state)
        new_state["epoch"] = e
        new_state["load_0"] = ili
        new_state["load_1"] = ilii
        new_state["batch"] = ibb
        xy = dispatch_tensor(bb)
        new_state["x"], new_state["y"] = xy["x"], xy["y"]
        new_state["model"] = model(new_state["x"], new_state["weights"])
        new_state["loss"] = loss(
            new_state["x"], new_state["weights"], new_state["y"]
        )
        new_state["penalized_loss"] = penalized_loss(
            new_state["x"], new_state["weights"], new_state["y"]
        )
        new_state["gradients"] = compute_grad_like(
            new_state["x"], new_state["weights"], new_state["y"]
        )
        new_state["max_gradients"] = lmap(
            lambda g: lmap(lambda g_: jnp.max(jnp.abs(g_)), g),
            new_state["gradients"],
        )
        new_state["max_weights"] = lmap(
            lambda g: lmap(lambda g_: jnp.max(jnp.abs(g_)), g),
            new_state["weights"],
        )
        opts = optimize_split(
            new_state["weights"],
            new_state["moments"],
            new_state["gradients"],
        )
        new_state["weights"] = opts[0]
        new_state["moments"] = opts[1]

        return new_state

    return generate_next_state
