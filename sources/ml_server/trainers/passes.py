from jax import numpy as jnp
from jax import jit
from jax import grad
from sources.common.collections import concatenate


def forward_computer(layers, model, typ="xy"):
    grad_ = grad(
        lambda x, ws, y: jnp.sum(jnp.array(model(x, ws, y)[1:3])), argnums=1
    )
    if typ == "xy":

        @jit
        def compute_forward(t, w):
            x, ws, y = t["x"], w, t["y"]
            m, g = model(x, ws, y), grad_(x, ws, y)
            predict = m[0][-1][0]
            penalized_loss = m[1] + m[2]
            loss = m[1]
            batch_norm_metrics = [y_[1:] for y_ in m[0]]
            update_metrics = [
                (g[i], concatenate(batch_norm_metrics[i][0:2], g[i][2:]))[
                    l["type"] == "batch_norm"
                ]
                for (i, li) in enumerate(layers)
            ]
            return {
                "gradient": g,
                "penalized_loss": penalized_loss,
                "loss": loss,
                "batch_norm_metrics": batch_norm_metrics,
                "update_metrics": update_metrics,
            }

    else:
        raise Exception("Not implemented")
    return compute_forward


def backward_computer(optimizer_model):
    def compute_backward(bm, fm, optimizer_params):
        return optimizer_model(
            bm,
            fm["update_metrics"],
            optimizer_params["lr"],
            optimizer_params["bs"],
            optimizer_params["b_batch_norm"],
            optimizer_params["lps"],
            optimizer_params["lds"],
        )

    return compute_backward
