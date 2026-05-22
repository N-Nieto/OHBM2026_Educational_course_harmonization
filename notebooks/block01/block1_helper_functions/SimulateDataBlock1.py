# batch_simulator.py

from __future__ import annotations

import numpy as np
import pandas as pd
# SimulateDataGUI.py


def make_simulator_input_gui():
    import numpy as np
    import pandas as pd
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    batch_names = ["Batch1", "Batch2", "Batch3"]

    # Manually set the starting values here
    BATCH_DEFAULTS = {
        "Batch1": {
            "age_mean": 25.0,
            "age_sd": 10.0,
            "age_max": 90.0,
            "sex_p": 0.60,
            "height_mean": 168.0,
            "height_sd": 8.0,
            "height_max": 210.0,
            "weight_mean": 65.0,
            "weight_sd": 12.0,
            "weight_max": 180.0,
            "add_mean": 0.00,
            "add_sd": 0.05,
            "multi_mean": 1.00,
            "multi_shape": 25.0,
        },
        "Batch2": {
            "age_mean": 40.0,
            "age_sd": 24.0,
            "age_max": 67.0,
            "sex_p": 0.45,
            "height_mean": 178.0,
            "height_sd": 9.0,
            "height_max": 220.0,
            "weight_mean": 78.0,
            "weight_sd": 14.0,
            "weight_max": 200.0,
            "add_mean": 0.00,
            "add_sd": 0.05,
            "multi_mean": 1.00,
            "multi_shape": 25.0,
        },
        "Batch3": {
            "age_mean": 75.0,
            "age_sd": 14.0,
            "age_max": 100.0,
            "sex_p": 0.52,
            "height_mean": 165.0,
            "height_sd": 4.0,
            "height_max": 200.0,
            "weight_mean": 72.0,
            "weight_sd": 16.0,
            "weight_max": 190.0,
            "add_mean": 0.00,
            "add_sd": 0.05,
            "multi_mean": 1.00,
            "multi_shape": 25.0,
        },
    }

    def equal_batch_sizes(n: int):
        base = n // 3
        rem = n % 3
        sizes = [base, base, base]
        for i in range(rem):
            sizes[i] += 1
        return sizes

    def build_batch_controls(batch_name: str):
        d = BATCH_DEFAULTS[batch_name]
        return {
            "age_mean": widgets.BoundedFloatText(
                value=d["age_mean"],
                min=0.0,
                max=100.0,
                step=0.1,
                description="Age mean",
            ),
            "age_sd": widgets.BoundedFloatText(
                value=d["age_sd"], min=0.0, max=50.0, step=0.1, description="Age SD"
            ),
            "age_max": widgets.BoundedFloatText(
                value=d["age_max"], min=1.0, max=100.0, step=1.0, description="Age max"
            ),
            "sex_p": widgets.BoundedFloatText(
                value=d["sex_p"], min=0.0, max=1.0, step=0.01, description="% female"
            ),
            "height_mean": widgets.BoundedFloatText(
                value=d["height_mean"],
                min=0.0,
                max=250.0,
                step=0.1,
                description="Ht mean",
            ),
            "height_sd": widgets.BoundedFloatText(
                value=d["height_sd"], min=0.0, max=100.0, step=0.1, description="Ht SD"
            ),
            "height_max": widgets.BoundedFloatText(
                value=d["height_max"],
                min=1.0,
                max=250.0,
                step=1.0,
                description="Ht max",
            ),
            "weight_mean": widgets.BoundedFloatText(
                value=d["weight_mean"],
                min=0.0,
                max=300.0,
                step=0.1,
                description="Wt mean",
            ),
            "weight_sd": widgets.BoundedFloatText(
                value=d["weight_sd"], min=0.0, max=100.0, step=0.1, description="Wt SD"
            ),
            "weight_max": widgets.BoundedFloatText(
                value=d["weight_max"],
                min=1.0,
                max=400.0,
                step=1.0,
                description="Wt max",
            ),
            "add_mean": widgets.BoundedFloatText(
                value=d["add_mean"],
                min=-1000.0,
                max=1000.0,
                step=0.01,
                description="Add mean",
            ),
            "add_sd": widgets.BoundedFloatText(
                value=d["add_sd"], min=0.0, max=1000.0, step=0.01, description="Add SD"
            ),
            "multi_mean": widgets.BoundedFloatText(
                value=d["multi_mean"],
                min=0.01,
                max=1000.0,
                step=0.01,
                description="Multi mean",
            ),
            "multi_shape": widgets.BoundedFloatText(
                value=d["multi_shape"],
                min=1.01,
                max=10000.0,
                step=0.1,
                description="Multi shape",
            ),
        }

    # ... keep the rest of your code the same ...

    def panel_for_batch(batch_name: str, c: dict):
        grid = widgets.GridBox(
            children=[
                c["age_mean"],
                c["age_sd"],
                c["age_max"],
                c["sex_p"],
                c["height_mean"],
                c["height_sd"],
                c["height_max"],
                c["weight_mean"],
                c["weight_sd"],
                c["weight_max"],
                c["add_mean"],
                c["add_sd"],
                c["multi_mean"],
                c["multi_shape"],
            ],
            layout=widgets.Layout(
                grid_template_columns="repeat(2, minmax(220px, 1fr))",
                grid_gap="8px 12px",
                width="100%",
            ),
        )
        return widgets.VBox([widgets.HTML(f"<b>{batch_name}</b>"), grid])

    # Global controls
    global_controls = {
        "n": widgets.BoundedIntText(
            value=900, min=3, max=1000000, step=1, description="n"
        ),
        "seed": widgets.IntText(value=42, description="Seed"),
        "data_mean": widgets.BoundedFloatText(
            value=0.0, min=-1000.0, max=1000.0, step=0.1, description="Data mean"
        ),
        "data_sd": widgets.BoundedFloatText(
            value=1.0, min=0.0, max=1000.0, step=0.1, description="Data SD"
        ),
        "noise_sd": widgets.BoundedFloatText(
            value=1.0, min=0.0, max=1000.0, step=0.1, description="Noise SD"
        ),
        "sex_beta": widgets.BoundedFloatText(
            value=0.2, min=-1000.0, max=1000.0, step=0.01, description="Sex beta"
        ),
        "age_beta": widgets.BoundedFloatText(
            value=-0.3, min=-1000.0, max=1000.0, step=0.01, description="Age beta"
        ),
        "height_beta": widgets.BoundedFloatText(
            value=0.4, min=-1000.0, max=1000.0, step=0.01, description="Ht beta"
        ),
        "weight_beta": widgets.BoundedFloatText(
            value=-0.1, min=-1000.0, max=1000.0, step=0.01, description="Wt beta"
        ),
    }

    batch_controls = {b: build_batch_controls(b) for b in batch_names}

    global_box = widgets.GridBox(
        children=list(global_controls.values()),
        layout=widgets.Layout(
            grid_template_columns="repeat(2, minmax(220px, 1fr))",
            grid_gap="8px 12px",
            width="100%",
        ),
    )

    tabs = widgets.Tab()
    tabs.children = [panel_for_batch(b, batch_controls[b]) for b in batch_names]
    for i, b in enumerate(batch_names):
        tabs.set_title(i, b)

    done_button = widgets.Button(
        description="Done", button_style="primary", icon="check"
    )
    output = widgets.Output()

    def on_done():
        n = int(global_controls["n"].value)
        seed = int(global_controls["seed"].value)
        rng = np.random.default_rng(seed)

        # Baseline outcome vector
        data = rng.normal(
            loc=float(global_controls["data_mean"].value),
            scale=float(global_controls["data_sd"].value),
            size=n,
        )

        # Batch labels
        sizes = equal_batch_sizes(n)
        batch = np.concatenate(
            [
                np.repeat(batch_names[0], sizes[0]),
                np.repeat(batch_names[1], sizes[1]),
                np.repeat(batch_names[2], sizes[2]),
            ]
        )

        covariate_specs = {
            "age": {
                "dist": "normal",
                "mean": {
                    b: float(batch_controls[b]["age_mean"].value) for b in batch_names
                },
                "sd": {
                    b: float(batch_controls[b]["age_sd"].value) for b in batch_names
                },
                "clip": (
                    0,
                    float(max(batch_controls[b]["age_max"].value for b in batch_names)),
                ),
            },
            "sex": {
                "dist": "bernoulli",
                "p": {b: float(batch_controls[b]["sex_p"].value) for b in batch_names},
            },
            "height": {
                "dist": "normal",
                "mean": {
                    b: float(batch_controls[b]["height_mean"].value)
                    for b in batch_names
                },
                "sd": {
                    b: float(batch_controls[b]["height_sd"].value) for b in batch_names
                },
                "clip": (
                    0,
                    float(
                        max(batch_controls[b]["height_max"].value for b in batch_names)
                    ),
                ),
            },
            "weight": {
                "dist": "normal",
                "mean": {
                    b: float(batch_controls[b]["weight_mean"].value)
                    for b in batch_names
                },
                "sd": {
                    b: float(batch_controls[b]["weight_sd"].value) for b in batch_names
                },
                "clip": (
                    0,
                    float(
                        max(batch_controls[b]["weight_max"].value for b in batch_names)
                    ),
                ),
            },
        }

        betas = {
            "sex": float(global_controls["sex_beta"].value),
            "age": float(global_controls["age_beta"].value),
            "height": float(global_controls["height_beta"].value),
            "weight": float(global_controls["weight_beta"].value),
        }

        batch_params = {
            b: {
                "add_mean": float(batch_controls[b]["add_mean"].value),
                "add_sd": float(batch_controls[b]["add_sd"].value),
                "multi_mean": float(batch_controls[b]["multi_mean"].value),
                "multi_shape": float(batch_controls[b]["multi_shape"].value),
            }
            for b in batch_names
        }

        result = {
            "data": data,
            "batch": batch,
            "covariate_specs": covariate_specs,
            "betas": betas,
            "batch_params": batch_params,
        }

        make_simulator_input_gui.last_result = result

        summary = pd.DataFrame(
            {
                "batch": batch_names,
                "n": sizes,
                "age_mean": [batch_controls[b]["age_mean"].value for b in batch_names],
                "sex_p": [batch_controls[b]["sex_p"].value for b in batch_names],
                "height_mean": [
                    batch_controls[b]["height_mean"].value for b in batch_names
                ],
                "weight_mean": [
                    batch_controls[b]["weight_mean"].value for b in batch_names
                ],
                "add_mean": [batch_controls[b]["add_mean"].value for b in batch_names],
                "multi_mean": [
                    batch_controls[b]["multi_mean"].value for b in batch_names
                ],
            }
        )

        with output:
            clear_output(wait=True)
            display(summary)
            print("Built objects: data, batch, covariate_specs, betas, batch_params")

    done_button.on_click(on_done)

    ui = widgets.VBox(
        [
            widgets.HTML("<h3>Simulator input builder</h3>"),
            widgets.HTML("<b>Global settings</b>"),
            global_box,
            widgets.HTML("<b>Batch-specific settings</b>"),
            tabs,
            done_button,
            output,
        ]
    )

    make_simulator_input_gui.last_result = None
    display(ui)
    return ui


def simulate_batched_data(
    data,
    batch,
    covariate_specs=None,
    betas=None,
    batch_params=None,
    noise_sd: float = 1.0,
    relative_batch_effects: bool = True,
    seed: int | None = None,
):
    """
    Simulate batched data from a baseline 1D outcome vector.

    Parameters
    ----------
    data : array-like, shape (n,)
        Baseline outcome vector to be modified.
    batch : array-like, shape (n,)
        Batch labels for each observation.
    covariate_specs : dict or None
        Optional specification for simulated covariates.
        Example:
        {
            "age":    {"dist": "normal",    "mean": {"Batch1": 30, "Batch2": 50, "Batch3": 20}, "sd": 5,  "clip": (0, 100)},
            "sex":    {"dist": "bernoulli",  "p":    {"Batch1": 0.5, "Batch2": 0.7, "Batch3": 0.4}},
            "height": {"dist": "normal",    "mean": 170, "sd": 10, "clip": (120, 220)},
            "weight": {"dist": "normal",    "mean": 70,  "sd": 15, "clip": (30, 200)}
        }
        For normal covariates, mean/sd can be scalars or dicts keyed by batch.
    betas : dict or None
        Shared covariate effect sizes across batches.
        Example: {"sex": 0.2, "age": -0.3, "height": 0.4, "weight": -0.1}
    batch_params : dict or None
        Batch-specific parameters. Each batch can have:
        {
            "add_mean": 0.0,      # additive batch effect mean
            "add_sd": 0.10,       # additive batch effect SD
            "multi_mean": 1.0,    # expected multiplicative noise scale (>0)
            "multi_shape": 20.0   # inverse-gamma shape (>1)
        }

        If relative_batch_effects=True, add_mean and add_sd are interpreted
        relative to std(data), so the actual additive shift is in the same units
        as data.

        Example:
        {
            "Batch1": {"add_mean": 0.0,  "add_sd": 0.10, "multi_mean": 1.0, "multi_shape": 25.0},
            "Batch2": {"add_mean": 0.15, "add_sd": 0.12, "multi_mean": 1.1, "multi_shape": 18.0},
            "Batch3": {"add_mean": -0.10,"add_sd": 0.08, "multi_mean": 0.9, "multi_shape": 30.0}
        }
    noise_sd : float
        Base SD of the residual noise before multiplicative scaling.
    relative_batch_effects : bool
        If True, additive batch effects are scaled by std(data).
    seed : int or None
        Random seed.

    Returns
    -------
    pandas.DataFrame
        Contains the baseline data, simulated covariates, batch terms, and final outcome.

        Columns include:
        - batch
        - y_base
        - y_covariate
        - batch_add
        - noise_scale
        - noise
        - y
        - any simulated covariates
    """
    rng = np.random.default_rng(seed)

    y_base = np.asarray(data, dtype=float).reshape(-1)
    batch = np.asarray(batch)

    if y_base.shape[0] != batch.shape[0]:
        raise ValueError(
            f"`data` and `batch` must have the same length. "
            f"Got {y_base.shape[0]} and {batch.shape[0]}."
        )

    if covariate_specs is None:
        covariate_specs = {}

    if betas is None:
        betas = {}

    if batch_params is None:
        batch_params = {}

    # Preserve the order batches first appear in the input
    batch_levels = list(dict.fromkeys(batch.tolist()))
    n = y_base.shape[0]
    y_sd = float(np.std(y_base, ddof=1)) if n > 1 else float(np.std(y_base))
    scale_for_add = y_sd if relative_batch_effects else 1.0

    out = pd.DataFrame(
        {
            "batch": batch,
            "y_base": y_base,
        }
    )

    # Storage for covariates
    for cov_name in covariate_specs:
        out[cov_name] = np.nan

    out["y_covariate"] = 0.0
    out["batch_add"] = 0.0
    out["noise_scale"] = 1.0
    out["noise"] = 0.0

    def resolve(value, label, default=None):
        """
        Allow scalar or dict inputs keyed by batch label.
        """
        if value is None:
            return default
        if isinstance(value, dict):
            return value.get(label, default)
        return value

    for b in batch_levels:
        mask = batch == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue

        # ---------
        # Covariates
        # ---------
        cov_effect = np.zeros(n_b, dtype=float)

        for cov_name, spec in covariate_specs.items():
            dist = spec.get("dist", "normal").lower()

            if dist == "normal":
                mu = resolve(spec.get("mean", 0.0), b, 0.0)
                sd = resolve(spec.get("sd", 1.0), b, 1.0)
                x = rng.normal(loc=float(mu), scale=float(sd), size=n_b)

                if "clip" in spec and spec["clip"] is not None:
                    lo, hi = spec["clip"]
                    x = np.clip(x, lo, hi)

            elif dist in {"bernoulli", "binary"}:
                p = resolve(spec.get("p", 0.5), b, 0.5)
                p = float(p)
                if not (0.0 <= p <= 1.0):
                    raise ValueError(
                        f"Bernoulli probability for '{cov_name}' in '{b}' must be between 0 and 1."
                    )
                x = rng.binomial(n=1, p=p, size=n_b)

            else:
                raise ValueError(
                    f"Unsupported distribution '{dist}' for covariate '{cov_name}'. "
                    f"Use 'normal' or 'bernoulli'."
                )

            out.loc[mask, cov_name] = x

            beta = float(betas.get(cov_name, 0.0))
            cov_effect += beta * x

        out.loc[mask, "y_covariate"] = cov_effect

        # ----------------
        # Additive batch term
        # ----------------
        p = batch_params.get(b, {})
        add_mean = float(p.get("add_mean", 0.0))
        add_sd = float(p.get("add_sd", 0.0))

        if relative_batch_effects:
            add_mean *= scale_for_add
            add_sd *= scale_for_add

        batch_add = rng.normal(loc=add_mean, scale=add_sd, size=n_b)
        out.loc[mask, "batch_add"] = batch_add

        # ----------------------------
        # Multiplicative noise scaling
        # ----------------------------
        # We use an inverse-gamma style positive scale:
        # if S ~ InvGamma(shape=a, scale=s), then E[S] = s/(a-1) for a > 1.
        # We choose s so that the mean is multi_mean.
        multi_mean = float(p.get("multi_mean", 1.0))
        multi_shape = float(p.get("multi_shape", 20.0))

        if multi_mean <= 0:
            raise ValueError(f"`multi_mean` for batch '{b}' must be > 0.")
        if multi_shape <= 1:
            raise ValueError(
                f"`multi_shape` for batch '{b}' must be > 1 for a finite mean."
            )

        invgamma_scale = multi_mean * (multi_shape - 1.0)

        # Draw InvGamma(shape, scale) via Gamma(shape, rate) then invert.
        gamma_draw = rng.gamma(shape=multi_shape, scale=1.0 / invgamma_scale, size=n_b)
        noise_scale = 1.0 / gamma_draw

        out.loc[mask, "noise_scale"] = noise_scale

        # Base noise, then scaled by the batch-specific multiplicative effect
        eps = rng.normal(loc=0.0, scale=noise_sd, size=n_b)
        noise = eps * noise_scale
        out.loc[mask, "noise"] = noise

        # ----------------
        # Final outcome
        # ----------------
        out.loc[mask, "y"] = (
            out.loc[mask, "y_base"].to_numpy()
            + out.loc[mask, "y_covariate"].to_numpy()
            + out.loc[mask, "batch_add"].to_numpy()
            + out.loc[mask, "noise"].to_numpy()
        )

    return out
