"""Helper functions for the Block 3 notebooks (effects of site in ML pipelines).

Sections
--------
- B3_N1 helpers: simulate_multisite_data_with_covariates, run_fpr_simulation
- Shared helpers: simulate_classification, cross_validated_auc, site_auc,
  plot_scenario, summarize_seed_results
- B3_N2 (EoS in ML): run_classification_scenario, seed_summary
- B3_N3 (metrics by site): make_good_and_bad_sites, run_metrics_by_site_scenario,
  plot_metrics_by_site; limitations: site_size_experiment,
  plot_site_size_experiment, metrics_table_by_site, auc_under_site_effects
- B3_N4 (ComBat with site-class imbalance): evaluate_harmonization_pipelines,
  run_harmonization_scenario, plot_pipeline_auc
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split
from uniharmony import verbosity
from uniharmony.datasets import make_multisite_classification
from uniharmony.metrics import report_metrics_by_site
from uniharmony.plot import plot_decision_boundary_2d

sns.set_theme(style="whitegrid")

verbosity("error")

# Imported after setting the verbosity: uniharmony.combat binds its logger at
# import time, so importing it earlier would print debug messages.
from uniharmony.combat import NeuroComBat  # noqa: E402


# =============================================================================
# B3_N1 helpers
# =============================================================================
def simulate_multisite_data_with_covariates(n_samples, n_features):

    # ---------------------------
    # Biological variables
    # ---------------------------
    age = np.random.uniform(20, 80, n_samples)
    sex = np.random.binomial(1, 0.5, n_samples)
    disease = np.random.binomial(1, 0.3, n_samples)

    # ---------------------------
    # Scanner
    # ---------------------------
    scanner = np.random.choice(["A", "B", "C"], size=n_samples)
    scanner_map = {"A": 1, "B": 2, "C": 3}
    scanner_num = np.array([scanner_map[s] for s in scanner])
    # ---------------------------
    # Generate true signal
    # ---------------------------
    Y_true_nonlinear = []
    Y_true_linear = []

    for j in range(n_features):
        beta_age_linear = np.random.uniform(0.4, 0.6)
        beta_age_nonlinear = np.random.uniform(0.1, 0.2)
        beta_sex = np.random.uniform(1.5, 2.5)
        beta_disease = np.random.uniform(4, 6)

        base_bio_nonlinear = (
            beta_age_nonlinear * (age - 50) ** 2
            + beta_sex * sex
            + beta_disease * disease
        )
        base_bio_linear = (
            beta_age_linear * age + beta_sex * sex + beta_disease * disease
        )

        offset = np.random.normal(0, 1)

        y_j_nonlinear = base_bio_nonlinear + offset
        y_j_linear = base_bio_linear + offset
        Y_true_nonlinear.append(y_j_nonlinear)
        Y_true_linear.append(y_j_linear)

    Y_true_nonlinear = np.array(Y_true_nonlinear)
    Y_true_linear = np.array(Y_true_linear)

    # ---------------------------
    # Generate observed data
    # ---------------------------
    Y_obs_nonlinear = []
    Y_obs_linear = []

    for j in range(n_features):
        # Additive scanner effect
        scanner_effect = scanner_num * np.random.uniform(10, 20)

        # Multiplicative scanner effect
        scale_effect = scanner_num * np.random.uniform(0.5, 1.5)
        noise = np.random.normal(0, 2 * scale_effect, n_samples)

        nonlinear_distortion = scanner_num * 0.03 * (age - 50) ** 2
        slope_distortion = (scanner_num * np.random.uniform(0.2, 0.5)) * (0.5 * age)

        y_obs_j_nonlinear = Y_true_nonlinear[j] + scanner_effect + noise
        y_obs_j_linear = Y_true_linear[j] + scanner_effect + noise
        Y_obs_nonlinear.append(y_obs_j_nonlinear)
        Y_obs_linear.append(y_obs_j_linear)

    Y_obs_nonlinear = np.array(Y_obs_nonlinear)
    Y_obs_linear = np.array(Y_obs_linear)

    return (
        Y_true_linear,
        Y_obs_linear,
        Y_true_nonlinear,
        Y_obs_nonlinear,
        age,
        sex,
        disease,
        scanner,
    )


# Run multiple simulations to assess FPR and correlations
def run_fpr_simulation(seed, n=800, n_features=20):

    np.random.seed(seed)

    # Scanner
    scanner = np.random.choice(["A", "B", "C"], size=n)
    scanner_map = {"A": 0, "B": 1, "C": 2}
    scanner_num = np.array([scanner_map[s] for s in scanner])

    # correlated age
    age = np.zeros(n)
    for i, s in enumerate(scanner):
        if s == "A":
            age[i] = np.random.normal(30, 5)
        elif s == "B":
            age[i] = np.random.normal(50, 5)
        else:
            age[i] = np.random.normal(70, 5)

    sex = np.random.binomial(1, 0.5, n)
    disease = np.random.binomial(1, 0.3, n)

    Y_obs = []
    Y_true = []

    # Simulate observed features with scanner effects and interactions
    for j in range(n_features):
        scanner_effect = scanner_num * np.random.uniform(1.5, 3)
        interaction = scanner_num * 0.15 * age
        noise = np.random.normal(0, 2, n)

        # TRUE BIO: all linear effects + interaction
        true_bio = 0.05 * (age - 50) + 2 * sex + 5 * disease + 0.03 * age * disease

        y_j = true_bio + scanner_effect + interaction + noise
        Y_obs.append(y_j)
        Y_true.append(true_bio)

    Y_obs = np.array(Y_obs)
    Y_true = np.array(Y_true)

    # null group
    group = np.random.binomial(1, 0.5, n)

    categorical_covariates = np.vstack((sex, disease, group)).T

    harm_model = NeuroComBat()
    Y_combat = harm_model.fit_transform(
        X=Y_obs.T,
        sites=scanner,
        continuous_covariates=age.reshape(-1, 1),
        categorical_covariates=categorical_covariates,
    )

    Y_combat = Y_combat.T

    # -----------------------------
    # P-values
    # -----------------------------
    pvals = []

    for j in range(n_features):
        p = ttest_ind(Y_combat[j][group == 0], Y_combat[j][group == 1]).pvalue
        pvals.append(p)

    pvals = np.array(pvals)

    # -----------------------------
    # Correlations (KEY ADDITION)
    # -----------------------------
    corr_scanner_age = np.corrcoef(scanner_num, age)[0, 1]
    corr_scanner_group = np.corrcoef(scanner_num, group)[0, 1]

    return pvals, corr_scanner_age, corr_scanner_group


# =============================================================================
# Shared helpers (B3_N2, B3_N3, B3_N4)
# =============================================================================
def simulate_classification(
    n_samples, random_state, n_sites=3, n_features=2, signal_type="blobs", **sim_kwargs
):
    """Simulate a multi-site classification problem with uniharmony.

    Thin wrapper around ``make_multisite_classification`` so that every notebook
    simulates data in the same way. Extra keyword arguments (``signal_strength``,
    ``site_effect_strength``, ``site_effect_type``, ``balance_per_site``, ...) are
    passed through unchanged.

    Note: ``balance_per_site`` is ``[proportion of Class 0, proportion of Class 1]``
    for each site, e.g. ``[[0.1, 0.9], [0.9, 0.1]]``. A flat list such as
    ``[0.1, 0.9]`` is applied to *every* site, so it creates no site-class imbalance.
    """
    return make_multisite_classification(
        n_samples=n_samples,
        n_sites=n_sites,
        n_features=n_features,
        signal_type=signal_type,
        random_state=random_state,
        **sim_kwargs,
    )


def cross_validated_auc(X, y, clf, cv):
    """Mean AUC over the cross-validation folds."""
    return float(cross_val_score(clf, X, y, cv=cv, scoring="roc_auc").mean())


def site_auc(y, scores, sites):
    """AUC of one (pooled) model's scores, computed on the subjects of each site.

    No model is trained per site: the same scores are only *evaluated* per site.
    This is a narrow diagnostic. It ignores a site effect only when the effect is a
    pure location shift (same shift for every subject of a site), it does not carry
    over to threshold metrics (accuracy, sensitivity, ...), and it is noisy for
    sites with few subjects of one class. See the limitations section of B3_N3.
    """
    return {
        s: roc_auc_score(y[sites == s], scores[sites == s]) for s in np.unique(sites)
    }


def plot_scenario(ax, X, y, sites, title, clf=None, fontsize=13):
    """Scatter the first two features (colour = site, marker = class).

    If a fitted 2-feature classifier is given, its decision boundary is drawn.
    """
    site_names = [f"Site {s}" for s in np.unique(sites)]
    df = pd.DataFrame(
        {
            "Feature 1": X[:, 0],
            "Feature 2": X[:, 1],
            "Class": [f"Class {c}" for c in y],
            "Site": [f"Site {s}" for s in sites],
        }
    )
    sns.scatterplot(
        data=df,
        x="Feature 1",
        y="Feature 2",
        hue="Site",
        style="Class",
        s=100,
        alpha=0.7,
        ax=ax,
        hue_order=site_names,
        style_order=["Class 0", "Class 1"],
    )
    ax.set_title(title, fontsize=fontsize, fontweight="bold")
    if clf is not None and X.shape[1] == 2:
        plot_decision_boundary_2d(ax, clf)


def summarize_seed_results(results, by="Scenario"):
    """Combine the per-seed results of several scenarios into one summary table.

    Parameters
    ----------
    results : list of pandas.DataFrame or pandas.DataFrame
        Outputs of ``run_classification_scenario``, ``run_metrics_by_site_scenario``
        or ``run_harmonization_scenario`` (one row per seed).
    by : str or list of str
        Column(s) that identify a scenario (and a pipeline, for B3_N4).

    Returns
    -------
    pandas.DataFrame
        Mean, SD, min and max of every metric, in the order the scenarios were run.
    """
    if not isinstance(results, (list, tuple)):
        results = [results]
    frames = []
    for r in results:
        frame = r.copy()
        frame.attrs = {}  # the plotted example is not part of the summary
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    by = [by] if isinstance(by, str) else list(by)
    metrics = [c for c in df.columns if c not in by + ["seed"]]
    return (
        df.groupby(by, sort=False)[metrics].agg(["mean", "std", "min", "max"]).round(3)
    )


# =============================================================================
# B3_N2: effects of site in ML pipelines
# =============================================================================
def seed_summary(
    n_samples,
    clf,
    cv,
    n_seeds,
    title,
    n_sites=3,
    n_features=2,
    signal_type="blobs",
    **sim_kwargs,
):
    """Repeat one scenario with ``n_seeds`` different random datasets.

    Returns one row per seed with the mean cross-validated AUC.
    """
    rows = []
    for seed in range(n_seeds):
        X, y, _ = simulate_classification(
            n_samples, seed, n_sites, n_features, signal_type, **sim_kwargs
        )
        rows.append(
            {"Scenario": title, "seed": seed, "auc": cross_validated_auc(X, y, clf, cv)}
        )
    return pd.DataFrame(rows)


def run_classification_scenario(
    title,
    n_samples,
    clf,
    cv,
    random_state,
    n_seeds=20,
    n_sites=3,
    n_features=2,
    signal_type="blobs",
    verbose=True,
    **sim_kwargs,
):
    """Simulate, evaluate with CV and plot one scenario, then repeat it over seeds.

    The dataset drawn with ``random_state`` is plotted and evaluated; the same
    scenario is then simulated with seeds ``0 ... n_seeds - 1`` to show how much
    the result depends on the particular dataset.

    Returns
    -------
    pandas.DataFrame
        One row per seed (see ``seed_summary``). The AUC of the plotted example is
        stored in ``results.attrs["example"]``. Pass a list of these tables to
        ``summarize_seed_results`` to build the final table of the notebook.
    """
    X, y, sites = simulate_classification(
        n_samples, random_state, n_sites, n_features, signal_type, **sim_kwargs
    )
    auc = cross_validated_auc(X, y, clf, cv)

    _, ax = plt.subplots(figsize=(10, 8))
    plot_scenario(
        ax,
        X,
        y,
        sites,
        f"{title}\nMean CV AUC for example dataset (random_state={random_state}): {auc:.3f}",
        clf=clone(clf).fit(X, y),  # decision boundary for visualization only
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()

    results = seed_summary(
        n_samples,
        clf,
        cv,
        n_seeds,
        title,
        n_sites,
        n_features,
        signal_type,
        **sim_kwargs,
    )
    results.attrs["example"] = {"random_state": random_state, "auc": auc}

    if verbose:
        print(
            f"Across {n_seeds} other datasets: "
            f"AUC {results.auc.mean():.3f} [{results.auc.min():.3f}-{results.auc.max():.3f}]"
        )
    return results


# =============================================================================
# B3_N3: metrics by site
# =============================================================================
def make_good_and_bad_sites(
    n_samples_per_site, bad_sites, signal_strength=1.0, n_features=10, random_state=None
):
    """Simulate sites with ("good") and without ("bad") a feature-class relationship.

    All sites come from one generative problem without effects of site, so the
    feature distributions are the same everywhere. In the bad sites the class
    labels are shuffled: the features carry no information about the class there
    (think of unreliable labels or unusable scans).

    Parameters
    ----------
    n_samples_per_site : list of int
        Number of subjects in each site.
    bad_sites : list of int
        Indices of the sites without signal (empty list = all sites good).
    """
    X, y, sites = make_multisite_classification(
        n_sites=len(n_samples_per_site),
        n_samples=list(n_samples_per_site),
        n_features=n_features,
        signal_type="linear",
        signal_strength=signal_strength,
        site_effect_strength=0,  # no effects of site
        random_state=random_state,
    )
    rng = np.random.default_rng(random_state)
    y = y.copy()
    for s in bad_sites:
        idx = np.flatnonzero(sites == s)
        y[idx] = rng.permutation(y[idx])
    return X, y, sites


def metrics_by_site(X, y, sites, clf, cv, metrics):
    """Cross-validated metrics, overall and per site.

    The model is always trained on subjects from all sites; only the metric is
    computed separately per site. The predicted probabilities are passed to
    ``report_metrics_by_site``: AUC is computed from them, and metrics that need
    class labels (accuracy, sensitivity, balanced accuracy, ...) threshold them at 0.5.
    """
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    return report_metrics_by_site(y, proba, sites, metrics)


def specificity_score(y_true, y_pred):
    """Specificity = recall of Class 0 (true negative rate)."""
    return recall_score(y_true, y_pred, pos_label=0)


def class_counts_by_site(y, sites):
    """{site: (n Class 0, n Class 1)}."""
    return {
        s: (int(np.sum(y[sites == s] == 0)), int(np.sum(y[sites == s] == 1)))
        for s in np.unique(sites)
    }


def plot_metrics_by_site(
    ax, results, metric_name, n_per_site, title, color="steelblue"
):
    """Bar plot of a metric per site, labelled with the number of subjects, and the overall value.

    No average across sites is drawn: an unweighted average gives a site of 20
    subjects the same weight as a site of 1000 (see the limitations section).
    """
    scores = results[metric_name]
    site_ids = [s for s in scores if s != "overall"]
    values = [scores[s] for s in site_ids]
    labels = [f"Site {s}\n(n={n_per_site[s]})" for s in site_ids]

    sns.barplot(x=labels, y=values, color=color, ax=ax)
    ax.bar_label(ax.containers[0], fmt="%.2f", padding=3)
    ax.axhline(
        scores["overall"],
        color="black",
        linestyle="--",
        label=f"Overall (all subjects): {scores['overall']:.3f}",
    )
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.7, label="Chance level: 0.5")
    ax.set_ylim([0, 1.05])
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)


def run_metrics_by_site_scenario(
    title,
    n_samples_per_site,
    bad_sites,
    clf,
    cv,
    random_state,
    metrics,
    signal_strength=1.0,
    n_features=10,
    n_seeds=20,
    summary_metric=roc_auc_score,
):
    """Evaluate one good/bad-sites scenario and repeat it over seeds.

    Returns
    -------
    results : dict
        ``report_metrics_by_site`` output for the example dataset (``random_state``).
    n_per_site : dict
        Number of subjects per site.
    seed_results : pandas.DataFrame
        One row per seed with the overall value and the value in each site for
        ``summary_metric`` (no average across sites).
    """
    X, y, sites = make_good_and_bad_sites(
        n_samples_per_site, bad_sites, signal_strength, n_features, random_state
    )
    results = metrics_by_site(X, y, sites, clf, cv, metrics)
    n_per_site = {s: int(np.sum(sites == s)) for s in np.unique(sites)}

    rows = []
    name = summary_metric.__name__
    for seed in range(n_seeds):
        Xs, ys, ss = make_good_and_bad_sites(
            n_samples_per_site, bad_sites, signal_strength, n_features, seed
        )
        r = metrics_by_site(Xs, ys, ss, clf, cv, [summary_metric])[name]
        row = {"Scenario": title, "seed": seed, "overall": r["overall"]}
        row.update({f"site_{k}": v for k, v in r.items() if k != "overall"})
        rows.append(row)
    return results, n_per_site, pd.DataFrame(rows)


# ---- B3_N3 limitations ------------------------------------------------------
def site_size_experiment(
    n_samples_per_site, clf, cv, n_datasets=100, signal_strength=0.5, n_features=10
):
    """Per-site AUC when sites differ in size, over many random datasets.

    All sites come from the same problem (no bad sites, no effects of site), so
    every site has the same true AUC. Any difference between sites is sampling noise.

    Returns
    -------
    pandas.DataFrame
        One row per dataset: overall AUC, AUC of each site, and the unweighted
        mean of the site AUCs.
    """
    rows = []
    for seed in range(n_datasets):
        X, y, sites = make_good_and_bad_sites(
            n_samples_per_site, [], signal_strength, n_features, seed
        )
        proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
        per_site = site_auc(y, proba, sites)
        row = {"seed": seed, "overall": roc_auc_score(y, proba)}
        row.update(
            {f"Site {s} (n={n})": per_site[s] for s, n in enumerate(n_samples_per_site)}
        )
        row["mean of sites (unweighted)"] = float(np.mean(list(per_site.values())))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_site_size_experiment(df, ax=None):
    """Box plot of (site AUC - overall AUC) and (mean of sites - overall AUC) per dataset."""
    cols = [c for c in df.columns if c not in ("seed", "overall")]
    diff = (
        df[cols]
        .sub(df["overall"], axis=0)
        .melt(var_name="", value_name="AUC minus overall AUC")
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=diff, x="", y="AUC minus overall AUC", ax=ax, color="lightsteelblue"
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title(
        f"Difference from the overall AUC over {len(df)} datasets (same true AUC in every site)"
    )
    plt.tight_layout()
    return ax


def metrics_table_by_site(X, y, sites, clf, cv, threshold=0.5):
    """Overall and per-site AUC and threshold metrics of one pooled model, with class counts.

    The model is trained on all sites (out-of-fold probabilities from cv). AUC
    uses the probabilities; the other metrics use the predicted class
    (probability >= threshold).
    """
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    y_pred = (proba >= threshold).astype(int)
    groups = [("Overall", np.ones(len(y), dtype=bool))] + [
        (f"Site {s}", sites == s) for s in np.unique(sites)
    ]
    rows = []
    for name, m in groups:
        rows.append(
            {
                "": name,
                "n Class 0": int(np.sum(y[m] == 0)),
                "n Class 1": int(np.sum(y[m] == 1)),
                "AUC": roc_auc_score(y[m], proba[m]),
                "Accuracy": accuracy_score(y[m], y_pred[m]),
                "Sensitivity": recall_score(y[m], y_pred[m]),
                "Specificity": specificity_score(y[m], y_pred[m]),
                "Balanced accuracy": balanced_accuracy_score(y[m], y_pred[m]),
            }
        )
    return pd.DataFrame(rows).set_index("").round(3)


def auc_under_site_effects(random_state=42, n_samples=600, plot=True):
    """Which site effects leave the AUC *inside one site* unchanged?

    A model is trained on half of a one-site dataset without effects of site. The
    other half (the test site) is then distorted in different ways and scored with
    the same, unchanged model. A pure location shift moves every score by the same
    amount, so the ranking (and the AUC) stays identical; other effects change it.

    Returns
    -------
    pandas.DataFrame
        AUC of the fixed model on the test site after each distortion.
    """
    X, y, _ = simulate_classification(
        n_samples, random_state, n_sites=1, signal_strength=2, site_effect_strength=0
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=random_state
    )
    model = clone(LogisticRegression()).fit(X_train, y_train)
    rng = np.random.default_rng(random_state)
    effects = {
        "No site effect": X_test,
        "Location: same shift for every subject": X_test + np.array([3.0, -2.0]),
        "Scale: each feature multiplied differently": X_test * np.array([3.0, 0.3]),
        "Location varying across subjects": X_test + rng.normal(0, 1.5, X_test.shape),
        "Nonlinear: x + 0.3 x²": X_test + 0.3 * X_test**2,
    }
    table = pd.DataFrame(
        {
            "AUC (same model)": {
                k: roc_auc_score(y_test, model.decision_function(v))
                for k, v in effects.items()
            }
        }
    ).round(3)

    if plot:
        fig, axes = plt.subplots(1, len(effects), figsize=(5 * len(effects), 5))
        for ax, (name, Xe) in zip(axes, effects.items()):
            plot_scenario(
                ax,
                Xe,
                y_test,
                np.zeros(len(y_test), dtype=int),
                f"{name.split(':')[0]}\nAUC {table.loc[name, 'AUC (same model)']:.3f}",
                clf=model,
                fontsize=11,
            )
            ax.get_legend().remove()
        plt.tight_layout()
        plt.show()
    return table


# =============================================================================
# B3_N4: ComBat with site-class imbalance
# =============================================================================
HARMONIZATION_PIPELINES = {
    "raw": "No harmonization",
    "combat": "ComBat, target not protected",
    "combat_target_test_labels": "ComBat, target protected, test labels used [leak]",
    "combat_target_no_test_labels": "ComBat, target protected, test labels not used",
    "combat_target_all_data": "ComBat fitted on all data, target protected [leak]",
}


def evaluate_harmonization_pipelines(X, y, sites, clf, cv, pipelines=None):
    """Cross-validate a classifier after different ways of applying ComBat.

    In every fold ComBat is fitted on the training subjects only, except for
    ``"combat_target_all_data"``, the common mistake of harmonizing the whole
    dataset before cross-validation.

    Pipelines (keys of ``HARMONIZATION_PIPELINES``):

    - ``raw``: no harmonization.
    - ``combat``: ComBat without covariates; the target is not protected.
    - ``combat_target_test_labels``: target protected; test subjects are
      transformed with their true labels [leak: unknown in a real application].
    - ``combat_target_no_test_labels``: target protected; test subjects are
      transformed with the target set to the training proportion of Class 1,
      so no test label is used.
    - ``combat_target_all_data``: ComBat fitted on all subjects with their
      labels before the CV split [leak].

    The target is passed as a *continuous* covariate. For a 0/1 variable this
    gives exactly the same design matrix as a categorical covariate, but it lets
    us transform test subjects with the training proportion (e.g. 0.5) instead of
    a label.

    Returns
    -------
    proba : dict
        Out-of-fold probabilities of Class 1 for each pipeline.
    X_harm : dict
        Out-of-fold harmonized features for each pipeline (each subject is
        transformed by a model that did not see it, except for the all-data pipeline).
    """
    pipelines = list(HARMONIZATION_PIPELINES) if pipelines is None else list(pipelines)
    proba = {p: np.zeros(len(y)) for p in pipelines}
    X_harm = {p: np.zeros(X.shape, dtype=float) for p in pipelines}
    y_col = y.reshape(-1, 1).astype(float)

    if "combat_target_all_data" in pipelines:
        # LEAK: every subject and every label is used before the CV split.
        X_all = NeuroComBat().fit_transform(X, sites, continuous_covariates=y_col)

    for train, test in cv.split(X, y):

        def fit_predict(H_train, H_test, key):
            model = clone(clf).fit(H_train, y[train])
            proba[key][test] = model.predict_proba(H_test)[:, 1]
            X_harm[key][test] = H_test

        if "raw" in pipelines:
            fit_predict(X[train], X[test], "raw")

        if "combat" in pipelines:
            combat = NeuroComBat().fit(X[train], sites[train])
            fit_predict(
                combat.transform(X[train], sites[train]),
                combat.transform(X[test], sites[test]),
                "combat",
            )

        if {"combat_target_test_labels", "combat_target_no_test_labels"} & set(
            pipelines
        ):
            y_train = y_col[train]
            combat = NeuroComBat().fit(
                X[train], sites[train], continuous_covariates=y_train
            )
            H_train = combat.transform(
                X[train], sites[train], continuous_covariates=y_train
            )
            if "combat_target_test_labels" in pipelines:
                # LEAK: the true test labels are given to the harmonizer.
                H_test = combat.transform(
                    X[test], sites[test], continuous_covariates=y_col[test]
                )
                fit_predict(H_train, H_test, "combat_target_test_labels")
            if "combat_target_no_test_labels" in pipelines:
                # No test label: every test subject gets the training proportion of Class 1.
                y_unknown = np.full((len(test), 1), y_train.mean())
                H_test = combat.transform(
                    X[test], sites[test], continuous_covariates=y_unknown
                )
                fit_predict(H_train, H_test, "combat_target_no_test_labels")

        if "combat_target_all_data" in pipelines:
            fit_predict(X_all[train], X_all[test], "combat_target_all_data")

    return proba, X_harm


def _pipeline_rows(title, seed, y, sites, proba):
    """Pooled AUC and mean within-site AUC for each pipeline.

    within_site_auc = AUC of the same out-of-fold scores computed on the
    subjects of each site, averaged over sites (no model is trained per site).
    In B3_N4 it is only used under the conditions where it is interpretable:
    AUC, a homogeneous location effect and sites of equal size. See the
    limitations section of B3_N3.
    """
    rows = []
    for key, p in proba.items():
        pooled = roc_auc_score(y, p)
        within = float(np.mean(list(site_auc(y, p, sites).values())))
        rows.append(
            {
                "Scenario": title,
                "Pipeline": HARMONIZATION_PIPELINES[key],
                "seed": seed,
                "pooled_auc": pooled,
                "within_site_auc": within,
                "pooled_minus_within": pooled - within,
            }
        )
    return rows


def run_harmonization_scenario(
    title,
    n_samples,
    clf,
    cv,
    random_state,
    n_seeds=20,
    pipelines=None,
    n_sites=2,
    n_features=2,
    signal_type="blobs",
    plot=True,
    **sim_kwargs,
):
    """Simulate one scenario, compare the ComBat pipelines and repeat over seeds.

    For 2-feature data the out-of-fold harmonized features of the example dataset
    are plotted, one panel per pipeline.

    Returns
    -------
    pandas.DataFrame
        One row per seed and pipeline. The example dataset (``random_state``) is in
        ``results.attrs["example"]``.
    """
    X, y, sites = simulate_classification(
        n_samples, random_state, n_sites, n_features, signal_type, **sim_kwargs
    )
    proba, X_harm = evaluate_harmonization_pipelines(X, y, sites, clf, cv, pipelines)
    example = pd.DataFrame(_pipeline_rows(title, random_state, y, sites, proba))

    if plot and n_features == 2:
        n_panels = len(proba)
        fig, axes = plt.subplots(
            1, n_panels, figsize=(6.5 * n_panels, 6.5), sharex=True, sharey=True
        )
        for ax, (key, row) in zip(
            np.atleast_1d(axes), zip(proba, example.itertuples())
        ):
            label = HARMONIZATION_PIPELINES[key].replace(", ", ",\n", 1)
            plot_scenario(
                ax,
                X_harm[key],
                y,
                sites,
                f"{label}\npooled AUC {row.pooled_auc:.3f} | within-site {row.within_site_auc:.3f}",
                clf=clone(clf).fit(
                    X_harm[key], y
                ),  # decision boundary for visualization only
                fontsize=12,
            )
            ax.grid(alpha=0.3, color="black", linestyle="--")
        fig.suptitle(title, fontsize=15, fontweight="bold")
        plt.tight_layout()
        plt.show()

    counts = ", ".join(
        f"Site {s}: {n0} / {n1}"
        for s, (n0, n1) in class_counts_by_site(y, sites).items()
    )
    print(f"Subjects per site (Class 0 / Class 1): {counts}")
    print(f"Example dataset (random_state={random_state}), out-of-fold AUC:")
    print(
        example.set_index("Pipeline")[["pooled_auc", "within_site_auc"]]
        .round(3)
        .to_string()
    )

    rows = []
    for seed in range(n_seeds):
        Xs, ys, ss = simulate_classification(
            n_samples, seed, n_sites, n_features, signal_type, **sim_kwargs
        )
        p, _ = evaluate_harmonization_pipelines(Xs, ys, ss, clf, cv, pipelines)
        rows += _pipeline_rows(title, seed, ys, ss, p)
    results = pd.DataFrame(rows)
    results.attrs["example"] = example.to_dict("records")
    return results


def plot_pipeline_auc(results, title=None, ax=None):
    """Mean +/- SD across seeds of pooled and within-site AUC for each pipeline."""
    summary = results.groupby("Pipeline", sort=False)[
        ["pooled_auc", "within_site_auc"]
    ].agg(["mean", "std"])
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 0.8 * len(summary) + 2))
    ypos = np.arange(len(summary))
    for offset, (col, color, label) in zip(
        (-0.15, 0.15),
        (
            ("pooled_auc", "tab:blue", "pooled AUC"),
            ("within_site_auc", "tab:orange", "mean within-site AUC"),
        ),
    ):
        ax.errorbar(
            summary[(col, "mean")],
            ypos + offset,
            xerr=summary[(col, "std")],
            fmt="o",
            color=color,
            capsize=3,
            label=label,
        )
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(ypos, summary.index)
    ax.invert_yaxis()
    ax.set_xlim(0.2, 1.0)
    ax.set_xlabel(
        f"Cross-validated AUC (mean ± SD over {results.seed.nunique()} datasets)"
    )
    if title:
        ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    plt.tight_layout()
    return ax
