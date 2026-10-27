"""Helper functions for the Block 3 notebooks.

Which notebook uses what
------------------------
B3_N2 (effects of site in ML)        -> run_classification_scenario, summarize_seed_results
B3_N3 (metrics by site)              -> run_metrics_by_site_scenario, plot_metrics_by_site
B3_N4 (ComBat with class imbalance)  -> run_harmonization_scenario, plot_pipeline_scores,
                                        summarize_seed_results
B3_N1 (alternative version)          -> simulate_multisite_data_with_covariates, run_fpr_simulation

All data are simulated with ``make_multisite_classification`` from uniharmony.
Its argument ``balance_per_site`` is one ``[proportion of Class 0, proportion of Class 1]``
list per site, e.g. ``[[0.1, 0.9], [0.9, 0.1]]``. A flat list such as ``[0.1, 0.9]``
is applied to *every* site, so it creates no site-class imbalance.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_predict, cross_val_score
from uniharmony import verbosity
from uniharmony.datasets import make_multisite_classification
from uniharmony.metrics import report_metrics_by_site
from uniharmony.plot import plot_decision_boundary_2d

sns.set_theme(style="whitegrid")

# Hide uniharmony's info/debug messages.
verbosity("error")

# Imported after setting the verbosity: uniharmony.combat creates its logger when
# it is imported, so importing it earlier would print debug messages.
from uniharmony.combat import NeuroComBat  # noqa: E402


# =============================================================================
# Small helpers used by several notebooks
# =============================================================================
def plot_scenario(ax, X, y, sites, title, clf=None, fontsize=13):
    """Scatter plot of the first two features.

    Colour = site, marker = class. If a fitted classifier is given (and the data
    have 2 features), its decision boundary is drawn as a line.
    """
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
        hue_order=[f"Site {s}" for s in np.unique(sites)],
        style_order=["Class 0", "Class 1"],
    )
    ax.set_title(title, fontsize=fontsize, fontweight="bold")
    if clf is not None and X.shape[1] == 2:
        plot_decision_boundary_2d(ax, clf)


def summarize_seed_results(results, by="Scenario"):
    """Mean and SD of every metric over the repeated datasets (seeds).

    ``results`` is a list of the tables returned by ``run_classification_scenario``
    (B3_N2) or ``run_harmonization_scenario`` (B3_N4): one row per dataset.
    ``by`` names the column(s) that identify a scenario, e.g. ["Scenario", "Pipeline"].
    """
    if not isinstance(results, (list, tuple)):
        results = [results]
    df = pd.concat(results, ignore_index=True)
    by = [by] if isinstance(by, str) else list(by)
    metric_columns = [c for c in df.columns if c not in by + ["seed"]]
    # sort=False keeps the scenarios in the order in which they were run
    return df.groupby(by, sort=False)[metric_columns].agg(["mean", "std"]).round(3)


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
    """Simulate the same scenario with seeds 0, 1, ..., n_seeds - 1.

    For each simulated dataset, the AUC is averaged over the cross-validation
    folds. Returns one row per dataset.
    """
    rows = []
    for seed in range(n_seeds):
        X, y, _ = make_multisite_classification(
            n_samples=n_samples,
            n_sites=n_sites,
            n_features=n_features,
            signal_type=signal_type,
            random_state=seed,
            **sim_kwargs,
        )
        auc = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc").mean()
        rows.append({"Scenario": title, "seed": seed, "auc": auc})
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
    """One B3_N2 scenario: simulate, cross-validate, plot, then repeat over seeds.

    Extra keyword arguments (signal_strength, site_effect_strength,
    site_effect_type, balance_per_site, ...) go to make_multisite_classification.
    Returns the per-seed table of ``seed_summary``.
    """
    # 1. Simulate the example dataset.
    X, y, sites = make_multisite_classification(
        n_samples=n_samples,
        n_sites=n_sites,
        n_features=n_features,
        signal_type=signal_type,
        random_state=random_state,
        **sim_kwargs,
    )

    # 2. Cross-validated AUC: the model is always tested on subjects it has not seen.
    auc = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc").mean()

    # 3. Plot the data. The decision boundary comes from a model fitted on all
    #    subjects; it is only for the picture, not for the AUC.
    _, ax = plt.subplots(figsize=(10, 8))
    plot_scenario(
        ax,
        X,
        y,
        sites,
        f"{title}\nMean CV AUC for example dataset (random_state={random_state}): {auc:.3f}",
        clf=clone(clf).fit(X, y),
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()

    # 4. Repeat the scenario on other random datasets to see how stable the AUC is.
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
    """Simulate "good" sites (features predict the class) and "bad" sites (they don't).

    All sites are drawn from the same problem without effects of site, so the
    features look the same everywhere. In the bad sites we shuffle the class
    labels: there, features and class are unrelated (think of unreliable
    diagnoses or unusable scans).
    """
    X, y, sites = make_multisite_classification(
        n_sites=len(n_samples_per_site),
        n_samples=list(n_samples_per_site),  # one number of subjects per site
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
        y[idx] = rng.permutation(y[idx])  # shuffle the labels inside the bad site
    return X, y, sites


def run_metrics_by_site_scenario(
    n_samples_per_site,
    bad_sites,
    clf,
    cv,
    random_state,
    metrics,
    signal_strength=1.0,
    n_features=10,
):
    """One B3_N3 scenario: overall and per-site metrics of a single model.

    Returns
    -------
    results : dict
        Output of ``report_metrics_by_site``: {metric name: {"overall": value, site: value}}.
    n_per_site : dict
        Number of subjects in each site.
    """
    # 1. Simulate the good and bad sites.
    X, y, sites = make_good_and_bad_sites(
        n_samples_per_site, bad_sites, signal_strength, n_features, random_state
    )

    # 2. ONE model trained on all sites. cross_val_predict gives every subject a
    #    prediction from a model that did not see that subject (out-of-fold).
    #    We keep the probability of Class 1, which the AUC needs.
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]

    # 3. Compute the metrics on all subjects ("overall") and on each site's subjects.
    #    Metrics that need predicted classes (e.g. balanced accuracy) threshold
    #    the probabilities at 0.5 inside report_metrics_by_site.
    results = report_metrics_by_site(y, proba, sites, metrics)

    n_per_site = {s: int(np.sum(sites == s)) for s in np.unique(sites)}
    return results, n_per_site


def plot_metrics_by_site(
    ax, results, metric_name, n_per_site, title, color="steelblue"
):
    """Bar plot: one bar per site (labelled with its number of subjects) + the overall value.

    The site values are shown one by one and never averaged: an average would
    give a small site the same weight as a large one.
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


# =============================================================================
# B3_N4: ComBat with site-class imbalance
# =============================================================================
PIPELINES = [
    "No harmonization",
    "ComBat, target not protected",
    "ComBat, target protected [leak]",
    "ComBat fitted on all data [leak]",
]


def cross_validate_pipeline(pipeline, X, y, sites, clf, cv):
    """Cross-validate a classifier after harmonizing the data with one of PIPELINES.

    In every fold: (1) harmonize the training and test subjects, (2) train the
    classifier on the harmonized training subjects, (3) predict the test subjects.

    Returns
    -------
    proba : array
        Out-of-fold probability of Class 1 for every subject.
    X_harmonized : array
        Out-of-fold harmonized features (what the classifier saw for each test subject).
    """
    proba = np.zeros(len(y))
    X_harmonized = np.zeros(X.shape)
    y_column = y.reshape(-1, 1)  # ComBat expects covariates as a column

    if pipeline == "ComBat fitted on all data [leak]":
        # A common mistake, made on purpose: harmonize ALL subjects once, with ALL
        # their labels as covariate, and only then split into folds. The test
        # subjects' labels have been used to build their features.
        X_all = NeuroComBat().fit_transform(X, sites, categorical_covariates=y_column)

    for train, test in cv.split(X, y):
        if pipeline == "No harmonization":
            X_train, X_test = X[train], X[test]

        elif pipeline == "ComBat, target not protected":
            # Learn the site effects on the training subjects only...
            combat = NeuroComBat().fit(X[train], sites[train])
            # ...and remove them from training and test subjects. Only the site of
            # each subject is needed, so no label is used.
            X_train = combat.transform(X[train], sites[train])
            X_test = combat.transform(X[test], sites[test])

        elif pipeline == "ComBat, target protected [leak]":
            # Pass the class as a covariate so that ComBat keeps the class difference.
            combat = NeuroComBat().fit(
                X[train], sites[train], categorical_covariates=y_column[train]
            )
            X_train = combat.transform(
                X[train], sites[train], categorical_covariates=y_column[train]
            )
            # LEAK: ComBat also needs the covariate to transform the test subjects,
            # i.e. their true labels, which are unknown in a real application.
            X_test = combat.transform(
                X[test], sites[test], categorical_covariates=y_column[test]
            )

        elif pipeline == "ComBat fitted on all data [leak]":
            X_train, X_test = X_all[train], X_all[test]

        else:
            raise ValueError(f"Unknown pipeline: {pipeline}. Choose from {PIPELINES}")

        model = clone(clf).fit(X_train, y[train])
        proba[test] = model.predict_proba(X_test)[:, 1]
        X_harmonized[test] = X_test

    return proba, X_harmonized


def score_table(proba_per_pipeline, y, sites, scenario, seed):
    """One row per pipeline: balanced accuracy over all subjects and in each site.

    The classifier predicts Class 1 when its probability is >= 0.5. Balanced
    accuracy is the mean of sensitivity and specificity, so it depends on where
    the decision boundary falls, not only on how well the subjects are ranked.
    The per-site values use the same predictions (one model trained on all
    sites); they are only computed on each site's subjects, and never averaged.
    """
    rows = []
    for pipeline, proba in proba_per_pipeline.items():
        y_pred = (proba >= 0.5).astype(int)  # predicted class
        row = {
            "Scenario": scenario,
            "Pipeline": pipeline,
            "seed": seed,
            "pooled_bacc": balanced_accuracy_score(y, y_pred),
        }
        for s in np.unique(sites):
            in_site = sites == s
            row[f"site_{s}_bacc"] = balanced_accuracy_score(y[in_site], y_pred[in_site])
        rows.append(row)
    return pd.DataFrame(rows)


def run_harmonization_scenario(
    title,
    n_samples,
    clf,
    cv,
    random_state,
    n_seeds=20,
    pipelines=PIPELINES,
    n_sites=2,
    n_features=2,
    plot=True,
    **sim_kwargs,
):
    """One B3_N4 scenario: compare the pipelines on an example dataset, then over seeds.

    Extra keyword arguments (signal_strength, site_effect_strength,
    site_effect_type, balance_per_site, ...) go to make_multisite_classification.
    Returns one row per seed and pipeline (see ``score_table``).
    """
    # 1. Simulate the example dataset and print how many subjects of each class each site has.
    X, y, sites = make_multisite_classification(
        n_samples=n_samples,
        n_sites=n_sites,
        n_features=n_features,
        signal_type="blobs",
        random_state=random_state,
        **sim_kwargs,
    )
    counts = ", ".join(
        f"Site {s}: {np.sum(y[sites == s] == 0)} / {np.sum(y[sites == s] == 1)}"
        for s in np.unique(sites)
    )
    print(f"Subjects per site (Class 0 / Class 1): {counts}")

    # 2. Run every pipeline on the example dataset.
    results = {p: cross_validate_pipeline(p, X, y, sites, clf, cv) for p in pipelines}
    example = score_table(
        {p: proba for p, (proba, _) in results.items()}, y, sites, title, random_state
    )
    print(f"Example dataset (random_state={random_state}), balanced accuracy:")
    print(
        example.drop(columns=["Scenario", "seed"])
        .set_index("Pipeline")
        .round(3)
        .to_string()
    )

    # 3. Plot the harmonized test features of each pipeline (2-feature data only).
    #    The line is the decision boundary of a classifier fitted on these features.
    if plot and n_features == 2:
        fig, axes = plt.subplots(
            1,
            len(pipelines),
            figsize=(6.5 * len(pipelines), 6.5),
            sharex=True,
            sharey=True,
        )
        for ax, (pipeline, (proba, X_harmonized)), row in zip(
            np.atleast_1d(axes), results.items(), example.itertuples()
        ):
            site_scores = " | ".join(
                f"Site {s} {getattr(row, f'site_{s}_bacc'):.3f}"
                for s in np.unique(sites)
            )
            name = pipeline.replace(", ", ",\n", 1)  # break long names over two lines
            plot_scenario(
                ax,
                X_harmonized,
                y,
                sites,
                f"{name}\nbACC pooled {row.pooled_bacc:.3f} | {site_scores}",
                clf=clone(clf).fit(X_harmonized, y),  # boundary for the picture only
                fontsize=12,
            )
        fig.suptitle(title, fontsize=15, fontweight="bold")
        plt.tight_layout()
        plt.show()

    # 4. Repeat on other random datasets (seeds 0, 1, ...) to see how stable the results are.
    tables = []
    for seed in range(n_seeds):
        Xs, ys, ss = make_multisite_classification(
            n_samples=n_samples,
            n_sites=n_sites,
            n_features=n_features,
            signal_type="blobs",
            random_state=seed,
            **sim_kwargs,
        )
        proba_per_pipeline = {
            p: cross_validate_pipeline(p, Xs, ys, ss, clf, cv)[0] for p in pipelines
        }
        tables.append(score_table(proba_per_pipeline, ys, ss, title, seed))
    return pd.concat(tables, ignore_index=True)


def plot_pipeline_scores(results, title=None):
    """Box plots over seeds: balanced accuracy of each pipeline, pooled and in each site."""
    # Long format: one row per (seed, pipeline, evaluated group).
    score_columns = [c for c in results.columns if c.endswith("_bacc")]
    long = results.melt(
        id_vars=["Pipeline", "seed"],
        value_vars=score_columns,
        var_name="Evaluated on",
        value_name="Balanced accuracy",
    )
    long["Evaluated on"] = long["Evaluated on"].map(
        lambda c: "All subjects (pooled)" if c == "pooled_bacc" else f"Site {c.split('_')[1]}"
    )
    # Break the long pipeline names so they fit under the boxes.
    long["Pipeline"] = long["Pipeline"].str.replace(", ", ",\n", regex=False)

    # Same colours as the scatter plots: Site 0 blue, Site 1 orange, ...; pooled in grey.
    groups = long["Evaluated on"].unique()
    palette = {g: "0.6" if g.startswith("All") else sns.color_palette()[int(g.split()[-1])]
               for g in groups}

    _, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=long, x="Pipeline", y="Balanced accuracy", hue="Evaluated on",
                palette=palette, ax=ax)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)  # chance level
    ax.set_ylim(0.2, 1.0)
    ax.set_xlabel("")
    n_seeds = results.seed.nunique()
    ax.set_title(
        title or f"Balanced accuracy over {n_seeds} datasets", fontweight="bold"
    )
    ax.legend(title="Evaluated on", loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=len(score_columns), frameon=False)
    plt.tight_layout()
    return ax


# =============================================================================
# B3_N1 (alternative version) helpers
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
