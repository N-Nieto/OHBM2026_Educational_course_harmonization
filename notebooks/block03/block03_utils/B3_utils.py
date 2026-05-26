import numpy as np
from uniharmony import verbosity
from scipy.stats import ttest_ind

verbosity("error")

from uniharmony.combat import NeuroComBat

harm_model = NeuroComBat()


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
