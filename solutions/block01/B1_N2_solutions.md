# Questions

## Section 1.4: Sample differences between scanners and sites

**As you work through the simulations, consider:**

- Which variables are true biological effects?

    Biological effects are variables that reflect genuine differences between individuals. Examples include age, sex, body size, disease status, and genetic factors. These variables are expected to influence imaging measurements through underlying biological mechanisms.

- Which variables might act as confounds?

    A confound is a variable that influences both the predictor and the outcome. In this example, site can become a confound if certain participant groups are disproportionately recruited at particular scanners. Age, sex, or disease status can also act as confounds if they are unevenly distributed across sites. Additionally, in the case of imaging, other confounds that may differ systematically by site or with a variable such as age or sex may include head motion, head position, head size scaling among others.

- What happens if site and biological variables become correlated?

    When site and biological variables are correlated, it becomes difficult to determine whether observed differences are due to biology or scanner effects. This can lead to biased estimates, spurious associations, or the masking of genuine biological relationships.

- Do the simulated relationships look biologically realistic?

    The simulated relationships should broadly resemble patterns often observed in neuroimaging studies, such as age-related changes or population variability. However, the simulation is intentionally simplified and does not capture all sources of biological complexity present in real datasets.

## Section 1.5: Non-biological site differences

- Which effects are biological?

    Biological effects are the relationships introduced through participant characteristics such as age, sex, disease status, or other physiological variables. These effects represent the signals that researchers are typically interested in studying.

- Which are scanner-related?

    Scanner-related effects include systematic shifts in measurement means, changes in variance, intensity scaling differences, and other technical factors introduced by acquisition hardware or imaging protocols. These effects do not reflect true biological variation.

- How difficult would it be to separate these effects in a real study?

    Separating biological and scanner-related effects can be challenging, particularly when participant characteristics are unevenly distributed across sites. In practice, researchers often use statistical adjustment, harmonisation methods such as ComBat, travelling-head datasets, and careful study design to reduce this problem. However, complete separation is rarely perfect, and residual site effects may still remain after correction.