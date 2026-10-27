### Question

What do you think would happen in Section 1 if an EoS were also simulated?

With site-class imbalance, each site is mostly one class: Site 0 is 90% Class 1 and Site 1 is 90% Class 0. An effect of site shifts all the subjects of a site in the same direction, so the shift ends up aligned with the class. The EoS and the class become **confounded**: from the data alone, a difference between the sites can be either the scanner or the biology.

What happens to each pipeline:

- **No harmonization.** The model can now use two sources of information: the real signal and the site shift (as a shortcut). The pooled balanced accuracy goes **up**, because recognizing the site already predicts most of the labels (shortcut learning, as in B3_N2). Inside each site, however, the site shift does not help, and the decision boundary is no longer placed for the real signal, so the balanced accuracy in each site goes **down**. Pooled and per-site values disagree: a warning sign.

- **ComBat, target not protected.** ComBat removes the difference between the sites. It cannot tell which part comes from the scanner and which part comes from the classes, so it removes both: the EoS **and** the real signal. The result is the same as in Section 1 without EoS. The sites are moved on top of each other, and the decision boundary crosses the middle of each site.

- **ComBat, target protected [leak].** With the class as covariate, ComBat estimates the site effect separately from the class effect. It removes the EoS and keeps the signal, so the results go back to those of Section 1 without EoS. But to harmonize the test subjects it needs their true labels, which are unknown in a real ML application, so this pipeline cannot be used.

- **ComBat fitted on all data [leak].** Same result as the previous pipeline here, but with an even larger leak: the test subjects' labels were used before the cross-validation split. With 1000 subjects and 2 features this barely changes the numbers, but the pipeline is still invalid (see Section 2 for when it creates performance from nothing).

### Take-away

When the classes are imbalanced across sites, the EoS and the class cannot be separated without knowing the labels. Inside an ML pipeline this leaves two bad options for standard ComBat: lose the signal (target not protected) or leak the test labels (target protected). Nieto et al. (2026) [1] study this problem in detail and propose PrettYharmonize, a leakage-free way to use ComBat-based harmonization in ML pipelines.

# Citations:
[1] Nieto, N., Eickhoff, S.B., Jung, C. et al. Impact of leakage on data harmonization in machine learning pipelines in class imbalance across sites. Neurocomputing 680, 133146 (2026). https://doi.org/10.1016/j.neucom.2026.133146
