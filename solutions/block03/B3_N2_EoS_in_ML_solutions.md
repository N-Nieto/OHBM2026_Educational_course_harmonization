# EoS in ML: Noise or Signal?

Effect of Site (EoS) is typically dismissed as nuisance variation: batch effects, instrument drift, or demographic differences we'd rather ignore. But from a machine learning perspective, EoS isn't always noise. Sometimes it can act as a shortcut.

The critical factor is class imbalance across sites. When target labels are unevenly distributed between sites, EoS becomes confounded with the true outcome. The model doesn't learn the pattern you want; it learns which site the sample came from, using site ID as a shortcut. Performance appears strong, but it's an artifact. The model is cheating.

This is not a problem with the algorithm, ML models are greedy and will take every possible correlation. To overcome this problem, we need to revisit the experimental design. If site predicts class better than your features do, the model will exploit that shortcut every time.

The takeaway: before trusting cross-site performance, **always** check the label distribution per site. Balanced sites reduce confounding. Imbalanced sites demand careful stratification or site-aware modeling. Otherwise, your "generalizable" model may just be memorizing the map.

### Question

- After harmonizing our data, how could we distinguish whether the harmonization model removed only the EoS, or also removed the real signal confounded with the sites?

From the performance alone, we cannot: a drop in performance after harmonization might happened because we removed important signal *or* because the model is now not using the EoS as *shortcut*. If the performance increased, it can be the case that we successfully removed the EoS that were acting as *noise*, but we need to be extra careful that we did not leak any information to the models in the process.

1. **How confounded are site and class?** The stronger the site–class imbalance, the less any method can separate the two, and the more cautious the interpretation must be.

2. **Can the site still be predicted from the harmonized features?** If a classifier can no longer tell the sites apart, the EoS was removed. This says nothing about the real signal, so it must be combined with the other checks.