# EoS in ML: Noise or Signal?

Effect of Site (EoS) is typically dismissed as nuisance variation—batch effects, instrument drift, or demographic differences we'd rather ignore. But from a machine learning perspective, EoS isn't always noise. Sometimes it's signal.

The critical factor is class imbalance across sites. When target labels are unevenly distributed between sites, EoS becomes confounded with the true outcome. The model doesn't learn the pattern you want, it learns which site the sample came from. Performance appears strong, but it's an artifact. The model is cheating.

This isn't a bug in the algorithm; it's a failure in experimental design. If site predicts class better than your features do, the model will exploit that shortcut every time.

The takeaway: before trusting cross-site performance, check the label distribution per site. Balanced sites reduce confounding. Imbalanced sites demand careful stratification or site-aware modeling. Otherwise, your "generalizable" model may just be memorizing the map.