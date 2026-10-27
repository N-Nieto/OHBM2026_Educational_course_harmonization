### Question

How is it possible that both scenarios have a similar overall performance? Where is the catch?

The overall metric is computed over all **subjects** together, so every subject counts the same, wherever it comes from. In both scenarios exactly half of the subjects (600 of 1200) come from sites without signal:

- Scenario 1: one bad site with 600 subjects, three good sites with 200 each.
- Scenario 2: three bad sites with 200 subjects each, one good site with 600.

The model therefore works for the same share of subjects in both scenarios, and the overall AUC is similar (0.728 vs 0.741).

The site-level picture is opposite: the model works in 3 of 4 sites in Scenario 1 but in only 1 of 4 sites in Scenario 2. If we had only reported the overall performance, we could not have uncovered this. Report the per-site metrics together with the number of subjects of each class in each site, and read them site by site. 

-------------------------------------
While it is true that the discrimination capability of our model is almost the same in both scenarios, would you recommend using the developed model in Site 0?

- The answer is of course no: we don't want to use a model to make important decisions when we know it is not performing well in that site. After we find these biases, we can dig deeper and find out *why* the model was performing badly in this site. In our scenario, we removed the true biological signal, but it can also be the case that the biological signal is masked by a strong EoS. 

-------------------------------------
In this example, no EoS were simulated in any of our sites. What do you think could happen if different EoS were present in different sites?

- As we saw in B3_N2, it depends not only on the EoS but also on the class imbalance across sites.
- If the classes are balanced in every site, the EoS acts as noise. Sites with a strong EoS become noisier and can behave like "bad" sites.
- If the classes are imbalanced across sites, the EoS becomes informative about the class. For example, with one big site of healthy controls and many small sites of patients, the model can learn to recognize the control site instead of the disease. The overall metric then looks good. In the extreme case, classifying the site is the same as classifying the target from the model's perspective.
