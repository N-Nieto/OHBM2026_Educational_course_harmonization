### Question

- How is it possible that they have an similar overall performance? Where is the catch?

The performance obtained in each site is not comparable because they have different number of samples!

In the first scenario, the bad site is 3 times bigger than the good sites and vice versa for the second scenario.

If we had only reported the overall performance, we would not be able to unravel the model's behavior.

- In this example, no Eos was used simulated in any of our sites. What do you think it could happen if different EoS are presented in different sites?

As we saw before, it will depend not only on the EoS but also in the site class imbalance. If the EoS acts as noise, the presence of EoS will create noisier sites, which could act as "bad" sites. If we have a big site class imbalance, and for example combined with having one big site (for example with healthy control) and many small sites with patients, the model will only learn the particular EoS of that site. In the extreme case, classifying the site  will be the same as classify the target, from the model's perspective. 

