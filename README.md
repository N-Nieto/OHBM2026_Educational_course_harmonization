[<img src="images/ohbmofficial_cover.jpeg" width="800">](https://www.humanbrainmapping.org/)
# 🧠 OHBM 2026 Educational Course. Data harmonization for neuroscientific research: Theory, challenges, and applications.


## [Nicolás Nieto](https://github.com/N-Nieto)[✉️](n.nieto@fz-juelich.de), [Johanna Bayer](https://github.com/likeajumprope), [Gaurav Bhalerao](https://github.com/gvbhalerao591), [Emma Prevot](https://github.com/emmaprevot), [Jacob Turnbull](https://github.com/Jake-Turnbull).

[![OHBM](https://img.shields.io/badge/OHBM-2026-EA9500?style=flat-square)](https://www.ohbmbrainmapping.com/) [![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
---

## 📋 Overview

Welcome to the official repository for the **OHBM 2026 Educational Course: Data harmonization for neuroscientific research: Theory, challenges, and applications.**. This course provides a comprehensive, hands-on journey from understanding the fundamental challenges of site effects in neuroimaging data to implementing state-of-the-art harmonization techniques.

---

### 🎯 Learning Objectives

By the end of this course, you will be able to:

- **Diagnose** site effects and batch artifacts in multi-site neuroimaging datasets
- **Implement** location-scale harmonization methods (ComBat, ComBat-GAM, CovBat)
- **Evaluate** harmonization quality using appropriate metrics and validation strategies
- **Handle** longitudinal and test-retest data with specialized techniques (Long-Combat, BART)
- **Integrate** harmonization into machine learning pipelines and statistical analyses
- **Apply** advanced alternatives including deep learning and normative modeling approaches

---

## 🗂️ Repository Structure

The course is organized into four progressive blocks, each containing interactive Jupyter notebooks, simulated datasets, and supplementary materials.

``` lua
📦 OHBM2026-harmonization-course
├── 📁 data/                    # Simulated and example datasets
├── 📁 notebooks/               # Jupyter notebooks by block
│   ├── block01_eos/            # Effects of Site Introduction
│   ├── block02_harmonization/  # Harmonization General
│   ├── block03_ls/             # Location-Scale Methods
│   └── block04_advanced/       # Alternatives & Future Directions
├── 📁 utils/                   # Helper functions and visualization tools
├── 📁 solutions/               # Solutions to exercises
└── 📄 requirements.txt         # Python dependencies
```


## Running the Notebooks on Binder.

You can run these notebooks instantly online without any local setup using Binder.

1.  **Click the badge** to launch the Binder environment:
    [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/N-Nieto/OHBM2026_Educational_course_harmonization/HEAD)

2.  **Wait for the environment to build.** This may take a few minutes. Binder is creating a live, interactive server with all the necessary packages and data pre-installed. If the environment exist, it will automatically connect.

3.  **Start exploring!** Once the Jupyter interface loads, navigate to the desired notebook (`*.ipynb` file) and click on it to open and run it.


[<img src="images/OHBM_2026_logo.png">](https://humanbrainmapping.org/i4a/pages/index.cfm?pageid=4293)