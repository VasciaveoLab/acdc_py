# acdc_py 🤘
[![pipy](https://img.shields.io/pypi/v/acdc-py?color=informational)](https://pypi.org/project/acdc-py/1.1.0/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15099170.svg)](https://doi.org/10.5281/zenodo.15099170)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/acdc-py)](https://pepy.tech/project/acdc-py)
[![Documentation Status](https://readthedocs.org/projects/acdc/badge/?version=latest)](https://acdc.readthedocs.io/en/latest/?badge=latest)


**A**utomated **C**ommunity **D**etection of **C**ell Populations in Python  

This repo contains the current Python implementation of ACDC, an optimization-based framework to automatize clustering of cell populations from scRNA-seq data using community detection algorithms. 
```acdc_py``` is currently **under development** and new functionalities will be released, following completion and benchmarking. 
```acdc_py``` is deployed as a Python package and fully compatible with ```Scanpy```.

<div align="center">
  <img width="240" alt="image" src="https://github.com/califano-lab/acdc-beta/assets/92543296/09feabaf-d868-48d7-b830-933210db6005">
  <img width="240" alt="image" src="https://github.com/califano-lab/acdc-beta/assets/92543296/28952fc8-841e-4d3a-80bd-d1a3a92c5a07"> 
  <img width="240" alt="image" src="https://github.com/califano-lab/acdc-beta/assets/92543296/41678fd3-c583-4b7b-939e-dbd443d44c97">
</div>

- Several graph-based clustering algorithms are available within ```acdc_py```, including Leiden and Louvain. 
- 2 optimization routines for parameter tuning are available, Grid Search and(generalized) Simulated Annealing.
- Optimization variables include the number of nearest neighbors, *k*, resolution, *res*, and the number of principal components, *PCs*.
- Several objective functions are available, including the Silhouette Score (default).


New releases will expand functionalities to new features, including the possibility to iteratively sub-cluster cell populations to find fine grain and biologically meaningful clustering solutions.

**To receive updates when novel functionalities are released, feel free to add your email to the following form:** https://forms.gle/NCRPJPmXzfbrMH7U7

``` 
STAY TUNED FOR UPDATES AND NOVEL DEVELOPMENTS!🤘🏾
```

**Please, be aware that while this project is "work in progress" and outcomes are continuously benchmarked, cross-platform compability might not yet be guaranteed. 


# Installation 
### pypi
```shell
pip install acdc-py
```
### local
```shell
git clone https://github.com/califano-lab/acdc_py/
cd acdc_py
pip install -e .
```

... Start playing around! 🎸



# References
1. Kiselev, VY, Andrews, TS, Hemberg, M. (2019) Challenges in unsupervised clustering of single-cell RNA-seq data. Nat Rev Genet 20, 273–282.
2. Blondel, V D, Guillaume, J, Lambiotte, R, Lefebvre, E (2008). Fast unfolding of communities in large networks". Journal of Statistical Mechanics: Theory and Experiment. (10) P10008.
3. Satija R, Farrell JA, Gennert D, Schier AF, Regev A (2015). “Spatial reconstruction of single-cell gene expression data.” Nature Biotechnology, 33, 495-502. 
4. Traag, V.A., Waltman, L. & van Eck, N.J. (2019) From Louvain to Leiden: guaranteeing well-connected communities. Sci Rep 9, 5233. 
5. Xiang, Y., Gubian, S., Suomela, B.P., & Hoeng, J. (2013). Generalized Simulated Annealing for Global Optimization: The GenSA Package. R J., 5, 13.


# Contacts

Alessandro Vasciaveo - avasciaveo@sbpdiscovery.org

Alexander Wang - aw3436@cumc.columbia.edu  

Luca Zanella - lz2841@cumc.columbia.edu  


