# annfore: [a]utoregressive [n]eural [n]etworks [for] [e]pidemics inference problems [![DOI](https://zenodo.org/badge/405138309.svg)](https://zenodo.org/badge/latestdoi/405138309)


The repository contains the code for an autoregressive neural network approach to solve epidemic inference problems on contact networks. The patient zero problems, risk assessment, or the inference of the infectivity of a class of individuals are important examples.

annfore supports the SIR compartmental model on contact networks; more complicated compartmental models can be easly added.

annfore can compute the probability of each individual being susceptible, infected, or recovered at a given time from a list of contacts and partial observations.
At the same time, it can infer the parameters of the propagation model (like the probability of infection <span>&lambda;</span>).

The approach is based on the autoregressive probability approximation of the posterior probability of the inference problem. See [here](https://doi.org/10.1038/s41598-022-20898-x) for more details.

## Install the code

Clone the repo and type: 
```
cd annfore 
pip install .
```

## Examples to run

See [example](./annfore/examples/first_example.ipynb) 

## Reference
If you use the repository, please cite: 

Biazzo, I., Braunstein, A., Dall’Asta, L. and Mazza, F. A Bayesian generative neural network framework for epidemic inference problems. Sci Rep 12, 19673 (2022). [https://doi.org/10.1038/s41598-022-20898-x](https://doi.org/10.1038/s41598-022-20898-x)


