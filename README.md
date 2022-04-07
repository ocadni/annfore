# ANNFORE: [A]utoregressive [N]eural [N]etworks [FOR] [E]pidemics inference problems

The repository contains the code for an autoregressive neural network approach to solve epidemic inference problems on contact newtorks. The patient zero problems, risk assmement or the inference of the infectivity of class of individuals are important examples.

Up until now ANNFORE supports the SIR compartimental model on contact networks, more complicated compartimental model can be easly added.

ANNFORE can compute the probability to each individuals to be susceptible, infected or recovered at a given time from a list of contacts and partial observations.
At the same time, it can infer the parameters of the propagation model (like the probability of infection <span>&lambda;</span>).

The approach is based on the autoregressive probability apporoximation of the postieror probability of the inference problem. See [here](https://arxiv.org/abs/2111.03383) for more details.

## Install the code

Clone the repo and type: 
```
cd annfore 
pip install .
```

## Examples to run

See [example](./annfore/examples/first_test.ipynb) 

## Reference
If you use the repository, please cite: 

Biazzo, I., Braunstein, A., Dall'Asta, L. and Mazza, F., 2021. Epidemic inference through generative neural networks. arXiv preprint [arXiv:2111.03383](https://arxiv.org/abs/2111.03383).

