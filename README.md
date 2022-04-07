# ANFORE: [A]utoregressive neural [N]etworks [FOR] [E]pidemics inference problems

Autoregressive neural network approach to solve epidemic inference problems, like the patient zero problem or the inference of parameters of the propagation model. Up to now it is implemented to support SIR compartimental model. More complicated compartimental model can be added.
ANNforE can compute the probability to each individuals to be susceptible, infected or recovered at a given time from a list of contacts and partial observations.
ANNforE, in the same time, can infer the parameters of the propagation model (like the probability of infection <span>&lambda;</span>).

The apporach is based on the autoregressive probability apporoximation of the postieror probability of the inference problem. See the [work](https://arxiv.org/abs/2111.03383).

## setup

Clone the repo and type: 
```
cd annfore 
pip install .
```

## run

See [example](annfore/examples/first_test.ipynb) 

### to cite this repository

Biazzo, I., Braunstein, A., Dall'Asta, L. and Mazza, F., 2021. Epidemic inference through generative neural networks. arXiv preprint [arXiv:2111.03383](https://arxiv.org/abs/2111.03383).

