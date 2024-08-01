
[comment]: # (# Time sampling for physics-informed neural networks (PINN))
This is a repository of codes that implement special sampling and weighting schemes for Physics-informed Neural Networks (PINN). 

References : arxiv papers by G. Turinici 

paper 1: arXiv:2404.18780 "Optimal time sampling in physics-informed neural networks" https://arxiv.org/abs/2404.18780

paper 2 arXiv:2407.21642 "Lyapunov weights to convey the meaning of time in physics-informed neural networks" https://arxiv.org/abs/2407.21642


## This repository is the official implementation of "Lyapunov weights to convey the meaning of time in physics-informed neural networks"  arXiv:2407.21642

## Requirements

The following packages are required: tensorflow numpy scipy matplotlib pickle and their dependencies. The installation 
is depending on your precise environment but usually reads :

```setup
pip install -r tensorflow numpy scipy matplotlib pickle 
```

Note: your environment may require "pip3".

## Obtaining the results from the paper

### To train the model(s) in the paper for the Lorenz system, run this command:

```
python pinn_lorenz.py
```

Note: your environment may require "python3" (here and in all similar situations below).


### To obtain the confidence interval run this command:

```
python pinn_lorenz_confidence_interval.py
```


Note: this computation is very long.

### To train the model(s) in the paper for the Burgers' equation, run this command:

```
python pinn_burgers.py
```

## Playing with the procedure (requires coding)

To evaluate the procedure with different parameters you can run:

```train
python pinn_lorenz_test.py
```

You need to adapt parameters in that code after line 360 (the definition of the PhysicsInformedNeuralNetwork class) in the code. 
In the example below the first line instantiates the class, here using a new seed= 123.
Training is realized in the second line 2500 iterations (choose whatever you like). You can choose between three methods "Lyapunov lambda" is the one in this paper,
'causal' and 'cst_lambda' are the other papers referenced in Lorenz results section.


Results are plotted and saved in pdf starting from the third line.

```
pinn_lr005 = PhysicsInformedNeuralNetwork(random_seed=123,
    weights_update_type=['lyapunov_lambda','causal','cst_lambda'][0],
	learning_rate=0.01)
pinn_lr005.train(2500) 
pinn_lr005.plot_last_estimators()
pinn_lr005.plot_last_weights()
pinn_lr005.plot_loss_convergence()
pinn_lr005.plot_train_results()
```

Similar considerations work for the Burgers equation.
