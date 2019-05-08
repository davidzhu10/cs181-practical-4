#cs181-practical-4

To run, run "python stub.py"


### Hyperparameter Tuning

Selection of optimal hyperparameters was performed by calculating two scoring metrics for a series of trials, each consisting of 20 training runs. The scoring metrics, one weighted according to negative Gaussian distribution and the other logarithmically, both accounted for a training burn-in period and were used to find the $\alpha$ (learning rate), $\epsilon$ (randomization fraction), $\gamma$ (Q-learning discount factor) values that resulted in most effective agents, as well as the best decay functions for $\alpha$ and $\epsilon$ over the epochs. The code for this can be found [here](stub_j.py).
