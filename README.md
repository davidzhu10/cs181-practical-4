# cs181-practical-4

To see our optimal model in action, run `python bestmodel.py`. Refer to code [here](bestmodel.py).

To use the SARSA agent with graphs of average score/epoch split by gravity, run `python stub_bill.py`.

To visualize some of our graphs, interpolated and split by gravity, run `python interpolation_graph.py`.

Note that all this code runs for Python 3 and requires the `pygame` module, which can be installed using the `pip` package manager.

### Hyperparameter Tuning

Selection of optimal hyperparameters was performed by calculating two scoring metrics for a series of trials, each consisting of 20 training runs. The scoring metrics, one weighted according to negative Gaussian distribution and the other logarithmically, both accounted for a training burn-in period and were used to find the &alpha; (learning rate), &epsilon; (randomization fraction), &gamma; (Q-learning discount factor) values that resulted in most effective agents, as well as the best decay functions for &alpha; and &epsilon; over the epochs. The code for this can be found [here](stub_j.py). The full results table is included below.

![results](results.jpg)
