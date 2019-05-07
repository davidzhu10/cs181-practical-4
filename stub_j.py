# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg
import pandas as pd
from enum import Enum

import matplotlib.pyplot as plt
from scipy.stats import norm
import math

from SwingyMonkey import SwingyMonkey


def sigmoid(init, epoch, shift):
    return init * (1 - 1 / (1 + math.exp(-(0.2 * epoch - shift))))

alphas = [0.1, 0.2, 0.5, 1]
epsilons = [0.01, 0.02, 0.05, 0.1, 0.2]
gammas = [0.7, 0.8, 0.9, 0.95, 1]
lambdas = [lambda x: sigmoid(x[0], x[1], 3), lambda x: sigmoid(x[0], x[1], 6), lambda x: sigmoid(x[0], x[1], 9), lambda x: sigmoid(x[0], x[1], 12), lambda x: x[2], lambda x: x[2] / 1.05, lambda x: x[2] / 1.1]

config = {
    'alpha': 1,
    'epsilon': 2,
    'gamma': 0,
    'elam': 5,
    'alam': 3
}

class Policy(Enum):
    E_GREEDY = 1

class Learner(object):
    '''
    This agent implements E-Greedy Q-Learning Policy
    '''

    def __init__(self, actions=[0, 1], epsilon=0.02, alpha=0.5, gamma=0.9):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

        # This tracks the epoch number
        self.epoch = 1

        # This tracks how many steps have been taken in the current run
        self.step = 0

        # This tracks the first recorded velocity of the Monkey for gravity calculation
        self.first_vel = 0

        # This tracks whether the gravity for the current run is 1 or 4 (initialized to 1)
        self.gravity = 1

        # These construct the bins which states will be discretized into (MODIFY THIS), for y-coordinates, tree's
        # coordinates, x-coordinates, and velocity
        self.y_bins = [500]
        self.t_bins = [0, 20, 40, 60, 80, 100, 150]
        self.x_bins = [500]
        self.vel_bins = [-15, -5, 0, 5]

        # This will store the q-values that the agent learns
        self.q = {}

        # These are the hyperparameters (MODIFY THE PARAMETERS UP THERE NOT HERE)
        self.actions = actions
        self.epsilon = epsilons[config['epsilon']]
        self.epsilon0 = epsilons[config['epsilon']]
        self.alpha = epsilons[config['alpha']]
        self.alpha0 = epsilons[config['alpha']]
        self.gamma = epsilons[config['gamma']]

    # Reset certain values after each death (before each new iteration)
    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

        self.step = 0
        self.first_vel = 0
        self.gravity = 1

        self.epoch += 1

        # MODIFY THESE

        #self.epsilon = self.epsilon / 1.05
        self.epsilon = lambdas[config['elam']]((self.epsilon0, self.epoch, self.epsilon))
        # self.alpha = self.alpha / 1.05
        # self.alpha = self.alpha0 * (1 - 1 / (1 + math.exp(-(0.2 * self.epoch - 5))))
        self.alpha = lambdas[config['alam']]((self.alpha0, self.epoch, self.alpha))

    # Calculate gravity as the difference between two consecutive velocities, where the Monkey does NOT jump
    def calc_gravity(self, new_y_vel, old_y_vel):
        self.gravity = old_y_vel - new_y_vel

    #----------------------------------------------------------------
    # EVERYTHING BELOW IS Q-LEARNING

    # Return the q-value at the (state, action) pair, or 0 if not found
    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        old_value = self.q.get((state, action), None)

        # If old value is not found, then initialize it to the given reward
        if old_value is None:
            self.q[(state, action)] = reward
        # Otherwise, calculate Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
        else:
            self.q[(state, action)] = old_value + self.alpha * (value - old_value)

    def chooseAction(self, state, policy=Policy.E_GREEDY):
        if policy == Policy.E_GREEDY:
            if npr.random() < self.epsilon:
                action = npr.choice(self.actions)
            else:
                q_vals = [self.getQ(state, action) for action in self.actions]
                max_q = max(q_vals)
                action = q_vals.index(max_q)

                # if action == 1 and self.step < 10:
                #     print(state)
                #     print(q_vals)
            return action

    def learn(self, p_state, p_action, reward, n_state):
        # Calculate the max q-value for q-learning (off policy)
        q_new = max([self.getQ(n_state, action) for action in self.actions])
        self.learnQ(p_state, p_action, reward, reward + self.gamma * q_new)

    # EVERYTHING ABOVE IS Q-LEARNING
    #-------------------------------------------------------------

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        # Discretize the relevant features
        x_dist = np.digitize(state['tree']['dist'], bins=self.x_bins)
        y_tree = np.digitize(state['monkey']['bot'] - state['tree']['bot'], bins=self.t_bins)
        y_monk = np.digitize(state['monkey']['bot'], bins=self.y_bins)
        vel_monk = np.digitize(state['monkey']['vel'], bins=self.vel_bins)
        # print(state['monkey']['vel'])

        # Store these values into a tuple for the new state
        new_state = (int(x_dist), int(y_tree), int(y_monk), int(vel_monk), self.gravity)

        self.step += 1

        # If step = 2 then can calculate gravity
        if self.step == 2:
            self.calc_gravity(state['monkey']['vel'], self.first_vel)

        # If step = 1 then don't jump (new action = 0) so that we can calculate gravity ASAP
        if self.step < 2:
            self.first_vel = state['monkey']['vel']
            new_action = 0
        # Otherwise, choose action based on policy and update q-values
        else:
            new_action = self.chooseAction(new_state)
            self.learn(self.last_state, self.last_action, self.last_reward, new_state)

        self.last_action = new_action
        self.last_state  = new_state

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        if reward > 0:
            reward *= 5
        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''

    for ii in range(iters):
        # print(str(ii) + "------------------------")
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
        
        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


def score(l):
    delta = 0.003
    beta = 1.5
    score = 0
    c = delta * int(np.sum([pow(i, 1 / beta) for i in range(len(l))]))
    sd = np.std(l)
    m = np.median(l)
    for i in range(len(l)):
        w = delta * pow(i, 1 / beta) / c
        f = norm.pdf(l[i], loc=m, scale=sd)
        score += w * (1 - f) * l[i]
    return score

def score2(l):
    beta = 1.5
    score = 0
    c = float(np.sum([pow(i, 1 / beta) for i in range(len(l))]))
    for i in range(len(l)):
        # Since if 0 then don't increment score, if 1 then log(1) = 0
        if l[i] > 1:
            w = pow(i, 1 / beta) / c
            score += w * np.log(l[i])
    return score


def iterate():

    n_trials = 20

    df_cols = ["score", "max", "mean"]
    scores_df = pd.DataFrame(columns=df_cols)

    for i in range(n_trials):
        # Select agent.
        agent = Learner()

        # Empty list to save history.
        hist = []

        # Run games.
        run_games(agent, hist, 100, 1)

        scores = {"score": [score(hist)], "max": [max(hist)], "mean": [int(np.mean(hist))]}
        scores_df = scores_df.append(pd.DataFrame(scores), ignore_index = True)

        # scores_row = pd.DataFrame(columns=df_cols)


        print("ITERATION #" + str(i + 1))
        print("Score:", scores["score"])
        print("Max:", scores["max"])
        print("Mean:", scores["mean"])
        print()

        # Print out all of the q-values
        # print(pd.DataFrame(index=agent.q.keys(), data=agent.q.values()))

        # # Print out the attained scores and their frequencies
        # unique_elements, counts_elements = np.unique(np.array(hist), return_counts=True)
        # scores.append(pd.DataFrame(index=unique_elements, data=counts_elements))

        # means.append(np.mean(hist))
        # sds.append(np.std(hist))
        # maxes.append(np.max(hist))
        # # quantiles.append(np.quantile(hist, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
        # print("Mean: " + str(means[-1]))
        # print("SD: " + str(sds[-1]))
        # print("Max: " + str(maxes[-1]))
        # # print("Quantiles: " + str(quantiles[-1]))

    results = scores_df.apply(np.mean, axis=0)
    print(scores_df.apply(np.mean, axis=0))

    output_str = ""
    for k in config:
        output_str += k + ": "
        if k == 'alpha':
            output_str += str(alphas[config[k]])
        elif k == 'epsilon':
            output_str += str(epsilons[config[k]])
        elif k == 'gamma':
            output_str += str(gammas[config[k]])
        else:
            output_str += str(config[k])
        output_str += ", "
    print(output_str)
    print(config)
    return results['score']



if __name__ == '__main__':
    # scores = []
    # means = []
    # sds = []
    # maxes = []
    # quantiles = []

    n_trials = 20

    df_cols = ["score", "max", "mean"]
    scores_df = pd.DataFrame(columns=df_cols)

    for i in range(n_trials):
        # Select agent.
        agent = Learner()

        # Empty list to save history.
        hist = []

        # Run games.
        run_games(agent, hist, 100, 1)

        scores = {"score": [score(hist)], "logscore": [score2(hist)], "max": [max(hist)], "mean": [int(np.mean(hist))]}
        scores_df = scores_df.append(pd.DataFrame(scores), ignore_index = True)

        # scores_row = pd.DataFrame(columns=df_cols)


        print("ITERATION #" + str(i + 1))
        print("Score:", scores["score"])
        print("Log-score:", scores["logscore"])
        print("Max:", scores["max"])
        print("Mean:", scores["mean"])
        print()

        # Print out all of the q-values
        # print(pd.DataFrame(index=agent.q.keys(), data=agent.q.values()))

        # # Print out the attained scores and their frequencies
        # unique_elements, counts_elements = np.unique(np.array(hist), return_counts=True)
        # scores.append(pd.DataFrame(index=unique_elements, data=counts_elements))

        # means.append(np.mean(hist))
        # sds.append(np.std(hist))
        # maxes.append(np.max(hist))
        # # quantiles.append(np.quantile(hist, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
        # print("Mean: " + str(means[-1]))
        # print("SD: " + str(sds[-1]))
        # print("Max: " + str(maxes[-1]))
        # # print("Quantiles: " + str(quantiles[-1]))

    print(scores_df.apply(np.mean, axis=0))

    output_str = ""
    for k in config:
        output_str += k + ": "
        if k == 'alpha':
            output_str += str(alphas[config[k]])
        elif k == 'epsilon':
            output_str += str(epsilons[config[k]])
        elif k == 'gamma':
            output_str += str(gammas[config[k]])
        else:
            output_str += str(config[k])
        output_str += ", "
    print(output_str)
    print(config)

    # print(means)
    # print(sds)
    # print(maxes)
    # print(quantiles)
    
    # # Save history.
    # np.save('hist',np.array(hist))


    # maxscore = 0
    # best = 0
    # for i in range(len(alphas)):
    #     config['alpha'] = i
    #     newresult = iterate()
    #     if maxscore < newresult:
    #         maxscore = newresult
    #         best = i
    # config['alpha'] = best

    # maxscore = 0
    # best = 0
    # for i in range(len(epsilons)):
    #     config['epsilon'] = i
    #     newresult = iterate()
    #     if maxscore < newresult:
    #         maxscore = newresult
    #         best = i
    # config['epsilon'] = best

    # maxscore = 0
    # best = 0
    # for i in range(len(gammas)):
    #     config['gamma'] = i
    #     newresult = iterate()
    #     if maxscore < newresult:
    #         maxscore = newresult
    #         best = i
    # config['gamma'] = best

    # maxscore = 0
    # best = 0
    # for i in range(len(lambdas)):
    #     config['elam'] = i
    #     newresult = iterate()
    #     if maxscore < newresult:
    #         maxscore = newresult
    #         best = i
    # config['elam'] = best

    # maxscore = 0
    # best = 0
    # for i in range(len(lambdas)):
    #     config['alam'] = i
    #     newresult = iterate()
    #     if maxscore < newresult:
    #         maxscore = newresult
    #         best = i
    # config['alam'] = best

    # output_str = ""
    # for k in config:
    #     output_str += k + ": "
    #     if k == 'alpha':
    #         output_str += str(alphas[config[k]])
    #     elif k == 'epsilon':
    #         output_str += str(epsilons[config[k]])
    #     elif k == 'gamma':
    #         output_str += str(gammas[config[k]])
    #     else:
    #         output_str += str(config[k])
    #     output_str += ", "
    # print(output_str)
    # print(config)



