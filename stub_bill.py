# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from enum import Enum

from SwingyMonkey import SwingyMonkey

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

        # These construct the bins which states will be discretized into, for y-coordinates, tree's
        # coordinates, x-coordinates, and velocity
        self.t_bins = [0, 20, 40, 60, 80, 100, 150]
        self.x_bins = [300]
        self.vel_bins = [-15, -5, 0, 5]

        # This will store the q-values that the agent learns
        self.q = {}

        # These are the hyperparameters (MODIFY THE PARAMETERS UP THERE NOT HERE)
        self.actions = actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

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
        self.epsilon = self.epsilon / 1.05
        self.alpha = self.alpha / 1.05

    # Calculate gravity as the difference between two consecutive velocities, where the Monkey does NOT jump
    def calc_gravity(self, new_y_vel, old_y_vel):
        self.gravity = old_y_vel - new_y_vel

    #----------------------------------------------------------------
    # EVERYTHING BELOW IS Q-LEARNING/SARSA

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

    def learn(self, p_state, p_action, reward, n_state, n_action):
        # Calculate the q-value from the policy (SARSA)
        q_new = self.getQ(n_state, n_action)
        self.learnQ(p_state, p_action, reward, reward + self.gamma * q_new)

    # EVERYTHING ABOVE IS Q-LEARNING/SARSA
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
        vel_monk = np.digitize(state['monkey']['vel'], bins=self.vel_bins)
        # print(state['monkey']['vel'])

        # Store these values into a tuple for the new state
        new_state = (y_tree, vel_monk, self.gravity)

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
            self.learn(self.last_state, self.last_action, self.last_reward, new_state, new_action)

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
        if learner.gravity == 1:
            hist[0].append([ii, swing.score])
        else:
            hist[1].append([ii, swing.score])

        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':
    hists_1 = []
    hists_4 = []

    for i in range(15):
        print(i)
        # Select agent.
        agent = Learner()

        # Empty list to save history.
        hist = [[], []]

        # Run games.
        run_games(agent, hist, 60, 1)

        hists_1 = hists_1 + hist[0]
        hists_4 = hists_4 + hist[1]

    print(hists_1)
    print(hists_4)
    scores_1 = pd.DataFrame(hists_1)
    scores_1 = scores_1.groupby(0).mean()
    scores_4 = pd.DataFrame(hists_4)
    scores_4 = scores_4.groupby(0).mean()

    print(scores_1.index.tolist())
    print(scores_1.iloc[:, 0].tolist())
    print(scores_4.index.tolist())
    print(scores_4.iloc[:, 0].tolist())
    spl_1 = UnivariateSpline(scores_1.index.tolist(), scores_1.iloc[:, 0].tolist())
    spl_4 = UnivariateSpline(scores_4.index.tolist(), scores_4.iloc[:, 0].tolist())
    new_x = np.linspace(0, 59, 200)
    plt.plot(new_x, spl_1(new_x))
    plt.plot(new_x, spl_4(new_x))

    plt.legend(['gravity = 1', 'gravity = 4'], loc='upper left')
    plt.title("SARSA Agent (T-Dist, Vel, Gravity)")
    plt.ylabel("Average Score")
    plt.xlabel("Epoch")
    plt.show()


