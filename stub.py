# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg
import pandas as pd
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

        # These construct the bins which states will be discretized into (MODIFY THIS), for y-coordinates, tree's
        # coordinates, x-coordinates, and velocity
        #self.y_bins = [50, 250]
        self.y_bins = [500] # delete
        self.t_bins = [0, 20, 40, 60, 80, 100, 150]
        self.x_bins = [500] # delete
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
        #print(state['monkey']['vel'])
        #print(y_tree)
        #print(state['tree']['dist'])
        #print(x_dist)
        #print(state['monkey']['bot'] - state['tree']['bot'])
        #print(vel_monk)

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
            reward *= 10
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


if __name__ == '__main__':
    scores = []
    means = []
    sds = []
    maxes = []
    quantiles = []

    for i in range(10):
        # Select agent.
        agent = Learner()

        # Empty list to save history.
        hist = []

        # Run games.
        run_games(agent, hist, 100, 0)

        # Print out all of the q-values
        # print(pd.DataFrame(index=agent.q.keys(), data=agent.q.values()))

        # Print out the attained scores and their frequencies
        unique_elements, counts_elements = np.unique(np.array(hist), return_counts=True)
        scores.append(pd.DataFrame(index=unique_elements, data=counts_elements))

        means.append(np.mean(hist))
        sds.append(np.std(hist))
        maxes.append(np.max(hist))
        # quantiles.append(np.quantile(hist, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
        print("Mean: " + str(means[-1]))
        print("SD: " + str(sds[-1]))
        print("Max: " + str(maxes[-1]))
        # print("Quantiles: " + str(quantiles[-1]))

    print(means)
    print(sds)
    print(maxes)
    print(quantiles)
    print(np.mean(means))
    
    # # Save history.
    # np.save('hist',np.array(hist))


