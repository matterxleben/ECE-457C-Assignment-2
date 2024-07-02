
import numpy as np
import pandas as pd


class rlalgorithm:
    def __init__(self, actions, gamma=0.9, alpha=0.1, epsilon=0.1, *args, **kwargs):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.display_name = "Q-Learning"
        self.Q = {}
        self.actions = actions
        self.num_actions = len(actions)

    def choose_action(self, observation, **kwargs):
        self.check_state_exist(observation)
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            action = self.actions[np.argmax(self.Q[observation])]
        return action

    def learn(self, s, a, r, s_, **kwargs):
        self.check_state_exist(s_)
        q_predict = self.Q[s][a]
        if s_ != 'terminal':
            q_target = r + self.gamma * np.max(self.Q[s_])
        else:
            q_target = r
        self.Q[s][a] += self.alpha * (q_target - q_predict)
        return s_, self.choose_action(s_)

    def check_state_exist(self, state):
        if state not in self.Q:
            self.Q[state] = np.zeros(self.num_actions)
