import numpy as np
import pandas as pd


class rlalgorithm:
    def __init__(self, actions, gamma=0.9, alpha=0.1, epsilon=0.1, *args, **kwargs):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.display_name = "Double Q-Learning"
        self.Q1 = {}
        self.Q2 = {}
        self.actions = actions
        self.num_actions = len(actions)

    def choose_action(self, observation, **kwargs):
        self.check_state_exist(observation)
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            q_sum = self.Q1[observation] + self.Q2[observation]
            action = self.actions[np.argmax(q_sum)]
        return action

    def learn(self, s, a, r, s_, **kwargs):
        self.check_state_exist(s_)
        if np.random.uniform() < 0.5:
            q_predict = self.Q1[s][a]
            if s_ != 'terminal':
                q_target = r + self.gamma * self.Q2[s_][np.argmax(self.Q1[s_])]
            else:
                q_target = r
            self.Q1[s][a] += self.alpha * (q_target - q_predict)
        else:
            q_predict = self.Q2[s][a]
            if s_ != 'terminal':
                q_target = r + self.gamma * self.Q1[s_][np.argmax(self.Q2[s_])]
            else:
                q_target = r
            self.Q2[s][a] += self.alpha * (q_target - q_predict)
        return s_, self.choose_action(s_)

    def check_state_exist(self, state):
        if state not in self.Q1:
            self.Q1[state] = np.zeros(self.num_actions)
        if state not in self.Q2:
            self.Q2[state] = np.zeros(self.num_actions)
