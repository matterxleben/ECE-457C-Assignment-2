import numpy as np
import pandas as pd
DEBUG=1

def debug(debuglevel, msg, **kwargs):
    if debuglevel <= DEBUG:
        if 'printNow' in kwargs:
            if kwargs['printNow']:
                print(msg)
        else:
            print(msg)

class rlalgorithm:
    def __init__(self, actions, gamma=0.9, alpha=0.1, epsilon=0.1, *args, **kwargs):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.display_name = "SARSA"
        self.Q = {}
        self.actions = actions
        self.num_actions = len(actions)
        debug(1, 'Init new RL Algorithm {}: |A|={} A={} gamma={} alpha={} epsilon={}'.format(
            self.display_name, self.num_actions, self.actions, self.gamma, self.alpha, self.epsilon))

    def choose_action(self, observation, **kwargs):
        self.check_state_exist(observation)
        debug(2, 'pi({})'.format(observation))
        debug(2, 'Q({})={}'.format(observation, self.Q[observation]))
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.actions)
            debug(2, '   a_rand: {}'.format(action))
        else:
            action = self.actions[np.argmax(self.Q[observation])]
            debug(2, '   a_max: {}'.format(action))
        return action

    def learn(self, s, a, r, s_, **kwargs):
        debug(3, '  (learning...)')
        debug(2, 'Learn: s={}\n  a={}\n  r={}\n  s_={}'.format(s, a, r, s_))
        self.check_state_exist(s_)
        
        a_ = self.choose_action(s_) if s_ != 'terminal' else None
        q_predict = self.Q[s][a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.Q[s_][a_]
        else:
            q_target = r
        self.Q[s][a] += self.alpha * (q_target - q_predict)
        debug(2, '  updated Q[{}][{}]={}'.format(s, a, self.Q[s][a]))
        return s_, a_
    
    def check_state_exist(self, state):
        debug(3, '(checking state...)')
        if state not in self.Q:
            self.Q[state] = np.zeros(self.num_actions)
            debug(2, 'Adding state {}'.format(state))    
