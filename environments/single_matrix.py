import numpy as np
from gym.spaces import Discrete, Box
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class SingleMatrixEnv(MultiAgentEnv):

    def __init__(self, env_config):
        # Define a single payoff matrix
        payoff_matrix1 = np.array([[[2, 1], [4, 3], [1, 2]],
                                   [[3, 2], [1, 4], [4, 0]],
                                   [[0, 4], [2, 1], [3, 3]]])

        payoff_matrix2 = np.array([[[3, 3], [2, 1], [4, 2]],
                                   [[1, 4], [3, 3], [2, 1]],
                                   [[4, 2], [1, 4], [3, 3]]])

        payoff_matrix3 = np.array([[[2, 4], [1, 1], [5, 1]],
                                   [[0, 0], [4, 0], [1, 5]],
                                   [[4, 1], [3, 3], [2, 2]]])

        payoff_matrix4 = np.array([[[0.5, 0.5], [0.0, 1.0], [1.0, 0.0]],
                                   [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
                                   [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]])  # Rock-Paper-Scissors

        self.payoff_matrix = payoff_matrix3

        num_actions = 3

        # Define observation and action spaces
        obs_space = Box(0.0, 1.0, shape=(1,))
        self.observation_space_dict = {"row": obs_space, "column": obs_space}

        action_space = Discrete(num_actions)
        self.action_space_dict = {"row": action_space, "column": action_space}

        # Define default observations
        self._dones = {"row": True, "column": True, "__all__": True}
        self._not_dones = {"row": False, "column": False, "__all__": False}
        self._infos = {"row": {}, "column": {}}

        self._current_matrix = self.payoff_matrix

    def reset(self):
        return {"row": np.array([1.0]), "column": np.array([1.0])}

    def step(self, action_dict):
        row_action = action_dict["row"]
        column_action = action_dict["column"]

        row_payoff = self._current_matrix[row_action, column_action, 0]
        column_payoff = self._current_matrix[row_action, column_action, 1]

        dones = self._dones
        rewards = {"row": row_payoff, "column": column_payoff}

        return {"row": np.array([1.0]), "column": np.array([1.0])}, rewards, dones, self._infos

    def nash_conv(self, policy_dict):
        row_actions = np.arange(self._current_matrix.shape[0])
        column_actions = np.arange(self._current_matrix.shape[1])

        row_logits = policy_dict["row"].compute_log_likelihoods(row_actions, [np.array([1.0])] * len(row_actions))
        column_logits = policy_dict["column"].compute_log_likelihoods(column_actions,
                                                                      [np.array([1.0])] * len(column_actions))

        row_strategy = np.exp(row_logits)
        column_strategy = np.exp(column_logits)

        row_strategy /= np.sum(row_strategy)
        column_strategy /= np.sum(column_strategy)

        # Calculate Nash Conv
        row_payoffs = self._current_matrix[:, :, 0].dot(column_strategy)
        column_payoffs = row_strategy.dot(self._current_matrix[:, :, 1])

        row_expl = np.max(row_payoffs) - np.dot(row_strategy, row_payoffs)
        column_expl = np.dot(column_strategy, column_payoffs) - np.min(column_payoffs)

        exploitability = row_expl + column_expl

        return {
            "nash_conv": exploitability
        }
