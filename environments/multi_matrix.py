from gym.spaces import Discrete, Box
import numpy as np
from scipy.special import rel_entr

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class Context:

    def __init__(self, obs, permutation):
        self.obs = obs
        self.permutation = permutation


class MultiMatrixEnv(MultiAgentEnv):

    def __init__(self, env_config):
        # Payoff matrices
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

        self.payoff_matrices = [payoff_matrix1, payoff_matrix2, payoff_matrix3, payoff_matrix4]

        num_matrices = len(self.payoff_matrices)
        num_actions = 3
        num_contexts = env_config.get("num_contexts", num_matrices)
        permute = env_config.get("permute", True)

        # Define observation and action spaces
        obs_space = Box(0.0, 1.0, shape=(num_contexts,))
        self.observation_space_dict = {"row": obs_space, "column": obs_space}

        action_space = Discrete(num_actions)
        self.action_space_dict = {"row": action_space, "column": action_space}

        # Define default observations
        self._dones = {"row": True, "column": True, "__all__": True}
        self._not_dones = {"row": False, "column": False, "__all__": False}
        self._infos = {"row": {}, "column": {}}

        # Define contexts and permutations
        self._contexts = []

        for idx in range(num_contexts):
            obs = np.zeros(num_contexts)
            obs[idx] = 1.0
            obs = {"row": obs, "column": obs}
            permutation = np.random.permutation(num_actions) if permute else np.arange(num_actions)
            self._contexts.append(Context(obs, permutation))

        self._current_context = None
        self._current_matrix = None

    def reset(self, context_idx=None):
        if context_idx is None:
            context_idx = np.random.choice(len(self._contexts))
        self._current_context = self._contexts[context_idx]
        self._current_matrix = self.payoff_matrices[context_idx]
        return self._current_context.obs

    def step(self, action_dict):
        row_action = action_dict["row"]
        column_action = action_dict["column"]

        row_action = self._current_context.permutation[row_action]
        column_action = self._current_context.permutation[column_action]

        row_payoff = self._current_matrix[row_action, column_action, 0]
        column_payoff = self._current_matrix[row_action, column_action, 1]

        dones = self._dones

        rewards = {"row": row_payoff, "column": column_payoff}

        return self._current_context.obs, rewards, dones, self._infos

    def nash_conv(self, policy_dict):
        exploitability = 0.0
        row_values = 0.0
        column_values = 0.0

        for idx, context in enumerate(self._contexts):
            # 对应当前上下文找到对应的 payoff 矩阵
            self._current_matrix = self.payoff_matrices[idx]

            row_actions = np.arange(self._current_matrix.shape[0])
            column_actions = np.arange(self._current_matrix.shape[1])

            # Compute strategies - use the log-likelihood method provided by RLLib policies
            row_obs = [context.obs["row"]] * self._current_matrix.shape[0]
            column_obs = [context.obs["column"]] * self._current_matrix.shape[1]

            row_logits = policy_dict["row"].compute_log_likelihoods(row_actions, row_obs)
            column_logits = policy_dict["column"].compute_log_likelihoods(column_actions, column_obs)

            row_strategy = np.exp(row_logits)
            column_strategy = np.exp(column_logits)

            row_strategy /= np.sum(row_strategy)
            column_strategy /= np.sum(column_strategy)

            # Compute exploitabilities and values
            row_payoffs = self._current_matrix[:, :, 0].dot(column_strategy)
            column_payoffs = row_strategy.dot(self._current_matrix[:, :, 1])

            row_values += 1.0 - np.max(column_payoffs)
            column_values += 1.0 - np.max(row_payoffs)

            exploitability += np.max(row_payoffs) + np.max(column_payoffs) - 1.0

        return {
            "nash_conv": exploitability / len(self._contexts),
            "row_value": row_values / len(self._contexts),
            "column_value": column_values / len(self._contexts)
        }
