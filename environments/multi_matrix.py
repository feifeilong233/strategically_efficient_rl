from gym.spaces import Discrete, Box
import numpy as np
from scipy.special import rel_entr
import nashpy as nash

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class Context:

    def __init__(self, obs, permutation):
        self.obs = obs
        self.permutation = permutation


class MultiMatrixEnv(MultiAgentEnv):

    def __init__(self, env_config):
        payoff_matrix1 = np.array([[[1, -1], [-2, 2], [0, 0]],
                                   [[0, 0], [1, -1], [-1, 1]],
                                   [[-1, 1], [2, -2], [1, -1]]])

        payoff_matrix2 = np.array([[[2, -2], [-1, 1], [-3, 3]],
                                   [[-2, 2], [3, -3], [1, -1]],
                                   [[-1, 1], [-3, 3], [2, -2]]])

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
        permute = env_config.get("permute", False)

        # Define observation and action spaces
        obs_space = Box(0.0, 1.0, shape=(num_contexts,))
        self.observation_space_dict = {"0": obs_space, "1": obs_space}

        action_space = Discrete(num_actions)
        self.action_space_dict = {"0": action_space, "1": action_space}

        # Define default observations
        self._dones = {"0": True, "1": True, "__all__": True}
        self._not_dones = {"0": False, "1": False, "__all__": False}
        self._infos = {"0": {}, "1": {}}

        # Define contexts and permutations
        self._contexts = []

        for idx in range(num_contexts):
            obs = np.zeros(num_contexts)
            obs[idx] = 1.0
            obs = {"0": obs, "1": obs}
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
        row_action = action_dict["0"]
        column_action = action_dict["1"]

        row_action = self._current_context.permutation[row_action]
        column_action = self._current_context.permutation[column_action]

        row_payoff = self._current_matrix[row_action, column_action, 0]
        column_payoff = self._current_matrix[row_action, column_action, 1]

        dones = self._dones

        rewards = {"0": row_payoff, "1": column_payoff}

        return self._current_context.obs, rewards, dones, self._infos

    def compute_nash_equilibrium(self):
        nash_equilibrium = None
        # 构建支付矩阵
        row_payoff_matrix = self._current_matrix[:, :, 0]
        column_payoff_matrix = self._current_matrix[:, :, 1]

        # 创建博弈对象
        game = nash.Game(row_payoff_matrix, column_payoff_matrix)

        # 计算混合策略纳什均衡
        equilibria = game.support_enumeration()

        # 选取第一个均衡（如果存在）
        for eq in equilibria:
            nash_equilibrium = eq
            break

        return nash_equilibrium

    def nash_conv(self, policy_dict):
        exploitability = 0.0
        row_values = 0.0
        column_values = 0.0
        avg_kl_divergence = 0.0

        for idx, context in enumerate(self._contexts):
            # 对应当前上下文找到对应的 payoff 矩阵
            self._current_matrix = self.payoff_matrices[idx]

            # 理论上的混合策略分布
            theoretical_distribution = self.compute_nash_equilibrium()

            row_actions = np.arange(self._current_matrix.shape[0])
            column_actions = np.arange(self._current_matrix.shape[1])

            # Compute strategies - use the log-likelihood method provided by RLLib policies
            row_obs = [context.obs["0"]] * self._current_matrix.shape[0]
            column_obs = [context.obs["1"]] * self._current_matrix.shape[1]

            row_logits = policy_dict["0"].compute_log_likelihoods(row_actions, row_obs)
            column_logits = policy_dict["1"].compute_log_likelihoods(column_actions, column_obs)

            row_strategy = np.exp(row_logits)
            column_strategy = np.exp(column_logits)

            row_strategy /= np.sum(row_strategy)
            column_strategy /= np.sum(column_strategy)

            # Compute KL divergence for row strategy
            epsilon = 1e-10
            kl_divergences = []

            for i in range(len(theoretical_distribution)):
                smoothed_dist = (np.array(theoretical_distribution[i]) + epsilon) / (
                            1 + epsilon * len(theoretical_distribution[i]))
                kl_div = np.sum(rel_entr(row_strategy, smoothed_dist))
                kl_divergences.append(kl_div)

            # Compute exploitabilities and values
            row_payoffs = self._current_matrix[:, :, 0].dot(column_strategy)
            column_payoffs = row_strategy.dot(self._current_matrix[:, :, 1])

            row_expl = np.max(row_payoffs) - np.dot(row_strategy, row_payoffs)
            column_expl = np.dot(column_strategy, column_payoffs) - np.min(column_payoffs)

            row_values += np.dot(row_strategy, row_payoffs)
            column_values += np.dot(column_strategy, column_payoffs)

            exploitability += row_expl + column_expl

        avg_kl_divergence += np.mean(kl_divergences)

        return {
            "nash_conv": exploitability / len(self._contexts),
            "row_value": row_values / len(self._contexts),
            "column_value": column_values / len(self._contexts),
            "kl_divergence": avg_kl_divergence / len(self._contexts),
        }

    def compute_nash_conv(self, policy_dict, idx):
        self._current_matrix = self.payoff_matrices[idx]

        # 理论上的混合策略分布
        theoretical_distribution = self.compute_nash_equilibrium()

        row_actions = np.arange(self._current_matrix.shape[0])
        column_actions = np.arange(self._current_matrix.shape[1])

        # Compute strategies - use the log-likelihood method provided by RLLib policies
        row_obs = [self._contexts[idx].obs["0"]] * self._current_matrix.shape[0]
        column_obs = [self._contexts[idx].obs["1"]] * self._current_matrix.shape[1]

        row_logits = policy_dict["0"].compute_log_likelihoods(row_actions, row_obs)
        column_logits = policy_dict["1"].compute_log_likelihoods(column_actions, column_obs)

        row_strategy = np.exp(row_logits)
        column_strategy = np.exp(column_logits)

        row_strategy /= np.sum(row_strategy)
        column_strategy /= np.sum(column_strategy)

        print(row_strategy, column_strategy)

        # Compute KL divergence for row strategy
        epsilon = 1e-10
        kl_divergences = []

        for i in range(len(theoretical_distribution)):
            smoothed_dist = (np.array(theoretical_distribution[i]) + epsilon) / (
                    1 + epsilon * len(theoretical_distribution[i]))
            kl_div = np.sum(rel_entr(row_strategy, smoothed_dist))
            kl_divergences.append(kl_div)

        # Compute exploitabilities and values
        row_payoffs = self._current_matrix[:, :, 0].dot(column_strategy)
        column_payoffs = row_strategy.dot(self._current_matrix[:, :, 1])

        row_expl = np.max(row_payoffs) - np.dot(row_strategy, row_payoffs)
        column_expl = np.dot(column_strategy, column_payoffs) - np.min(column_payoffs)

        row_values = np.dot(row_strategy, row_payoffs)
        column_values = np.dot(column_strategy, column_payoffs)

        exploitability = row_expl + column_expl

        avg_kl_divergence = np.mean(kl_divergences)

        return {
            "nash_conv": exploitability,
            "row_value": row_values,
            "column_value": column_values,
            "kl_divergence": avg_kl_divergence,
        }
