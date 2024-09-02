import ray
from algorithms.agents.ppo.ppo import PPOTrainer
from ray.tune.registry import ENV_CREATOR, _global_registry

import yaml
import algorithms
import environments

ray.init(num_cpus=4, num_gpus=1)


def main():
    EXPERIMENTS = dict()
    with open("experiment_configs/custom/multi_matrix_ppo_contextual_rnd_self.yaml") as f:
        EXPERIMENTS.update(yaml.load(f, Loader=yaml.FullLoader))

    experiment = next(iter(EXPERIMENTS.values()))
    exp_config = experiment["config"]

    # Create temporary env instance to query observation space, action space and number of agents
    env_name = exp_config["env"]
    env_creator = _global_registry.get(ENV_CREATOR, env_name)
    test_env = env_creator(exp_config.get("env_config", {}))

    # One policy per agent for multiple individual learners
    policies = dict()
    for pid in test_env.observation_space_dict.keys():
        policies[f"policy_{pid}"] = (
            None,
            test_env.observation_space_dict[pid],
            test_env.action_space_dict[pid],
            {}
        )

    exp_config["multiagent"] = {"policies": policies,
                                "policy_mapping_fn": lambda pid: f"policy_{pid}"}

    ppo_trainer = PPOTrainer(config=exp_config)
    checkpoint_path = "2024-08-28/multi_2matrix_contextual/PPO_CURIOSITY_multi_matrix_0_2024-08-29_17-14-46vjhn6wws/checkpoint_100/checkpoint-100"
    ppo_trainer.restore(checkpoint_path)

    # 获取所有代理的策略映射
    policy_mapping_fn = exp_config["multiagent"]["policy_mapping_fn"]

    # 测试每个支付矩阵
    num_matrices = len(test_env.payoff_matrices)
    for context_idx in range(num_matrices):
        print(f"\nTesting with payoff matrix {context_idx + 1}/{num_matrices}")
        observation = test_env.reset(context_idx=context_idx)
        done = False
        while not done:
            agent_ids = ["row", "column"]
            # 计算每个代理的动作
            actions = {}
            # 创建一个空的policy_dict
            policy_dict = {}
            for agent_id in agent_ids:
                policy_id = policy_mapping_fn(agent_id)
                actions[agent_id] = ppo_trainer.compute_action(observation[agent_id], policy_id=policy_id)

                policy = ppo_trainer.get_policy(policy_id)  # 从ppo_trainer获取策略
                policy_dict[agent_id] = policy  # 将策略添加到字典中

            conv = test_env.compute_nash_conv(policy_dict, context_idx)  # 传入policy_dict
            print(f"Nash Conv: {conv}")

            # 执行动作并获取新状态
            observation, rewards, done, info = test_env.step(actions)

            # 输出或记录动作、奖励等
            for agent_id, action in actions.items():
                print(f"Agent: {agent_id}, Action: {action}, Reward: {rewards[agent_id]}")


if __name__ == "__main__":
    main()
