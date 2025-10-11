# trainer.py
"""
    rollout 采样、策略优化、并行环境管理;
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam
from typing import List
from agentic_env import SearchEnv, Trajectory, Observation
from planner import Planner
from executor import Executor
from verifier import Verifier
from generator import Generator
from agentic_env import ActionType

import multiprocessing as mp

class Trainer:
    def __init__(self, search_engine, model_name: str = "microsoft/DialoGPT-small", lr=3e-5):
        self.search_engine = search_engine
        self.planner = Planner(model_name)
        self.executor = Executor(search_engine)
        self.verifier = Verifier()
        self.generator = Generator()

        self.optimizer = Adam(self.planner.model.parameters(), lr=lr)
        self.clip_epsilon = 0.2
        self.beta = 0.01
        self.target_kl = 0.02

    def rollout_episode(self, question: str, ground_truth: str, max_steps: int = 10) -> Trajectory:
        env = SearchEnv(self.search_engine)
        obs = env.reset(question)
        env.memory["ground_truth"] = ground_truth

        traj = Trajectory()
        for step in range(max_steps):
            action = self.planner.decide(obs)
            traj.planner_actions.append(action)
            traj.contexts.append(obs.context)
            obs_next, r, done = env.step(action)
            # record local reward
            traj.local_rewards.append(r)
            # if tool call, also record result
            if action.action_type == ActionType.TOOL:
                # last tool call in memory
                query, result = env.memory["tool_calls"][-1]
                traj.tool_results.append((query, result))
            if action.action_type == ActionType.THINK:
                traj.think_texts.append(action.content)
            if done:
                break
            obs = obs_next
        # final answer
        traj.final_answer = env.context  # 可以后处理
        # total reward = sum local rewards
        traj.reward = sum(traj.local_rewards)
        return traj

    def compute_advantages(self, rewards: List[float]) -> torch.Tensor:
        rt = torch.tensor(rewards, dtype=torch.float32)
        adv = (rt - rt.mean()) / (rt.std() + 1e-8)
        return adv

    def recompute_log_probs(self, traj: Trajectory) -> torch.Tensor:
        """
        用 planner 模型重新计算这条轨迹上每一步 action 的 log prob.
        因为 planner 是一个语言模型 + sampling 决策策略，这里简化假设：
        把 Planner.decide 的 prompt + 输出看作生成 token 的概率。
        对真实项目，需要把 planner 的输出结构化为policy network来计算 log prob。
        """
        # 这里示例代码用最简化方法：不重算，返回 dummy
        return torch.zeros(len(traj.planner_actions))

    def update_policy(self, trajectories: List[Trajectory]):
        if not trajectories:
            return
        rewards = [t.reward for t in trajectories]
        advantages = self.compute_advantages(rewards)

        # 简化：旧 log probs placeholder
        old_log_probs = [self.recompute_log_probs(t) for t in trajectories]

        # 一个 epoch 的更新
        for epoch in range(3):
            total_loss = 0.0
            for i, traj in enumerate(trajectories):
                # new log probs (dummy)
                new_lp = self.recompute_log_probs(traj)
                old_lp = old_log_probs[i]
                # ratio
                ratio = torch.exp(new_lp - old_lp)
                adv = advantages[i]
                # 假设这一条 trajectory 的各步等权
                loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv).mean()
                total_loss += loss
            total_loss = total_loss / len(trajectories)
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.planner.model.parameters(), 1.0)
            self.optimizer.step()

    def train(self, queries: List[str], truths: List[str], epochs: int = 10, batch_size: int = 2):
        for ep in range(epochs):
            batch = list(zip(queries, truths))
            trajectories = []
            for q, gt in batch:
                traj = self.rollout_episode(q, gt, max_steps=8)
                trajectories.append(traj)
            self.update_policy(trajectories)
            print(f"Epoch {ep} reward avg: {sum(t.reward for t in trajectories) / len(trajectories):.3f}")

    def test(self, question: str):
        traj = self.rollout_episode(question, question, max_steps=8)
        print("Context:", traj.contexts)
        print("Actions:", [a.action_type + ":" + a.content for a in traj.planner_actions])
        print("Final context:", traj.final_answer)
        print("Reward:", traj.reward)
