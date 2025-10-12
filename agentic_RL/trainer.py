# trainer.py
"""
    rollout 采样、策略优化、并行环境            action, act_id, content_id, log_prob = self.planner.decide(obs)
            traj.planner_actions.append(action)
            traj.contexts.append(obs.context)
            
            # 记录planner步骤
            traj.planner_steps.append(PlannerStep(
                act_id=act_id,
                content_id=content_id,
                old_logp=log_prob,
                context_input_ids=self.planner.tokenizer.encode(obs.context)
            ))
            
            # 记录生成的答案
            if action.action_type == ActionType.ANSWER:
                traj.generated_answers.append(action.content)
import torch
import torch.nn.functional as F
from torch.optim import Adam
from typing import List
from agentic_env import SearchEnv, Trajectory, Observation, PlannerStep
from planner import Planner
from executor import Executor
from verifier import Verifier
from generator import Generator
from agentic_env import ActionType
from transformers import AutoTokenizer

import multiprocessing as mp

class Trainer:
    def __init__(self, search_engine, model_name, lr=3e-5):
        self.search_engine = search_engine
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.planner = Planner(model_name, tokenizer)
        self.executor = Executor(search_engine)
        self.verifier = Verifier()
        self.generator = Generator()

        # 设置优化器，使用torch.cuda.amp进行混合精度训练
        self.optimizer = Adam(self.planner.policy.parameters(), lr=lr)
        self.scaler = torch.cuda.amp.GradScaler()  # 用于混合精度训练
        self.batch_accumulation = 4  # 梯度累积步数
        self.clip_epsilon = 0.2
        self.beta = 0.01
        self.target_kl = 0.02

    def rollout_episode(self, question: str, ground_truth: str, max_steps: int = 10) -> Trajectory:
        env = SearchEnv(self.search_engine)
        obs = env.reset(question)
        env.memory["ground_truth"] = ground_truth

        traj = Trajectory()
        for step in range(max_steps):
            action, act_id, content_id, log_prob = self.planner.decide(obs)
            traj.planner_actions.append(action)
            traj.contexts.append(obs.context)
            # 记录planner步骤
            traj.planner_steps.append(PlannerStep(
                act_id=act_id,
                content_id=content_id,
                old_logp=log_prob,
                context_input_ids=self.planner.tokenizer.encode(obs.context)
            ))
            
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

    def recompute_log_probs(self, trajectories: List[Trajectory]) -> List[torch.Tensor]:
        all_logps = []
        for traj in trajectories:
            logps = []
            for step in traj.planner_steps:
                # 重建 context 输入 ids
                input_ids = torch.tensor(step.context_input_ids, dtype=torch.long).unsqueeze(0).to(self.planner.policy.device)
                # 用策略网络重新计算 log prob
                # step.act_id, step.content_id 是 trajectory 记录的
                # 得到 logp_new
                logp_new = self.planner.policy.get_log_prob(input_ids, step.act_id, step.content_id)
                # 这里 logp_new 是 tensor shape (1,),我们要 scalar
                logps.append(logp_new.squeeze(0))
            if logps:
                all_logps.append(torch.stack(logps))
            else:
                all_logps.append(torch.zeros(0, device=self.planner.policy.device))
        return all_logps

    def update_policy(self, trajectories: List[Trajectory]):
        if not trajectories:
            return
        rewards = [sum(traj.local_rewards) for traj in trajectories]
        advantages = self.compute_advantages(rewards)  # tensor (batch,)

        old_logps_list = []
        for traj in trajectories:
            old_vals = torch.tensor([step.old_logp for step in traj.planner_steps], dtype=torch.float32,
                                    device=self.planner.policy.device)
            old_logps_list.append(old_vals)
        
        for epoch in range(3):
            new_logps_list = self.recompute_log_probs(trajectories)
            total_loss = 0.0
            self.optimizer.zero_grad()
            
            # 将trajectories分成更小的批次以适应GPU内存
            batch_size = max(1, len(trajectories) // self.batch_accumulation)
            for batch_idx in range(0, len(trajectories), batch_size):
                batch_end = min(batch_idx + batch_size, len(trajectories))
                batch_trajectories = trajectories[batch_idx:batch_end]
                
                # 使用混合精度训练
                with torch.cuda.amp.autocast():
                    batch_loss = 0.0
                    for i, traj in enumerate(batch_trajectories):
                        old_lp = old_logps_list[batch_idx + i]
                        new_lp = new_logps_list[batch_idx + i]
                        if len(old_lp) == 0:
                            continue
                        # 逐步计算 ratio
                        ratio = torch.exp(new_lp - old_lp)
                        adv = advantages[batch_idx + i]
                        # 这里我们把 adv 扩展到每一步
                        adv_rep = adv.repeat(len(ratio))
                        # PPO 裁剪 surrogate
                        surr1 = ratio * adv_rep
                        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_rep
                        policy_loss = -torch.min(surr1, surr2).mean()
                        batch_loss = batch_loss + policy_loss
                    
                    batch_loss = batch_loss / len(batch_trajectories)
                    # 缩放损失以适应梯度累积
                    batch_loss = batch_loss / self.batch_accumulation
                    total_loss += batch_loss.item()
                
                # 反向传播
                self.scaler.scale(batch_loss).backward()
            
            # 梯度裁剪并更新
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.planner.policy.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

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
