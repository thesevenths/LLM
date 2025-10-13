# planner.py
"""
    根据 Observation 决定下一个 Action
"""

# planner.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from agentic_env import Observation, Action, ActionType
from transformers import AutoModelForCausalLM, AutoConfig

class PlannerPolicy(nn.Module):
    def __init__(self, hidden_size: int, tool_vocab_size: int, think_vocab_size: int, model_name):
        super().__init__()
        
        # 检测是否有可用的 GPU 并确定可用显存
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2  # 转换为MB
            has_gpu = True
        except:
            gpu_memory = 0
            has_gpu = False
        
        # 根据可用资源决定加载策略
        config = AutoConfig.from_pretrained(model_name)
        # 加载tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        if has_gpu and gpu_memory > 4000:  # 如果有超过4GB显存
            device = "cuda:0"
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                max_memory={
                    0: "3GiB",
                    "cpu": "12GiB"
                },
                trust_remote_code=True
            )
        else:
            print("使用CPU模式运行...")
            device = "cpu"
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map={"": device},  # 强制使用CPU
                trust_remote_code=True
            )
        
        # Qwen 模型的 hidden size
        qwen_hidden_size = config.hidden_size
        
        # 决策头：THINK / TOOL / ANSWER (需要一个projection 将 Qwen hidden size 映射到需要的维度)
        self.hidden_proj = nn.Linear(qwen_hidden_size, hidden_size).to(device)
        self.action_head = nn.Linear(hidden_size, 3).to(device)
        # 子策略 heads
        self.tool_head = nn.Linear(hidden_size, tool_vocab_size).to(device)
        self.think_head = nn.Linear(hidden_size, think_vocab_size).to(device)
        
        # 保存设备信息
        self.device = device
        # （如果 ANSWER 内容也要生成，可以直接使用 Qwen 的生成能力）
    
    def forward(self, input_ids: torch.Tensor):
        """
        input_ids: (batch, seq_len) 的 token id 序列，表示 context
        返回 dict，包含 logits / probs
        """
        # 确保输入在正确的设备上
        input_ids = input_ids.to(self.device)
        
        # 使用 Qwen 模型得到隐藏状态和logits
        outputs = self.base_model(input_ids, output_hidden_states=True)
        
        # DEBUG: 打印完整的输出内容
        print("\n=== Qwen Model Outputs ===")
        print(f"当前设备: {self.device}")
        
        # 1. 打印输入文本
        print("\n1. 输入文本:")
        input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print(input_text)
        
        # 2. 打印模型的预测/生成
        if hasattr(outputs, 'logits'):
            print("\n2. 模型预测的下一个token (Top 5):")
            next_token_logits = outputs.logits[0, -1, :]  # 取最后一个位置的logits
            top_k = 5
            topk_values, topk_indices = torch.topk(next_token_logits, top_k)
            
            # 将概率归一化
            probs = torch.softmax(topk_values, dim=0)
            
            for i, (prob, idx) in enumerate(zip(probs, topk_indices)):
                token = self.tokenizer.decode([idx])
                print(f"   {i+1}. '{token}' (概率: {prob:.3f})")
        
        # 3. 生成更长的回答
        print("\n3. 生成的完整回答:")
        try:
            with torch.no_grad():
                generated_ids = self.base_model.generate(
                    input_ids,
                    max_new_tokens=150,  # 生成最多150个新token
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    temperature=0.7,
                    top_p=0.9,
                )
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            # 只显示新生成的部分
            original_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            new_text = generated_text[len(original_text):]
            print(f"生成的新内容: {new_text}")
        except Exception as e:
            print(f"生成过程出现错误: {e}")
            
        print("========================\n")
        
        # 取最后一层最后一个 token 的隐藏状态作为 context embedding
        last_hidden_state = outputs.hidden_states[-1]  # (batch, seq_len, qwen_hidden_size)
        ctx = last_hidden_state[:, -1, :]  # (batch, qwen_hidden_size)
        # 投影到我们需要的 hidden size
        ctx = self.hidden_proj(ctx)  # (batch, hidden_size)
        
        action_logits = self.action_head(ctx)  # (b, 3)
        tool_logits = self.tool_head(ctx)      # (b, tool_vocab_size)
        think_logits = self.think_head(ctx)    # (b, think_vocab_size)
        
        return {
            "action_logits": action_logits,
            "tool_logits": tool_logits,
            "think_logits": think_logits
        }
    
    def get_log_prob(self, input_ids: torch.Tensor, action_type: int, content_id: int):
        """
        给定 context 的 input_ids、action_type 与 content_id，计算 log π(action_type, content_id | context)
        
        Args:
            input_ids: 输入token IDs
            action_type: 动作类型的ID (0:THINK, 1:TOOL, 2:ANSWER)
            content_id: 内容的ID
        """
        out = self.forward(input_ids)
        action_logits = out["action_logits"]
        logp_action = F.log_softmax(action_logits, dim=-1)
        
        # action_type直接作为索引使用
        if not (0 <= action_type <= 2):
            raise ValueError(f"Invalid action_type: {action_type}, must be 0, 1, or 2")
            
        logp = logp_action[:, action_type]  # (batch,)
        # 加上 content 子策略的 log prob
        if action_type == 0:  # THINK
            think_logits = out["think_logits"]
            logp_content = F.log_softmax(think_logits, dim=-1)
            logp = logp + logp_content[:, content_id]
        elif action_type == 1:  # TOOL
            tool_logits = out["tool_logits"]
            logp_content = F.log_softmax(tool_logits, dim=-1)
            logp = logp + logp_content[:, content_id]
        elif action_type == 2:  # ANSWER
            # ANSWER的情况，我们不需要额外的content logprob
            pass
        return logp  # 返回 tensor shape (batch,)


class Planner:
    def __init__(self, model_name: str, tokenizer, hidden_size: int = 768, tool_vocab_size: int = 100, think_vocab_size: int = 100):
        self.tokenizer = tokenizer
        # 创建 PlannerPolicy 实例
        self.policy = PlannerPolicy(
            hidden_size=hidden_size,
            tool_vocab_size=tool_vocab_size,
            think_vocab_size=think_vocab_size,
            model_name=model_name
        )
        # 使用策略网络的设备
        self.device = self.policy.device
        
        # 创建思考和工具的词汇表映射
        self.think_vocab = {
            0: "让我思考一下这个问题",
            1: "我需要更多信息",
            2: "这个问题可以分解为几个部分",
            # ... 可以添加更多思考模板
        }
        
        self.tool_vocab = {
            0: "搜索相关概念",
            1: "查找具体例子",
            2: "寻找最新信息",
            # ... 可以添加更多工具查询模板
        }
    
    def id_to_think_text(self, think_id: int) -> str:
        """将思考ID转换为具体的思考文本"""
        return self.think_vocab.get(think_id, "让我思考一下")
    
    def id_to_tool_query(self, tool_id: int) -> str:
        """将工具ID转换为具体的查询文本"""
        return self.tool_vocab.get(tool_id, "搜索相关信息")

    def decide(self, obs: Observation):
        try:
            # 把 obs.context 转为 token ids
            input_ids = self.tokenizer.encode(obs.context, return_tensors="pt").to(self.device)
            
            # 打印调试信息
            print(f"\n处理输入: {obs.context}")
            
            # 前向得到 logits
            out = self.policy(input_ids)
            action_logits = out["action_logits"]
            action_probs = F.softmax(action_logits, dim=-1)
            
            # 打印动作概率
            print(f"动作概率: THINK={action_probs[0][0]:.3f}, TOOL={action_probs[0][1]:.3f}, ANSWER={action_probs[0][2]:.3f}")
            
            # 先采样 action_type
            dist = torch.distributions.Categorical(action_probs)
            act_id = dist.sample().item()
            logp_act = dist.log_prob(torch.tensor(act_id).to(self.device)).item()
            
            print(f"选择的动作ID: {act_id}")
        except Exception as e:
            print(f"决策过程出错: {e}")
            # 返回一个默认的THINK动作
            return Action(ActionType.THINK, "出错了，让我重新思考"), 0, 0, 0.0

        # 初始化content_id为None
        content_id = None
        
        if act_id == 0:  # THINK
            think_logits = out["think_logits"].squeeze(0)  # shape (think_vocab_size,)
            dist2 = torch.distributions.Categorical(F.softmax(think_logits, dim=-1))
            content_id = dist2.sample().item()
            logp_content = dist2.log_prob(torch.tensor(content_id).to(self.device)).item()
            content = self.id_to_think_text(content_id)
            total_logp = logp_act + logp_content
            action = Action(ActionType.THINK, content)
            
        elif act_id == 1:  # TOOL
            tool_logits = out["tool_logits"].squeeze(0)
            dist2 = torch.distributions.Categorical(F.softmax(tool_logits, dim=-1))
            content_id = dist2.sample().item()
            logp_content = dist2.log_prob(torch.tensor(content_id).to(self.device)).item()
            content = self.id_to_tool_query(content_id)
            total_logp = logp_act + logp_content
            action = Action(ActionType.TOOL, content)
            
        else:  # ANSWER
            # 使用policy中的base_model生成答案
            try:
                with torch.no_grad():
                    generated_ids = self.policy.base_model.generate(
                        input_ids,
                        max_new_tokens=50,
                        num_return_sequences=1,
                        temperature=0.7,
                        top_p=0.9,
                    )
                    content = self.tokenizer.decode(generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
                    print(f"\n成功生成答案: {content}")  # 调试输出
            except Exception as e:
                print(f"生成答案时出错: {e}")
                content = "<生成答案失败>"
            
            content_id = 0  # ANSWER类型的content_id固定为0
            total_logp = logp_act
            action = Action(ActionType.ANSWER, content)
            # 打印生成的答案
            print(f"\n生成的答案: {content}")

        # 你还要在 trajectory 中记录 act_id, content_id, 和 logp 总和作为 old_log_prob
        return action, act_id, content_id, total_logp