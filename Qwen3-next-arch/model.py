import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class SimpleGatedDeltaNet(nn.Module):
    """
    简化的 Gated DeltaNet 层，基于 arXiv:2412.06464 的 Gated Delta Rule。
    用于替换传统自注意力机制，提供线性复杂度 O(N) 的长序列处理。
    生产环境中建议使用 flash-linear-attention 库的 GatedDeltaNet 实现以获得更高效率。
    """
    def __init__(self, hidden_size: int, num_heads: int = 16, head_dim: int = 64, chunk_size: int = 64):
        """
        初始化 Gated DeltaNet 层
        参数：
            chunk_size: 分块大小（默认 64，用于 GPU 优化，适配 tensor cores）
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size    # 分块大小，用于并行计算

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        # 门控参数 alpha 和 beta，用于 Gated Delta Rule 的记忆更新
        self.a_proj = nn.Linear(hidden_size, 1, bias=False)  # 计算 alpha（控制记忆保留/遗忘）
        self.b_proj = nn.Linear(hidden_size, 1, bias=False)  # 计算 beta（控制新信息替换强度）

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，计算 Gated DeltaNet 的输出。
        输入：
            x: 输入张量，形状 (batch_size, seq_len, hidden_size)
        输出：
            输出张量，形状同输入，用于后续层处理
        """
        B, T, C = x.shape  # B: batch_size, T: 序列长度, C: hidden_size
        # 将输入映射到 Q、K、V，并调整形状为多头格式 (B, num_heads, T, head_dim)
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # 计算门控参数 alpha 和 beta，范围 (0,1)，通过 sigmoid 激活
        alpha = torch.sigmoid(self.a_proj(x))  # (B, T, 1)，控制记忆遗忘
        beta = torch.sigmoid(self.b_proj(x))   # (B, T, 1)，控制新信息替换

        # 初始化状态矩阵 S，用于存储历史键值对的记忆，形状 (B, num_heads, head_dim, head_dim)
        state = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=x.device)
        out = torch.zeros_like(q)  # 输出张量，初始化为零，形状 (B, num_heads, T, head_dim)

        # 顺序更新状态
        for t in range(T):
            kt = k[:, :, t:t+1, :]  # 当前时间步的键，(B, num_heads, 1, head_dim)
            vt = v[:, :, t:t+1, :]  # 当前时间步的值，(B, num_heads, 1, head_dim)
            alphat = alpha[:, t:t+1, :]  # 当前 alpha，(B, 1, 1)
            betat = beta[:, t:t+1, :]    # 当前 beta，(B, 1, 1)
            # Gated Delta Rule 更新公式：
            # S_t = S_{t-1} * alpha_t * (I - beta_t * k_t * k_t^T) + beta_t * v_t * k_t^T
            I = torch.eye(self.head_dim, device=x.device).unsqueeze(0).unsqueeze(0)  # 单位矩阵 (1,1,D,D)
            delta = betat.unsqueeze(-1) * kt @ kt.transpose(-2, -1)  # 低秩更新，(B, num_heads, D, D)
            state = state * alphat.unsqueeze(-1).unsqueeze(-1) * (I - delta) + betat.unsqueeze(-1) * (vt @ kt.transpose(-2, -1))
            out[:, :, t, :] = (q[:, :, t:t+1, :] @ state.transpose(-2, -1)).squeeze(2)  # 当前时间步输出
        # 调整形状回 (B, T, hidden_size)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return out

class GatedAttention(nn.Module):
    """
    分组查询注意力（GQA）层，带门控机制，模仿 Qwen3-Next 的 Gated Attention。
    使用分组查询（num_heads_q > num_heads_kv）减少 KV 缓存，增加稀疏性。
    """
    def __init__(self, hidden_size: int, num_heads_q: int = 16, num_heads_kv: int = 2, head_dim: int = 256):
        """
        初始化 Gated Attention 层。
        参数：
            hidden_size: 隐藏维度（例如 4096）
            num_heads_q: 查询的头数（默认 16）
            num_heads_kv: 键/值的头数（默认 2，GQA 核心）
            head_dim: 每个头的维度（默认 256，num_heads_q * head_dim = hidden_size）
        """
        super().__init__()
        self.hidden_size = hidden_size      # 模型隐藏维度
        self.num_heads_q = num_heads_q      # 查询头数
        self.num_heads_kv = num_heads_kv    # 键/值头数
        self.head_dim = head_dim            # 每个头的维度
        # 线性层映射到 Q、K、V
        self.q_proj = nn.Linear(hidden_size, num_heads_q * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads_kv * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads_kv * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads_q * head_dim, hidden_size, bias=False)  # 输出投影
        self.gate_proj = nn.Linear(hidden_size, num_heads_q, bias=False)  # 门控权重，控制每个头的激活

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播，计算 Gated Attention 输出。
        输入：
            x: 输入张量，(batch_size, seq_len, hidden_size)
            mask: 可选的注意力掩码，(batch_size, 1, seq_len, seq_len)，用于因果注意力
        输出：
            输出张量，形状同输入
        """
        B, T, C = x.shape
        # 映射到 Q、K、V 并调整为多头格式
        q = self.q_proj(x).view(B, T, self.num_heads_q, self.head_dim).transpose(1, 2)  # (B, num_heads_q, T, head_dim)
        k = self.k_proj(x).view(B, T, self.num_heads_kv, self.head_dim).transpose(1, 2)  # (B, num_heads_kv, T, head_dim)
        v = self.v_proj(x).view(B, T, self.num_heads_kv, self.head_dim).transpose(1, 2)
        # GQA：将 K、V 重复以匹配查询头数
        k = k.repeat_interleave(self.num_heads_q // self.num_heads_kv, dim=1)
        v = v.repeat_interleave(self.num_heads_q // self.num_heads_kv, dim=1)
        # 计算注意力分数
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, num_heads_q, T, T)
        if mask is not None:
            scores = scores.masked_fill(mask[:, None, None, :, :], -1e9)  # 应用因果掩码
        attn = F.softmax(scores, dim=-1)  # 注意力权重
        # 门控机制：对每个头应用 sigmoid 门控
        gates = torch.sigmoid(self.gate_proj(x))[:, None, :, None, None]  # (B, 1, num_heads_q, 1, 1)
        attn = attn * gates  # 稀疏化注意力
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)  # (B, T, hidden_size)
        return self.o_proj(out)  # 最终输出投影

class MoE(nn.Module):
    """
    稀疏混合专家（Mixture of Experts）层，模仿 Qwen3-Next 的 MoE 设计。
    每个 token 选择 top-k（默认 top-1）专家，激活 ~0.4B 参数。
    包含一个共享专家，增强通用性。
    """
    def __init__(self, hidden_size: int, intermediate_size: int = 2048 * 4, num_experts: int = 24, top_k: int = 1):
        """
        初始化 MoE 层
        参数：
            intermediate_size: expert中间层维度（默认 8192，FFN 宽度）
            num_experts: 默认 24，控制稀疏性
            top_k: 每个 token 激活的专家数（默认 1，约 0.4B 参数）
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        # 路由器：决定每个 token 分配到哪些专家
        self.router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),  # 降维减少开销
            nn.ReLU(),
            nn.Linear(hidden_size // 4, num_experts)   # 输出专家得分
        )
        # 专家网络列表，每个专家是一个 FFN（SwiGLU 风格）
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.SiLU(),  # SiLU 激活，类似 SwiGLU
            nn.Linear(intermediate_size, hidden_size)
        ) for _ in range(num_experts)])
        # 共享专家，处理所有 token
        self.shared_expert = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.SiLU(),
            nn.Linear(intermediate_size, hidden_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，计算 MoE 输出。
        输入：
            x: 输入张量，(batch_size, seq_len, hidden_size)
        输出：
            输出张量，形状同输入
        """
        B, T, C = x.shape
        x_flat = x.view(-1, C)  # 展平为 (batch_size * seq_len, hidden_size)
        gates = F.softmax(self.router(x_flat), dim=-1)  # 路由得分，(B*T, num_experts)
        topk_gates, topk_idx = torch.topk(gates, self.top_k, dim=-1)  # 选择 top-k 专家
        topk_gates = F.softmax(topk_gates / self.top_k, dim=-1)  # 归一化 top-k 权重

        out = torch.zeros_like(x_flat)  # 初始化输出
        # 共享专家的贡献
        out += self.shared_expert(x_flat)
        # 稀疏专家的贡献：仅计算 top-k 专家
        for i in range(B * T):
            for j in range(self.top_k):
                expert_idx = topk_idx[i, j]  # 选择的专家索引
                gate = topk_gates[i, j]      # 专家权重
                out[i] += gate * self.experts[expert_idx](x_flat[i])  # 加权专家输出
        return out.view(B, T, C)  # 恢复形状

class Qwen3NextMoE(nn.Module):
    """
    完整的 Qwen3-Next 风格 MoE 模型，模仿 80B-A3B 架构。
    32 层，混合 Gated DeltaNet 和 Gated Attention，MoE 替换 FFN。
    总参数约 8B，激活约 0.4B，适合低资源环境。
    """
    def __init__(self, vocab_size: int = 50257, hidden_size: int = 1024, num_layers: int = 24, num_heads: int = 16):
        """
        初始化模型。
        参数：
            vocab_size: 默认 50257，GPT-2 词表
            num_layers: 默认 24，控制模型深度
            num_heads: 默认 16，Gated Attention 和 DeltaNet 共享
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList()  # 存储所有层
        delta_dim = 128  # Gated DeltaNet 头维度
        attn_dim = 256   # Gated Attention 头维度
        # 构建混合块：3个 (Gated DeltaNet + MoE) + 1个 (Gated Attention + MoE)，重复 6 次
        for block in range(num_layers // 4):
            for _ in range(3):
                self.layers.append(SimpleGatedDeltaNet(hidden_size, num_heads // 2, delta_dim))  # 线性注意力
                self.layers.append(MoE(hidden_size))  # 稀疏 FFN
            self.layers.append(GatedAttention(hidden_size, num_heads, num_heads // 8, attn_dim))
            self.layers.append(MoE(hidden_size))
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)  # 零中心 LayerNorm，模仿 Qwen
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)  # 语言模型头，预测 token

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """
        前向传播，计算模型输出或损失。
        输入：
            idx: 输入 token 索引，(batch_size, seq_len)
            targets: 目标 token 索引，(batch_size, seq_len)，用于训练时的损失计算
        输出：
            如果 targets=None，返回 logits (batch_size, seq_len, vocab_size)
            否则返回交叉熵损失
        """
        B, T = idx.shape
        tok_emb = self.embed(idx)  # 词嵌入，(B, T, hidden_size)
        x = tok_emb
        # 忽略 RoPE（旋转位置编码），用简单位置索引占位，生产中需添加
        for layer in self.layers:
            x = layer(x) + x  # 残差连接
        x = self.norm(x)  # 归一化
        logits = self.lm_head(x)  # 预测 logits，(B, T, vocab_size)
        if targets is None:
            return logits
        # 计算损失，忽略 padding（-1）
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return loss

    def generate(self, idx, max_new_tokens):
        """
        推理生成，基于输入生成新 token。
        输入：
            idx: 初始 token 索引，(batch_size, seq_len)
            max_new_tokens: 最大生成 token 数
        输出：
            生成的 token 索引，(batch_size, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= 1024 else idx[:, -1024:]  # 截断到最大 1024 长度
            logits = self(idx_cond)
            logits = logits[:, -1, :]  # 取最后一个 token 的 logits
            probs = F.softmax(logits, dim=-1)  # 转换为概率
            idx_next = torch.multinomial(probs, num_samples=1)  # 采样下一个 token
            idx = torch.cat((idx, idx_next), dim=1)
        return idx