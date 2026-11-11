# P2-02 Phase 0 — Pusher-v1 数据审计报告 (STOP GATE)

- 生成时间: `2026-11-11T02:54:26.650976+00:00`
- 数据集: `datasets\pusher_v1` · version `pusher-8af70418c4fb`
- manifest SHA-256: `de69a06753bdb4d7954a0e40aa106337236905582cba905033d9f51020248d33`
- 环境: Python 3.13.0 / gymnasium 1.2.1 / mujoco 3.3.7

## 结论: **PASS - proceed to P2-02 Phase 1**

全部硬性完整性检查通过，允许进入 P2-02 Phase 1（模型基础设施）。

## 1. 硬性检查

| 检查 | 结果 | 关键证据 |
|---|---|---|
| Manifest + 逐文件 SHA-256 | PASS | 声明 300 文件 / 磁盘 300 文件，哈希不符 0，未声明 0，缺失 0 |
| 观测/动作维度 | PASS | 300 episodes / 30000 transitions, obs_dim=23, action_dim=7 |
| Action bounds | PASS | 实测 [-2.000000, 2.000000] vs 声明 [-2.0, 2.0] |
| Episode ID 唯一性 | PASS | 300 个 / 300 唯一 / 文件名-meta 不符 0 |
| Split 隔离 | PASS | episode_id 交集 0 / reset seed 交集 0 |
| 时间连续性 obs[t+1]==next_obs[t] | PASS | 逐元素比对 683100，不匹配 0，max_abs_err=0.000000000 |
| Action 时间对齐 (精确重放 / float64 重建) | PASS | 重放 300 episodes / 30000 steps, max_abs_err=0.000000000, atol=0.001, 超阈值 0, 重建 action 偏差=0.000000000 |

### 1.1 方法学说明：为何 gate 采用 float64 精确重放

- 直接重放**磁盘上 float32 存储 action** 时，146/30000 步超过 atol=0.001，最大误差 0.580468（最严重 `train_random_00019` @ t=94）。
- **根因已实证**: 采集器生成的是 float64 action、落盘为 float32（量化差约 6e-8），该差异被 100 步非线性关节动力学放大，使重放轨迹分叉。用确定性 collection policy + `policy_rng_seed` 重建原始 float64 action 后，精确重放 max_abs_err=0.000000000、超阈值 0，且重建 action 与磁盘 action 偏差 0.000000000（即重建的就是原 action）。
- **这不是数据完整性问题**: `obs[t+1]==next_obs[t]` 全量位级一致（见上表）；监督三元组 `(x_t, u_t_float32, x_{t+1})` 来自同一次 `env.step`，自洽无错位。分叉只影响『用 gym 重新模拟』这一辅助验证手段，不影响训练样本。
- 因此 gate 判据采用 float64 精确重放；float32 直接重放的全部分叉数据保留在 `data_audit.json` 的 `action_alignment_replay_diagnostic_float32` 中，未掩盖。

## 2. 观测语义（来自锁定版本源码，非按维度猜测）

来源: `gymnasium 1.2.1 / mujoco 3.3.7` · `gymnasium.envs.mujoco.pusher_v5.PusherEnv._get_obs`

```python
    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flatten()[:7],
                self.data.qvel.flatten()[:7],
                self.get_body_com("tips_arm"),
                self.get_body_com("object"),
                self.get_body_com("goal"),
            ]
        )
```

| 语义分组 | 维度区间 | mean abs(delta) | max abs(delta) |
|---|---|---|---|
| joint_qpos | [0:7] | 0.029994 | 0.427121 |
| joint_qvel | [7:14] | 0.437989 | 9.815135 |
| fingertip_com | [14:17] | 0.007522 | 0.099109 |
| object_com | [17:20] | 0.000030 | 0.004136 |
| goal_com | [20:23] | 0.000000 | 0.000000 |
| robot_kinematics | [0:17] | 0.194026 | 9.815135 |

## 3. 接触统计（真实 MuJoCo 接触，非位置推断）

- **指尖 (`tips_arm`) ↔ 物体接触步数: 0 / 30000** (发生接触的 episode 数: 0/300)
- 其他机械臂连杆 ↔ 物体接触步数: 2819 / 30000 (发生接触的 episode 数: 116/300)
- 数据集内记录的 `contact_state`（物体↔机械臂）比例: mean=0.093967
  - guided: mean=0.156100
  - mixture: mean=0.125800
  - random: mean=0.000000

> **科研限制**: 指尖从未接触物体，Pusher-v1 只能支撑机械臂运动学 / 动作条件预测实验，**不能**证明接触式推物动力学。

## 4. 动作与状态变化分布

- Action 总体: min=-2.000000, max=2.000000, mean=0.029909, std=0.849666
- 逐步动作变化 |a[t+1]-a[t]| 均值: 0.802781 (>0 说明动作序列非常数，具备条件信息)
- 逐维平均 |Δx|（是否真的在变化）:

| dim | mean abs(delta) | dim | mean abs(delta) | dim | mean abs(delta) |
|---|---|---|---|---|---|
| 0 | 0.009043 | 1 | 0.010643 | 2 | 0.034319 |
| 3 | 0.033053 | 4 | 0.043135 | 5 | 0.036505 |
| 6 | 0.043260 | 7 | 0.059726 | 8 | 0.090793 |
| 9 | 0.478451 | 10 | 0.368500 | 11 | 0.681849 |
| 12 | 0.690321 | 13 | 0.696282 | 14 | 0.007533 |
| 15 | 0.007953 | 16 | 0.007078 | 17 | 0.000042 |
| 18 | 0.000047 | 19 | 0.000000 | 20 | 0.000000 |
| 21 | 0.000000 | 22 | 0.000000 |  |  |

## 5. 逐 Split 统计

| split | episodes | transitions | policy 分布 | init obj→goal | final obj→goal | distinct seeds |
|---|---|---|---|---|---|---|
| train | 240 | 24000 | guided:80, mixture:80, random:80 | 0.2397 | 0.2398 | 80 |
| validation | 30 | 3000 | guided:10, mixture:10, random:10 | 0.2223 | 0.2215 | 10 |
| frozen_test | 15 | 1500 | guided:5, mixture:5, random:5 | 0.2383 | 0.2392 | 5 |
| ood_test | 15 | 1500 | guided:5, mixture:5, random:5 | 0.2256 | 0.2233 | 5 |

> **OOD 声明边界**: ood_test differs from train only by reset seed; there is NO explicit distribution-shift mechanism. It must be treated as an additional held-out test set, NOT as evidence of OOD generalisation.

> **冻结测试隔离**: quarantined: Phase 0/3 never train on frozen_test

## 6. 复现命令

```bash
python scripts/audit_pusher_dynamics.py \
    --dataset datasets/pusher_v1 \
    --output outputs/dynamics/pusher_v1
```

原始数据集全程只读，未做任何修改。
