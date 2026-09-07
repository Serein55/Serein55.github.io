# π-RL：Flow Matching 与强化学习

## 1. Motivation

将 RL 用于基于 flow matching 的大规模 VLA（如 π₀、π₀.₅）时，面临一个根本性障碍：flow matching 生成的动作 log-likelihood 不可解析求解。 而现有 VLA-RL 工作（SimpleVLA-RL、RL4VLA 等）均建立在自回归 VLA 上（离散 softmax 或 Gaussian 参数化头），对 flow-based 模型完全不适用。

核心难点有二：
1.  flow ODE 是确定性的，缺乏随机性，无法做 RL 探索
2. 直接用 instantaneous change-of-variables（Hutchinson estimator）计算精确 log-likelihood 步数少时误差极大


## 2. 两个核心技术

### 2.1 Flow-Noise

**思路**：在每个 denoising step 注入一个**可学习噪声网络** $\sigma_{\theta'}$，将原来确定性的 Euler 步变成带随机性的高斯转移：

$$
p(\mathbf{A}^{\tau+\delta}|\mathbf{A}^{\tau}) \sim \mathcal{N}(\mu_\tau, \Sigma_\tau), \quad \mu_\tau = \mathbf{A}^\tau + \mathbf{v}^\tau \cdot \delta, \quad \Sigma_\tau = \text{diag}(\sigma_{\theta'}^2)
$$

由此将 denoising 过程建模为**离散时间 MDP**，整条 denoising 序列 $\mathcal{A} = (\mathbf{A}^0, \dots, \mathbf{A}^1)$ 的 log-likelihood 可精确分解为：

$$
\log \pi(\mathcal{A}|\mathbf{o}) = \log \pi(\mathbf{A}^0|\mathbf{o}) + \sum_{k=0}^{K-1} \log \pi(\mathbf{A}^{\tau_{k+1}} | \mathbf{A}^{\tau_k}, \mathbf{o})
$$

**噪声网络仅在训练时使用，推理时丢弃**，保持原有确定性推理效率不变（inspired by ReinFlow，但将固定噪声改为可学习噪声，灵活性更高）。

原始 flow ODE 的 Euler 步本来就是完全确定性的，均值就是 $\mu_\tau$，方差为零。Flow-Noise 做的是在这个确定性步上叠加一个可学习的方差。

---

#### 原始 ODE vs Flow-Noise

标准 flow matching 的推理是 Euler 积分：$\mathbf{A}^{\tau+\delta} = \mathbf{A}^\tau + \mathbf{v}_\theta(\mathbf{A}^\tau, \mathbf{o}) \cdot \delta$

给定 $\mathbf{A}^\tau$，输出 $\mathbf{A}^{\tau+\delta}$ 是唯一确定的。转移概率退化为 Dirac delta：
$$
p(\mathbf{A}^{\tau+\delta} | \mathbf{A}^\tau) = \delta\!\left(\mathbf{A}^{\tau+\delta} - \underbrace{(\mathbf{A}^\tau + \mathbf{v}^\tau \cdot \delta)}_{\mu_\tau}\right)
$$

即 $\Sigma_\tau = 0$，log-likelihood 不存在（Dirac delta 无法取 log），policy gradient 无从计算。
#### Flow-Noise 推导

##### Step 1：注入可学习噪声

引入噪声网络 $\sigma_{\theta'}(\mathbf{A}^\tau, \mathbf{o})$，将 Euler 步改写为随机转移：

$$
\boxed{p(\mathbf{A}^{\tau+\delta} | \mathbf{A}^\tau, \mathbf{o}) = \mathcal{N}(\mathbf{A}^{\tau+\delta};\, \mu_\tau,\, \Sigma_\tau)}
$$

其中：$\mu_\tau = \mathbf{A}^\tau + \mathbf{v}^\tau \cdot \delta, \qquad \Sigma_\tau = \text{diag}\!\left(\sigma_{\theta'}^2(\mathbf{A}^\tau, \mathbf{o})\right)$
采样实现：
$$
\mathbf{A}^{\tau+\delta} = \mu_\tau + \sigma_{\theta'} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

注意均值 $\mu_\tau$ 与原始 ODE 的 Euler 步一致，只是叠加了随机扰动。
##### Step 2：单步 log-likelihood

由 Gaussian 概率密度：

$$
\log p(\mathbf{A}^{\tau+\delta} | \mathbf{A}^\tau, \mathbf{o}) = -\frac{d}{2}\log(2\pi) - \sum_i \log \sigma_{\theta',i} - \frac{1}{2} \sum_i \frac{(\mathbf{A}^{\tau+\delta}_i - \mu_{\tau,i})^2}{\sigma_{\theta',i}^2}
$$

其中 $d$ 是动作维度，$i$ 遍历每个维度（因为 $\Sigma_\tau$ 是对角阵）。

##### Step 3：将 denoising 建模为离散时间 MDP

离散化时间轴为 $K$ 步：$\tau_0=0, \tau_1=\delta, \ldots, \tau_K=1$，$\delta = 1/K$。

定义 denoising 序列 $\mathcal{A} = (\mathbf{A}^{\tau_0}, \mathbf{A}^{\tau_1}, \ldots, \mathbf{A}^{\tau_K})$，由 Markov 性：

$$
p(\mathcal{A} | \mathbf{o}) = p(\mathbf{A}^{\tau_0} | \mathbf{o}) \prod_{k=0}^{K-1} p(\mathbf{A}^{\tau_{k+1}} | \mathbf{A}^{\tau_k}, \mathbf{o})
$$

取 log，加法分解：

$$
\boxed{\log \pi(\mathcal{A} | \mathbf{o}) = \underbrace{\log p(\mathbf{A}^0 | \mathbf{o})}_{\log \mathcal{N}(\mathbf{0}, \mathbf{I})} + \sum_{k=0}^{K-1} \underbrace{\log \mathcal{N}\!\left(\mathbf{A}^{\tau_{k+1}};\, \mu_{\tau_k},\, \Sigma_{\tau_k}\right)}_{\text{Gaussian，精确可算}}}
$$

第一项是初始噪声的 log-prior（固定），第二项每一个都是标准 Gaussian log-likelihood，**全部精确可算**，无需 Hutchinson 估计。

##### Step 4：代入 PPO policy gradient

原来需要的梯度：

$$
\nabla_\theta \mathcal{J} = \mathbb{E}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot A(s_t, a_t)\right]
$$

Flow-Noise 的处理：将环境时间步 $t$ 处的 $\log \pi_\theta(a_t | s_t)$ 替换为 denoising 序列的联合 log-likelihood：

$$
\log \pi_\theta(a_t | s_t) \;\longleftarrow\; \log \pi(\mathcal{A}_t | \mathbf{o}_t) = \log p(\mathbf{A}_t^0) + \sum_{k=0}^{K-1} \log \mathcal{N}(\mathbf{A}_t^{\tau_{k+1}}; \mu_{\tau_k}, \Sigma_{\tau_k})
$$

这在理论上等价（ReinFlow 有完整证明：在 one-layer MDP 框架下，对 denoising 序列的完整梯度等价于对最终动作的策略梯度）。

PPO 的 importance sampling ratio 也因此变为：

$$
r_\theta = \frac{\pi_\theta(\mathcal{A}_t | \mathbf{o}_t)}{\pi_{\theta_\text{old}}(\mathcal{A}_t | \mathbf{o}_t)} = \prod_{k=0}^{K-1} \frac{\mathcal{N}(\mathbf{A}^{\tau_{k+1}}; \mu_{\tau_k}^\theta, \Sigma_{\tau_k}^\theta)}{\mathcal{N}(\mathbf{A}^{\tau_{k+1}}; \mu_{\tau_k}^{\theta_\text{old}}, \Sigma_{\tau_k}^{\theta_\text{old}})}
$$

每一个比值都是两个 Gaussian 密度之比，**数值稳定，无需近似**。

---
> 推理时的处理

噪声网络 $\sigma_{\theta'}$ 在训练结束后**直接丢弃**：
$$
\mathbf{A}^{\tau+\delta}\big|_{\text{inference}} = \mathbf{A}^\tau + \mathbf{v}_\theta(\mathbf{A}^\tau, \mathbf{o}) \cdot \delta
$$
退化回原始确定性 ODE，**推理速度和原始 π₀/π₀.₅ 完全一致**，zero overhead。

### 2.2 Flow-SDE

**思路**：利用 probability flow ODE 与 SDE 的等价性（Song et al. 2020），将 flow matching 的 ODE 转换为保持相同边缘分布的 SDE：
$$
d\mathbf{A}^\tau = \underbrace{\left[\mathbf{v}^\tau + \frac{\sigma_\tau^2}{2\tau}\left(\mathbf{A}^\tau + (1-\tau)\mathbf{v}^\tau\right)\right]d\tau}_{\text{Drift}} + \underbrace{\sigma_\tau d\mathbf{w}_\tau}_{\text{Diffusion}}
$$

其中 score function 用速度场解析替换：$\nabla \log q_\tau(\mathbf{A}^\tau) = -\frac{\mathbf{A}^\tau}{\tau} - \frac{1-\tau}{\tau}\mathbf{v}^\tau$，噪声调度 $\sigma_\tau = a\sqrt{\frac{\tau}{1-\tau}}$。

在此基础上构建**两层 MDP**（two-layer MDP）：
- **内层**：denoising 步之间的转移（$\tau < 1$），状态为 $\bar{s}_t^\tau = (\mathbf{o}_t, \mathbf{A}_t^\tau)$
- **外层**：$\tau = 1$ 时执行最终动作 $\mathbf{A}_t^1$ 与环境交互，获得 reward，并重置为新 observation

奖励结构：$\tau < 1$ 时 reward 为 0，只在 $\tau = 1$ 完成环境交互后给予 $R_\text{ENV}$。

还引入**Hybrid ODE-SDE 采样**（借鉴 Mix-GRPO）：前段用 ODE 快速收敛，后段切换 SDE 引入探索，加速训练。

---

## 3. 策略优化

两种方案都接入 **PPO**，需要 critic 辅助估计 advantage $A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$。

critic 的设计是 ablation 的重点之一，论文对 critic 架构做了专项消融分析。

---

## 4. 架构关系总结

```
π₀ / π₀.₅ (SFT baseline)
       ↓  freeze/fine-tune VLM backbone
   Flow-Noise 分支                Flow-SDE 分支
  [注入可学习噪声网络]           [ODE → SDE 转换]
  [离散时间 MDP]                [两层 MDP]
       ↓                              ↓
   PPO (+ critic)            PPO (+ critic)
       ↓
   π_RL (推理时噪声丢弃 / 标准 ODE 采样)
```

---

## 5. 改进效果（实验前描述）

few-shot SFT 策略经 RL 微调后，性能可达到 full dataset baseline 的水平；同时在 ID 和 OOD 测试中均有显著提升，框架还支持通过 3D Gaussian Splatting 渲染的仿真器做 sim-to-real 零样本迁移。

这意味着该方法的核心价值在于：
- **数据效率**：少量 SFT 数据 + RL 可媲美大量 SFT
- **泛化性**：OOD 提升说明不是简单过拟合到训练环境
- **通用性**：同一框架适配 π₀ / π₀.₅ / GR00T N1.5（见 Appendix H）
