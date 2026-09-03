1. # 背景与动机

问题：GRPO 对每个 prompt 采样 $N$ 条回答并计算组内标准化优势，其中组均值 $\bar{r}^{(k)}$ 是用 $N$ 个样本估计的。当 $N$ 较小时，估计噪声会很大，导致优势估计方差高，训练不稳定。

方案：把同一批次内 $K$ 个 prompt 的组均值看成对各自真实期望奖励的独立观测，使用 James–Stein 收缩向全局均值回拉，压低组均值估计的均方误差，从而得到更稳定的优势信号。

收益：训练更稳、收敛更快、支持更小的 $N$，且几乎无额外实现开销。

## 2. 理论核心（极简版）

### 假设
奖励已做过组内标准化（或近似等比例缩放），使组内单次奖励方差 $\sigma^2 \approx 1$。对于第 $k$ 个 prompt，组均值

$$
X_k = \frac{1}{N}\sum_{i=1}^{N} r_i^{(k)}
$$

服从

$$
X_k \sim \mathcal{N}(\mu_k, V), \qquad V = \frac{1}{N}.
$$

这些组均值是我们要同时估计的多个正态均值。

### James–Stein 估计量
收缩中心取为所有组均值的平均值 $\bar{X}$，则

$$
\tilde{\mu}_k = \bar{X} + \left(1 - \frac{c}{S}\right)(X_k - \bar{X}),
$$

其中

$$
c = V(K-3), \qquad S = \sum_{k=1}^{K}(X_k - \bar{X})^2.
$$

当 $V = 1/N$ 时，

$$
c = \frac{K-3}{N}.
$$

如果 $S < c$，收缩因子会变成负值，此时将其截断为 $0$，即直接令估计为全局均值 $\bar{X}$。

### 为什么有效
收缩后的 $\tilde{\mu}_k$ 比原始 $X_k$ 的 MSE 更小（Stein 引理保证）。用它作为基线计算优势

$$
A_i^{(k)} = r_i^{(k)} - \tilde{\mu}_k
$$

时，优势方差会下降。

## 3. 算法实现方案

### 3.1 输入与位置
插入点：GRPO 训练循环中，在每个 batch 得到奖励矩阵后、计算优势之前。

- 输入：奖励矩阵 $\text{rewards}$，形状为 $(K, N)$，其中 $K$ 个 prompt，每个 prompt 有 $N$ 个回答。
- 输出：收缩后的组均值 $\mu_{\text{shrunk}}$，形状为 $(K,)$，用于计算新的优势。

### 3.2 步骤
1. 计算组均值：
   $$
   X = \mathrm{mean}(\text{rewards}, \mathrm{dim}=1)
   $$
   结果形状为 $(K,)$。

2. 计算全局均值：
   $$
   \bar{X} = \mathrm{mean}(X)
   $$

3. 计算离差平方和：
   $$
   \mathrm{diff} = X - \bar{X}, \qquad S = \sum \mathrm{diff}^2
   $$

4. 计算收缩强度：
   $$
   c = \frac{K-3}{N}
   $$

5. 若 $K < 4$，直接返回原始均值，不做收缩。

6. 计算收缩因子：
   $$
   \text{shrink\_factor} = 1 - \frac{c}{\max(S, \varepsilon)}
   $$

7. 若 $S = 0$（所有组均值相等），则收缩因子取 $1$，不收缩。

8. 应用截断：
   $$
   \text{shrink\_factor} = \max(0, \text{shrink\_factor})
   $$

9. 计算收缩后的组均值：
   $$
   \mu_{\text{shrunk}} = \bar{X} + \text{shrink\_factor} \cdot \mathrm{diff}
   $$

10. 计算优势：
   $$
   A_i^{(k)} = \text{rewards}[k, i] - \mu_{\text{shrunk}}[k]
   $$

如果还需要组内标准差标准化，可以在后面再除以组内标准差，或直接把奖励按原始标准差归一化。

### 3.3 伪代码

```python
def james_stein_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """
    Args:
        rewards: Tensor of shape (K, N), float.
    Returns:
        advantages: Tensor of shape (K, N), float.
    """
    K, N = rewards.shape
    X = rewards.mean(dim=1)  # (K,) 组均值

    if K < 4:
        # 不做收缩，退化为原始中心化的优势
        X_bar = X.mean()
        return rewards - X_bar.unsqueeze(1)

    X_bar = X.mean()
    diff = X - X_bar
    S = (diff ** 2).sum()
    c = (K - 3) / N
    eps = 1e-8

    shrink_factor = 1.0 - c / max(S, eps)
    shrink_factor = torch.clamp(shrink_factor, min=0.0)
    mu_shrunk = X_bar + shrink_factor * diff  # (K,)

    # 优势 = 奖励 - 收缩后的基线
    advantages = rewards - mu_shrunk.unsqueeze(1)
    return advantages
```

变体：如果想保留组内标准差缩放，可在后面再除以组内标准差（或直接用奖励原始标准差）。
