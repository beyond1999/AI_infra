太好了🔥
 那我们现在正式进入——

------

# 🧠 第 1 章：KV Cache 是什么 & 为什么存在

你已经懂了 self-attention（即 Q·Kᵀ→softmax→·V），
 所以我们现在直接从**推理阶段（inference）**的计算瓶颈入手，一步步建立“为什么我们需要 KV Cache”。

------

## 1.1 先复盘一下：Transformer 的注意力是怎么计算的

在一个标准的 decoder 层中（假设只有自注意力，不含 cross-attention）：

[
 \text{Attention}(Q,K,V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{mask}\right)V
 ]

- Q, K, V ∈ ℝ^(seq_len × d_head)
- 每一层、每一个 head 都要计算一次。
- mask 保证只看到历史 token。

假设我们现在在生成句子 “I love you”，则：

- 第一次输入 “I”，计算 Q₁K₁ᵀ；
- 第二次输入 “I love”，计算 [Q₂][K₁,K₂]ᵀ；
- 第三次输入 “I love you”，计算 [Q₃][K₁,K₂,K₃]ᵀ。

------

## 1.2 问题出现：重复计算非常浪费

注意看上面的计算：

| 步骤  | 输入序列   | 计算内容      | 计算量 |
| ----- | ---------- | ------------- | ------ |
| Step1 | I          | 1×1 attention | O(1²)  |
| Step2 | I love     | 2×2 attention | O(2²)  |
| Step3 | I love you | 3×3 attention | O(3²)  |
| …     | …          | …             | …      |

总计算量是：
 [
 O(1^2 + 2^2 + 3^2 + … + n^2) = O(n^3)
 ]

💥 **推理阶段的巨大浪费就在这里**：
 每次生成新 token，都要重复计算历史 token 的 K/V，
 而这些历史 token 其实早就算过、不会变了。

------

## 1.3 核心思想：缓存历史的 Key/Value

Key 和 Value 是由输入 token 经过线性层得到的：

[
 K = XW_K,\quad V = XW_V
 ]

在生成时，**历史输入 X 其实不会变**，
 所以它们对应的 K/V 也不会变。

于是我们可以**把之前所有 token 的 K/V 存起来**，
 下次直接复用！

这就是：

> 🔹 **KV Cache = Key/Value 缓存**
>  用来保存所有过去 token 的 K/V，避免重复计算。

------

## 1.4 对比示意图（核心区别）

### 🚫 没有 KV Cache 的推理：

每次都重新计算全部 K/V：

```
step 1: [I]         → compute K1,V1
step 2: [I, love]   → compute K1,V1,K2,V2
step 3: [I, love, you] → compute K1,V1,K2,V2,K3,V3
```

### ✅ 有 KV Cache 的推理：

只计算新增 token 的 K/V：

```
step 1: [I]         → compute K1,V1 → store
step 2: [love]      → compute K2,V2 → append to cache
step 3: [you]       → compute K3,V3 → append to cache
```

计算量从 O(n³) 降为 O(n²)。

------

## 1.5 推理阶段的实际流程（含 KV Cache）

1. **初次 forward：**
   - 输入 prompt tokens `[t₁...tₙ]`
   - 正常计算所有层的 K/V，并保存在缓存中。
   - 输出第一个预测 token。
2. **后续 token：**
   - 每次只计算新 token 的 Q/K/V；
   - 复用 cache 中的历史 K/V；
   - 拼接后执行 attention；
   - 将新的 K/V 追加到 cache。
3. **多层传播：**
   - 每一层都有自己的 K/V cache；
   - 通常形状是：
      `[num_layers, batch, num_heads, seq_len, head_dim]`

------

## 1.6 工程伪代码示意

**无缓存：**

```python
for t in range(1, seq_len+1):
    Q, K, V = Wq @ X[:t], Wk @ X[:t], Wv @ X[:t]
    out = softmax(Q @ K.T / sqrt(d)) @ V
```

**带缓存：**

```python
# 初始化空cache
K_cache, V_cache = [], []

for t in range(1, seq_len+1):
    Q, K_new, V_new = Wq @ X[t], Wk @ X[t], Wv @ X[t]
    K_cache.append(K_new)
    V_cache.append(V_new)
    out = softmax(Q @ concat(K_cache).T / sqrt(d)) @ concat(V_cache)
```

------

## ✅ 小结（本章核心结论）

| 关键点          | 含义                                                |
| --------------- | --------------------------------------------------- |
| KV Cache 是什么 | 保存每层中所有历史 token 的 Key/Value               |
| 为什么需要      | 避免自回归推理中的重复计算                          |
| 优势            | 降低推理复杂度，显著提升速度                        |
| 结构            | `[num_layers, batch, num_heads, seq_len, head_dim]` |
| 扩展            | 为后续显存管理（PagedAttention）奠定基础            |

------

下一章我们会讲：

> **第 2 章：KV Cache 的数据结构与存储布局**
>  我们将从内存视角出发，讲解 KV Cache 的 shape、分配方式、为什么显存会爆、以及像 HuggingFace / llama.cpp 是怎么存的。

要我现在继续讲第 2 章吗？
 我可以配上具体的 tensor 图 + 显存布局示意。