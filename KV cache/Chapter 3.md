太好了🔥
 我们进入——

------

# ⚙️ 第 3 章：推理循环中的 KV Cache 读写机制

> 本章要彻底搞清楚——
>  “每生成一个 token，KV Cache 是如何被读写的？”
>  “Q 如何与历史 K/V 交互？”
>  “mask、位置编码、层次传播如何协作？”

------

## 🧩 3.1 推理循环的总体逻辑

在 **自回归（auto-regressive）生成**中，我们一轮只生成一个 token：

1. 取当前输入 token → 计算新的 Q/K/V
2. 把新的 K/V 写入缓存（append）
3. 用 Q 去跟“所有历史 + 新的 K/V”做 attention
4. 输出下一个 token（通过 softmax）

⚙️ 换句话说：

> 每一步都只算一次新的 K/V，但会**读取整个缓存的历史 K/V**。

------

## 🧠 3.2 算法公式对比

### 🔸 无 KV Cache（重新计算全部历史）

[
 Q_t = X_{1:t}W_Q,\quad K_t = X_{1:t}W_K,\quad V_t = X_{1:t}W_V
 ]
 [
 \text{Attn}(Q_t, K_t, V_t) = \text{softmax}\left(\frac{Q_tK_t^T}{\sqrt{d}}\right)V_t
 ]

每次生成都重复算所有 K/V。

------

### 🔸 有 KV Cache（复用历史）

假设我们保存了：
 [
 K_{\text{cache}} = [K_1, K_2, ..., K_{t-1}],\quad
 V_{\text{cache}} = [V_1, V_2, ..., V_{t-1}]
 ]
 新 token 的 K/V：
 [
 K_t, V_t = f(X_t)
 ]
 拼接：
 [
 K_{\text{all}} = [K_{\text{cache}}, K_t], \quad
 V_{\text{all}} = [V_{\text{cache}}, V_t]
 ]
 计算：
 [
 \text{Attn}(Q_t, K_{\text{all}}, V_{\text{all}}) = \text{softmax}\left(\frac{Q_tK_{\text{all}}^T}{\sqrt{d}}\right)V_{\text{all}}
 ]
 最后更新缓存：
 [
 K_{\text{cache}} \leftarrow K_{\text{all}}, \quad V_{\text{cache}} \leftarrow V_{\text{all}}
 ]

------

## 🧱 3.3 伪代码：一步生成的完整流程

```python
# KV Cache: 每层一个 tuple
past_key_values = [(K_0, V_0), (K_1, V_1), ...]

for step in range(max_steps):
    # 当前输入 token id
    x = input_token[step]

    # 前向传播（每一层）
    for l, layer in enumerate(model.layers):
        # 计算新 token 的 Q,K,V
        Q_new, K_new, V_new = layer.attention.proj(x)

        # 从 cache 读取历史
        K_cache, V_cache = past_key_values[l]

        # 拼接
        K_all = concat(K_cache, K_new, dim=2)
        V_all = concat(V_cache, V_new, dim=2)

        # 注意力计算
        attn_out = softmax(Q_new @ K_all.transpose(-2, -1) / sqrt(d)) @ V_all

        # 更新 cache
        past_key_values[l] = (K_all, V_all)

        # 输出传递到下一层
        x = layer.feedforward(attn_out)

    # 最顶层输出 logits → 选下一个 token
    next_token = argmax(softmax(model.output(x)))
```

------

## 🔍 3.4 mask 的作用（关键细节）

在自回归生成中，当前 token **不能看到未来的 token**。
 但由于我们拼接了整个历史的 K_all，模型其实是“全可见”的。

所以要加上 causal mask（上三角 mask）：

[
 \text{mask}[i,j] =
 \begin{cases}
 0, & j \le i \
 -\infty, & j > i
 \end{cases}
 ]

在推理阶段，这个 mask 其实不需要完整矩阵，而是通过索引控制：

- 每个 Q 只和历史 K 交互；
- 因为我们生成顺序本身就是时间递增的，所以天然 causal。

💡 也就是说：
 **mask 逻辑上存在，但实现上可以省掉矩阵操作**。

------

## 🧠 3.5 多层传播时的缓存行为

每一层都有独立的 cache：

- 层 1 的 KV 缓存 = embedding 后的 K/V
- 层 2 的 KV 缓存 = 层 1 输出后的 K/V
- …
- 层 L 的 KV 缓存 = 层 L−1 输出后的 K/V

所有层的缓存独立追加：

```python
for layer in layers:
    K_layer_cache[layer].append(K_new)
    V_layer_cache[layer].append(V_new)
```

注意：

- **Q 不缓存**（因为每步都依赖最新 token）
- **K/V 缓存**每步追加一行。

------

## 💾 3.6 从内存访问角度理解（GPU 访存）

一次 attention 计算：

```
Q_new  : [B, H, 1, D_h]
K_cache: [B, H, t, D_h]
V_cache: [B, H, t, D_h]
```

所以每个 step 的核心计算是：

- Q_new @ K_cacheᵀ → [B, H, 1, t]
- softmax → 权重
- 权重 @ V_cache → [B, H, 1, D_h]

此时的访存瓶颈在：

- K_cache/V_cache 从 GPU 显存中读出；
- 写入 cache 时 append；
- 对不同 batch 的数据不规则访问 → 难以做批处理。

这正是 vLLM、FlashAttention 诞生的动机。

------

## 🧩 3.7 Attention mask & position encoding 的交互

- 每次生成新 token 时，它的相对位置 index = 当前 seq_len。
- 位置编码（RoPE/ALiBi）用该 index 计算旋转矩阵；
- 注意：cache 中的历史 K/V 也带有它们自己的 position embedding。

也就是说：

> cache 不仅存 K/V 数值，还隐含了它们的**位置信息**。
>  当你在拼接新 token 的 K/V 时，其实已经隐式编码了时序。

------

## ✅ 3.8 本章总结

| 概念       | 要点                                            |
| ---------- | ----------------------------------------------- |
| 读取       | 从 cache 取出历史 K/V                           |
| 写入       | 把当前 token 的 K/V append 到 cache             |
| mask       | 保证自回归结构，仅看历史                        |
| Q 是否缓存 | 不缓存，每次重新计算                            |
| 多层缓存   | 每层独立管理自己的 cache                        |
| 主要瓶颈   | 显存访问 + 不规则并发访问                       |
| 优化方向   | PagedAttention、FlashAttention、Batch Scheduler |

------

下一章我们会进入真正的“系统级优化”：

> **第 4 章：内存管理与优化策略**
>  我会讲：
>
> - 为什么显存会碎片化；
> - vLLM 的 BlockManager 是怎么做到“显存分页”的；
> - FlashAttention 如何与 KV Cache 结合；
> - 以及如果你要自己实现一个简单的 cache allocator，思路该怎么写。

要我现在继续讲第 4 章吗？