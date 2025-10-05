非常好🔥
 我们现在进入——

------

# 🧩 第 2 章：KV Cache 的数据结构与存储布局

> 你已经知道 KV Cache 是“缓存历史 token 的 Key 和 Value”。
>  本章我们要搞清楚三个问题：
>
> 1. 它到底在内存里长什么样？
> 2. 为什么显存会爆？
> 3. 工程上是如何分配和管理这块内存的？

------

## 2.1 Key / Value 的形状回顾

在 self-attention 里，每层都有自己的 `Q, K, V`：

[
 Q = X W_Q,\quad K = X W_K,\quad V = X W_V
 ]

其中：

- batch size = B
- head 数 = H
- 每个 head 的维度 = D_h
- 当前序列长度 = T

因此：

| 名称 | 形状           |
| ---- | -------------- |
| Q    | [B, H, T, D_h] |
| K    | [B, H, T, D_h] |
| V    | [B, H, T, D_h] |

------

## 2.2 KV Cache 在推理过程中的增长方式

自回归推理时，我们每次生成 1 个新 token：

假设当前 step=t 时：

- 新的 key: `K_new` = [B, H, 1, D_h]
- 新的 value: `V_new` = [B, H, 1, D_h]

我们要把它追加到历史 cache 里：

```
K_cache: [B, H, t-1, D_h]
V_cache: [B, H, t-1, D_h]

↓ append

K_cache: [B, H, t, D_h]
V_cache: [B, H, t, D_h]
```

所以，**每生成一个 token，cache 在 seq_len 维度上增长一格**。

------

## 2.3 多层 Transformer 的缓存结构

每一层的 attention 都有自己的 K/V：

- 层数 = L
- 所以整个模型的 cache：

[
 \text{shape} = [L, B, H, T, D_h]
 ]

实际实现中通常是一个 **list（长度=L）**，
 每层包含一对 `(K_i, V_i)`。

伪结构：

```python
past_key_values = [
    (K_1, V_1),
    (K_2, V_2),
    ...
    (K_L, V_L)
]
```

------

## 2.4 显存占用估算（为什么爆显存）

假设：

- L = 32 层
- H = 32 头
- D_h = 128
- 每个元素是 float16 (2 字节)
- T = 2048 (上下文长度)
- B = 1

计算单层单头：
 [
 2048 × 128 × 2 = 0.5 \text{MB}
 ]
 乘上 32 头 → 16 MB
 再乘 32 层 → 512 MB
 仅 KV Cache 一项就占了 **半个 G 显存**！

🧨 所以大模型（Llama-70B）上下文若拉长到 8k / 16k，显存爆炸是必然的。
 这就是后续出现 **PagedAttention / KV quantization** 的背景。

------

## 2.5 存储布局：连续 vs 分块

### ✅ **方式一：连续分配（naive）**

最简单方式：为每层、每头、每 batch 分配一块连续显存。

优点：

- 实现简单（常见于 HuggingFace 模型）

缺点：

- 不灵活，显存浪费大；
- 不利于动态 batching；
- 每个请求的 cache 不可重用。

示意：

```
K_cache[layer][batch][head][:seq_len][:head_dim]
```

------

### ✅ **方式二：分块管理（Paged / Chunked）**

在 vLLM 等系统中，会将 KV Cache 按页切分：

- 每页大小固定（如 16 或 32 token）；
- 一个请求的序列由若干页组成；
- 页之间可动态复用、回收；
- GPU 内存布局更像“操作系统的虚拟内存页表”。

好处：

- 避免碎片；
- 显存利用率高；
- 支持动态 batching；
- 能同时服务成千上万个 session。

示意：

```
Physical GPU memory (paged):
[Page_1][Page_2][Page_3]...[Page_N]

Logical sequence (for one request):
→ [Page_3][Page_8][Page_10]
```

映射关系通过一个 “page table” 保存。

------

## 2.6 工程实例：HuggingFace 与 llama.cpp

### 🧩 HuggingFace Transformers

使用 Python list 管理各层 cache：

```python
past_key_values = [
    (K_0, V_0), (K_1, V_1), ..., (K_L, V_L)
]
# 每个 K/V 是 [B, H, T, D_h]
```

每步 forward：

```python
K_cat = torch.cat([past_key, K_new], dim=2)
V_cat = torch.cat([past_value, V_new], dim=2)
```

显存连续，逻辑简单，但无法并发。

------

### 🧩 llama.cpp

底层用 C/C++ 数组存储：

```c
// 假设 float16
float16_t * memory_k = malloc(L * H * max_seq_len * D_h * sizeof(float16_t));
float16_t * memory_v = malloc(L * H * max_seq_len * D_h * sizeof(float16_t));
```

每步写入：

```c
memcpy(&memory_k[offset], K_new, size);
memcpy(&memory_v[offset], V_new, size);
```

这里的 offset 由当前 token index 决定。
 实现非常高效，但依旧是**连续布局**，不支持动态 page。

------

## 2.7 总结对比

| 方案     | 特点         | 优点                    | 缺点           |
| -------- | ------------ | ----------------------- | -------------- |
| 连续分配 | 固定显存块   | 简单、快速              | 浪费、不可复用 |
| 分块分配 | Page-based   | 高效复用、动态 batching | 实现复杂       |
| 分层结构 | 每层单独管理 | 可独立优化              | 有同步开销     |

------

## ✅ 本章总结

| 概念            | 要点                                                |
| --------------- | --------------------------------------------------- |
| KV Cache 的结构 | `[num_layers, batch, num_heads, seq_len, head_dim]` |
| 计算量          | 只保存 K/V，不保存 Q                                |
| 存储布局        | 连续 or 分页                                        |
| 显存占用        | 主要瓶颈来源                                        |
| 后续优化方向    | PagedAttention、ChunkedKV、Quantization             |

------

下一章我们就要讲：

> **第 3 章：推理循环中的 KV Cache 读写机制**
>  即：在每个生成 step 中，Q 如何与 Cache 交互，Cache 是怎么更新的，Attention mask 怎么起作用。

要我现在继续讲第 3 章吗？
 这章我会用**伪代码 + 图**详细演示 “一次生成一个 token 时 KV Cache 的读写过程”。