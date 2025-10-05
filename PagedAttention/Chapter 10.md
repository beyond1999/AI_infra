非常好🔥
 那我们进入 **第 10 章：PagedAttention 的掩码机制与特殊场景支持**。
 这一章是整个体系的“兼容性关键”——PagedAttention 不只是快、灵活，还必须**在分页结构下保持注意力计算语义正确**，包括因果掩码、前缀缓存、MoE、工具调用等复杂场景。

------

# 🎭 第 10 章：掩码机制与特殊场景（Masking & Prefix Handling）

------

## 🧠 10.1 为什么分页会影响 Mask？

传统连续 KV Cache：

- attention mask 是一个上三角矩阵；
- GPU kernel 能直接通过 token index 生成 mask。

但在分页结构中：

- token 不再连续；
- 页表里的每个 block 可能来自不同来源（共享 / 拷贝 / 撤销）；
- block 之间可能存在**段边界（segment boundary）**。

因此，PagedAttention 必须在 kernel 内支持**分段化的掩码计算**。

------

## ⚙️ 10.2 分页下的因果掩码（Causal Mask）

在自回归模型中：
 [
 \text{token}*i 只能看到 \text{token}*{\le i}
 ]

分页结构下的实现：

```
每个 block_i:
  只允许访问 [block_0 ... block_i] 内的 token
```

注意力矩阵的掩码形态如下：

```
        Past Blocks → 
      ┌───────────────────────────────┐
block0│██████████████████████████████ │
block1│░░████████████████████████████ │
block2│░░░░██████████████████████████ │
block3│░░░░░░████████████████████████ │
      └───────────────────────────────┘
          ↑ 当前解码块
```

> ░ 表示被 mask 的区域；
>  每个 block 内部仍是局部上三角结构。

------

## 💡 10.3 实现方式：Block-wise Mask

PagedAttention kernel 在计算前会生成一个「块级掩码表」：

| 当前块 | 可见块列表 |
| ------ | ---------- |
| blk#0  | [0]        |
| blk#1  | [0,1]      |
| blk#2  | [0,1,2]    |
| blk#3  | [0,1,2,3]  |

每个线程块只在这些块之间计算注意力分数。

伪代码：

```cpp
for (int i = 0; i <= curr_blk; ++i) {
    compute_attention(Q_blk, K_blk[i], mask=True);
}
```

> 因此，分页机制与因果掩码天然兼容，只要每个 block 能正确标记“逻辑顺序”。

------

## 📦 10.4 前缀缓存（Prefix Cache）

PagedAttention 还要支持“复用 Prompt”这种情况：

- 大模型服务常常缓存部分上下文；
- 用户每次请求都带有公共前缀（例如系统指令）。

### 传统做法：

重新 encode 每次 prompt → 浪费。

### PagedAttention 做法：

- 前缀页单独缓存；
- 新请求的页表直接复用这些页；
- 只在新 token 之后分配新页。

示意：

```
System Prefix: [blk#0, blk#1]
User Request:  [blk#2, blk#3]
```

复用时：

```
page_table = prefix_blocks + new_blocks
```

> ✅ 不再重新计算 KV，
>  ✅ 不占额外显存（因为 prefix 页共享 refcount++）。

------

## 🔁 10.5 复制写时保护（Copy-On-Write）

如果多个请求共享同一前缀页：

- 当其中一个要写入 prefix block → 触发 Copy-On-Write；
- 创建该页的副本；
- 页表更新为新 block。

```python
if refcount[blk] > 1 and needs_write:
    new_blk = alloc_block()
    copy_block(new_blk, blk)
    page_table[i] = new_blk
    refcount[blk] -= 1
```

------

## 🧩 10.6 多段输入（Segmented Input）

LLM 的上下文可能包含多种输入段，例如：

```
[system_prompt] [user_prompt] [function_call] [response]
```

每个段的 Mask 规则可能不同：

- system_prompt 可被所有段访问；
- user_prompt 可见；
- function_call 仅在特定区域可访问；
- response 段遵循因果掩码。

PagedAttention 用**Segment-Aware Mask**来实现：

1. 每个 block 有 segment_id；
2. Kernel 根据 segment relation matrix 生成 mask。

| From ↓ / To → | System | User | Func | Resp |
| ------------- | ------ | ---- | ---- | ---- |
| System        | ✅      | ✅    | ✅    | ✅    |
| User          | ❌      | ✅    | ✅    | ✅    |
| Func          | ❌      | ❌    | ✅    | ✅    |
| Resp          | ❌      | ❌    | ❌    | ✅    |

这张表由 host 端在推理时生成，kernel 在 block 粒度读取。

------

## 🌳 10.7 MoE（Mixture-of-Experts）场景

MoE 模型中，每层可能选择不同 expert，
 每个 expert 的 KV Cache 结构可能独立。

PagedAttention 的做法：

- 每个 expert 拥有独立 page pool；
- 页表在 `(layer_id, expert_id)` 维度上扩展；
- Scheduler 动态选择哪个 expert 页表可见。

示意：

```
Layer 8:
 Expert0 → [blk#40, blk#41]
 Expert1 → [blk#42]
 Expert2 → [blk#43, blk#44]
```

kernel 调用：

```cpp
for (int exp : active_experts[layer_id]) {
    gather_blocks(exp_page_table[exp])
    compute_attention(...)
}
```

------

## 🧮 10.8 Function Call / Tool Use 场景

在多轮对话 / 工具调用中，LLM 的上下文常出现：

```
[用户输入] → [模型生成工具参数] → [外部工具返回结果] → [继续生成]
```

PagedAttention 需要：

- 识别每一段的可见范围；
- 快速切换当前活跃 segment；
- 支持部分重用早期结果。

实现方式：

- 每段单独页表；
- Scheduler 维护 segment 栈；
- 动态拼接有效页表。

例如：

```
page_table_active = [user_blocks] + [tool_result_blocks]
```

------

## 📊 10.9 Mask 生成性能优化

由于每步解码都可能涉及不同的 block 可见性，
 kernel 在 mask 生成上采用了轻量策略：

1. **提前计算块间关系矩阵**

   ```text
   block_relation[i][j] = (i >= j)
   ```

2. **仅在页表更新时刷新 mask**

   - decode 阶段只追加 block；
   - 前缀部分不变；
   - 因此 mask 复用率极高。

性能收益：mask generation 开销 < 1%。

------

## ✅ 10.10 小结

| 场景     | 支持机制                             | 效果               |
| -------- | ------------------------------------ | ------------------ |
| 因果掩码 | Block-wise causal mask               | 正确的上下文可见性 |
| 前缀缓存 | 页共享 + refcount                    | 高复用率           |
| 多段输入 | Segment-aware mask                   | 灵活语义控制       |
| MoE      | 多 page pool + layer/expert 维度扩展 | 分支独立           |
| 工具调用 | 多页表拼接                           | 快速切换上下文     |

> **一句话总结：**
>  PagedAttention 不只是“分页显存优化”，
>  它在掩码层面实现了对复杂语义上下文的精确控制，
>  保证任何场景下注意力的逻辑正确性。

------

下一章（第 11 章）我们将讲：

> 🧮 **多 GPU 与分布式分页机制** ——
>  PagedAttention 如何扩展到多卡、流水并行和跨节点场景（Page Sharding、Tensor Parallel、NVLink/RDMA 调页）。

是否继续进入第 11 章？