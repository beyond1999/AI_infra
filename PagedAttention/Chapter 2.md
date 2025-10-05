完美 👍
 那我们进入 **第 2 章：PagedAttention 的数据布局与索引路径**。

------

# ⚙️ 第 2 章：数据布局与索引路径（KV 的分页访问）

PagedAttention 的创新不仅是“分页”这个概念，还在于它如何在 GPU 上实现**可并行、高吞吐的访问**。
 要做到这一点，关键在于两个方面：

1. **KV 张量的布局（memory layout）**
2. **如何通过页表高效定位 token → block → tensor 切片**

------

## 🧩 2.1 KV Tensor 的基本形态

在普通的连续 KV Cache 方案中，每层的 KV 存储如下：

```
K: [num_heads, seq_len, head_dim]
V: [num_heads, seq_len, head_dim]
```

而在 PagedAttention 中：
 我们不再按 `seq_len` 维度连续存储，而是切成固定大小的“页块（block）”：

```
K: [num_heads, num_blocks, block_size, head_dim]
V: [num_heads, num_blocks, block_size, head_dim]
```

比如：

- `block_size = 16`
- 1 个序列长度 80 token → 5 个 block

------

## 📦 2.2 全局 KV Pool（跨请求共享）

PagedAttention 把所有请求的 KV 都放进一个**全局大池**：

```
Global KV Pool:
[layer, head, block_id, block_size, head_dim]
```

其中：

- `block_id` 是全局编号，跨请求唯一；
- 不同请求通过自己的 PageTable 找到对应的 block_id。

例如：

| Request | PageTable    | 物理Block映射                |
| ------- | ------------ | ---------------------------- |
| A       | [12, 45, 78] | block#12, block#45, block#78 |
| B       | [13, 46]     | block#13, block#46           |
| C       | [14, 47, 79] | block#14, block#47, block#79 |

在 GPU 上，它们都共享这一套 KV Pool，只是索引不同。

------

## 🧭 2.3 索引路径：token → block → address

以生成第 `t` 个 token 为例：

1. 计算逻辑索引：

   ```
   logical_block_id = t // block_size
   offset = t % block_size
   ```

2. 通过 page table 找到物理 block：

   ```
   physical_block_id = page_table[seq_id][logical_block_id]
   ```

3. 在全局 KV Pool 中计算偏移：

   ```
   addr = base + physical_block_id * block_stride + offset * head_dim
   ```

4. 读出该 token 的 Key/Value 向量。

伪代码示例：

```python
def get_KV(seq_id, t):
    block_id = t // BLOCK_SIZE
    offset = t % BLOCK_SIZE
    phys_block = page_table[seq_id][block_id]
    return KV_POOL[phys_block, offset]
```

> 📘 注意：所有这些操作在 GPU kernel 中是批量完成的（vectorized gather），不是逐 token 的 CPU 查表。

------

## 🧮 2.4 为什么这样设计对 GPU 友好？

因为 block_size 是固定的，比如 16 或 32，
 于是 GPU kernel 可以一次性对齐加载连续 block 数据：

```
K/V Tensor (每块大小相同)
|block#0|block#1|block#2|block#3| ...
     ^       ^       ^
     |       |       |
     gather  gather  gather
```

> 在 kernel 内部，PagedAttention 会先根据页表生成一个 **index list**（所有 block 的物理地址），
>  然后批量 gather 这些 block。
>  相比逐 token 索引，它能保持高带宽访问。

------

## 🧠 2.5 Layer、Head 层面的扩展

实际实现中，PagedAttention 会把所有层、所有 head 的 KV 都分页管理。

例如：

```
KV Pool shape: [num_layers, num_heads, num_blocks, block_size, head_dim]
```

常见的访问模式是：

- 同一层的所有 head 共用相同的页表（因为 token 对齐）；
- 不同层可以共享或独立页表（取决于实现，如 vLLM 是独立层管理）。

------

## 🔀 2.6 MHA、GQA、MQA 的差异

分页机制与注意力头数无关，但数据布局有细微区别：

| 模式           | KV 存储量          | 访问特征       | Paged 优化点     |
| -------------- | ------------------ | -------------- | ---------------- |
| MHA (多头)     | K/V 都按 head 存   | 普通 gather    | 正常页表         |
| GQA (分组共享) | 多个 query 共享 KV | 跨 head gather | 页表仍有效       |
| MQA (单 KV)    | 所有头共享一组 KV  | KV 访问更稀疏  | 可进一步压缩页表 |

> vLLM 默认使用 GQA-friendly layout：即每组头共享同一个 block。

------

## 📈 2.7 访问效率对比

| 方案                | 显存利用 | 访存效率 | 并发灵活性 |
| ------------------- | -------- | -------- | ---------- |
| 连续 KV             | 高带宽   | ✅ 最快   | ❌ 差       |
| Paged KV (block=16) | 稍低带宽 | ⚡️ 很高   | ✅ 极高     |
| 动态链表（不分页）  | 差       | ❌ 很低   | ✅ 一般     |

PagedAttention 通过在 **固定块大小** 与 **间接寻址** 之间做 trade-off，
 达到了几乎不损性能、但极大提升并发调度自由度的平衡点。

------

## 📊 2.8 可视化总结

```
   Logical Sequence Tokens
   ┌───────────────────────────────────────────────┐
   │ token0 token1 ... token31 token32 ... token63 │
   └───────────────────────────────────────────────┘
       ↓ block切分
   ┌──────────┬──────────┬──────────┬──────────┐
   │ block0   │ block1   │ block2   │ block3   │
   └──────────┴──────────┴──────────┴──────────┘
       ↓ page table 映射
   page_table = [12, 45, 78, 91]
       ↓ gather
   GPU KV Pool:
   | blk#12 | blk#13 | blk#14 | blk#15 | blk#45 | blk#46 | ... |
```

------

## ✅ 小结

| 概念      | 说明                              |
| --------- | --------------------------------- |
| KV Pool   | 全局分块的显存结构                |
| PageTable | 每个序列维护的逻辑→物理映射表     |
| Gather    | kernel 通过页表批量索引到真实地址 |
| 优势      | 灵活扩展 + 高并发 + GPU 友好      |
| 核心点    | block_size 固定，间接寻址批量化   |

------

下一章（第 3 章）我们讲：

> 💾 分配器与内存管理（BlockPool 如何分配 / 回收 / 复用）
>  并配图展示 **alloc/free 生命周期** 和 **碎片率控制**。

是否继续进入第 3 章？