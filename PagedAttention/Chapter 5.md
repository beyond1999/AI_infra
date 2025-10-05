非常棒👏
 那我们进入 **第 5 章：PagedAttention 的 Kernel 实现与 FlashAttention 的关系**。
 这是 PagedAttention 最“硬核”的部分 —— 如何让分页化的 KV Cache 依然能在 GPU 上跑得飞快。

------

# 🧮 第 5 章：PagedAttention Kernel 实现原理

PagedAttention 的核心挑战是：

> **在分页内存结构下，仍然保持高吞吐、高带宽的注意力计算性能。**

也就是说，我们要让「不连续的 K/V」在 GPU 上的访问几乎不比连续存储慢。

------

## 🧩 5.1 传统注意力计算回顾

对于一个序列长度 ( L )，注意力计算是：

[
 \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
 ]

其中：

- ( Q ) = 当前步 query
- ( K,V ) = 历史缓存的 key/value

传统实现需要访问形如：

```
K, V shape = [num_heads, seq_len, head_dim]
```

问题是：PagedAttention 的 K/V 不再是连续的 `[0:L]`，而是分布在多个 page/block 里。

------

## ⚙️ 5.2 分页化的 K/V 存储结构

PagedAttention 的全局 KV Pool 形状：

```
K/V: [num_layers, num_heads, num_blocks, block_size, head_dim]
```

每个序列有自己的页表：

```
page_table = [12, 45, 78, 91]   # 映射到物理block
```

目标是：

- 从这 4 个 block 中 gather 出真实的 token 数据；
- 在 kernel 内部拼成逻辑顺序；
- 执行 ( QK^T )、softmax、( *V )。

------

## 🚀 5.3 GPU kernel 的总体思路

PagedAttention 的 kernel 是一种 **“索引感知型批量注意力算子”**，大致包含以下步骤：

### Step 1️⃣ 构建索引表（Index List）

在 host 侧，根据 batch 中所有序列的页表，生成：

```text
block_indices = [12, 45, 78, 91, 13, 46, 14, 47, 79, ...]
block_offsets = [0, 16, 32, 48, ...]   # 逻辑拼接偏移
```

### Step 2️⃣ GPU 端 gather K/V 数据

GPU 内核启动时，每个线程块负责若干个 query token：

```cpp
for (int b = 0; b < num_blocks; ++b) {
    int block_id = block_indices[b];
    K_block = load_from(KV_pool + block_id * block_stride);
    V_block = load_from(KV_pool + block_id * block_stride);
    // 拼接到连续缓冲区 (或直接计算)
}
```

> ⚡️ 优化点：每个 block 是连续的、大小固定，因此可以高效 `ldmatrix` / `cp.async` 载入。

### Step 3️⃣ 计算 Attention(Q, K, V)

- 将所有 K/V block 逻辑拼接；
- 执行标准 attention 计算；
- 或者调用 FlashAttention 内核（下节解释）。

------

## 🧮 5.4 FlashAttention 与 PagedAttention 的关系

| 层面     | FlashAttention                   | PagedAttention               |
| -------- | -------------------------------- | ---------------------------- |
| 优化目标 | 减少显存 IO                      | 减少显存碎片                 |
| 优化单位 | tile（矩阵块）                   | page/block（显存块）         |
| 内核内部 | 流式 softmax + block-wise matmul | 分页 gather + batched kernel |
| 配合关系 | Flash 在页内部做优化             | Paged 在页外部调度优化       |

二者可以完美叠加：

- PagedAttention 负责**把分散在各页的 KV 整理成逻辑连续的输入流**；
- FlashAttention 内核负责**在 tile 级别进行高速矩阵乘和 softmax 累积**。

vLLM 实际上就是：

> “PagedAttention kernel wrapper + FlashAttention inner kernel”。

------

## 🧩 5.5 PagedAttention kernel 的关键输入输出

输入：

- `page_table`: 每个 sequence 的 block_id 映射表；
- `block_k_ptrs / block_v_ptrs`: 各页在 GPU KV Pool 中的起始地址；
- `q`: 当前步 query；
- `block_valid_lengths`: 每页有效 token 数；
- `attention_mask`: 可选（因果或padding）。

输出：

- `attn_out`: 对应 query 的注意力输出向量。

------

## 🧠 5.6 分页化 gather 的并行策略

### 线程粒度

- 每个 warp 处理一个 head 的一部分；
- 每个 CTA（thread block）处理若干序列的若干 block；
- Warp 内部做 tile-level 加载与 softmax。

### 优化技巧

1. **页内连续访问**：每个 block 的 token 连续，便于 coalesced memory access。
2. **页间批量 gather**：预先生成 index 数组，减少随机访存。
3. **共享内存缓冲**：tile K/V 临时缓存。
4. **可重用 softmax 中间态**：用于 beam search 或 speculative decode。

------

## ⚡️ 5.7 性能瓶颈与优化方向

| 潜在瓶颈         | 优化策略                                  |
| ---------------- | ----------------------------------------- |
| gather 随机访问  | block_size 固定、index list 批量加载      |
| kernel 启动开销  | 采用 persistent kernel（常驻线程）        |
| 多序列混合 batch | 同步对齐 page 数量、grouped kernel launch |
| 内存对齐         | block_stride 设为 16/32 对齐字节          |
| 量化 KV          | 降低传输带宽（FP16→FP8/INT8）             |

------

## 🧮 5.8 伪代码示意（GPU 内核）

```cpp
__global__ void paged_attention_kernel(
    const float* q,
    const float* k_pool,
    const float* v_pool,
    const int* page_table,
    const int* seq_lens,
    float* out
) {
    int seq_id = blockIdx.x;
    int head_id = threadIdx.y;
    int layer_id = blockIdx.y;

    // gather per-sequence blocks
    for (int i = 0; i < seq_lens[seq_id]; ++i) {
        int block_id = page_table[seq_id * MAX_BLOCKS + i];
        const float* K = k_pool + block_id * BLOCK_STRIDE;
        const float* V = v_pool + block_id * BLOCK_STRIDE;
        // compute attention(q, K, V)
        flashattention_tile(q, K, V, out);
    }
}
```

> 实际实现中是 C++ + CUDA kernel，融合了 FlashAttention 的算子逻辑。

------

## 📈 5.9 性能对比（vLLM Benchmark）

| 模型                 | 传统 KVCache | Paged + Flash | 吞吐提升 |
| -------------------- | ------------ | ------------- | -------- |
| Llama2-7B (8K ctx)   | ~70 tok/s    | ~210 tok/s    | 3×       |
| Llama2-13B (32K ctx) | OOM / fail   | ~150 tok/s    | ✅        |
| 多请求并发 8         | 不稳定       | 平稳并行      | ✅        |

------

## ✅ 小结

| 关键点                    | 说明                                  |
| ------------------------- | ------------------------------------- |
| **PagedAttention kernel** | 通过页表批量 gather K/V               |
| **固定 block_size**       | 保证页内连续、GPU 访存高效            |
| **FlashAttention 结合**   | tile 级别流式 softmax，与页级布局协同 |
| **性能核心**              | 减少随机访存 + 批量索引 + 常驻 kernel |
| **本质**                  | “虚拟内存 + tile kernel”的 GPU 化实现 |

------

下一章（第 6 章）我们会从系统层面看：

> 🧠 **并发与调度策略** ——
>  PagedAttention 如何配合调度器（scheduler）实现动态 batching、上下文合并（prefix merging）、并发隔离等能力。

是否继续进入第 6 章？