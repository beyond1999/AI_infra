太棒了🔥
 那我们进入 **第 15 章：从零实现一个简化版 PagedAttention Mini Demo**。
 这一章是整个系列的“实战压轴篇”——你将亲手构建一个能在 PyTorch 中跑起来的 **PagedAttention 原型系统**，完整覆盖页表、页池、写入与 gather 流程。

------

# 🧱 第 15 章：手写简化版 PagedAttention Demo

------

## 🎯 目标

我们将用 **不到 150 行代码** 实现以下功能：

1. 建立固定大小的显存页池；
2. 分配 / 回收页；
3. 将 token embedding 写入页；
4. 通过页表索引进行 Attention；
5. 模拟多个序列共享前缀、动态扩展。

------

## 📦 15.1 Demo 结构概览

```
paged_demo/
 ├── allocator.py      # 页分配器
 ├── kv_cache.py       # KV 存储
 ├── paged_attn.py     # 核心注意力逻辑
 └── run_demo.py       # 运行入口
```

------

## 🧩 15.2 BlockAllocator（页分配器）

```python
# allocator.py
import torch

class BlockAllocator:
    def __init__(self, num_blocks):
        self.free_blocks = list(range(num_blocks))
        self.refcount = [0] * num_blocks

    def alloc(self, n=1):
        blocks = [self.free_blocks.pop() for _ in range(n)]
        for b in blocks:
            self.refcount[b] = 1
        return blocks

    def free(self, blocks):
        for b in blocks:
            self.refcount[b] = 0
            self.free_blocks.append(b)

    def inc_ref(self, b):
        self.refcount[b] += 1

    def dec_ref(self, b):
        self.refcount[b] -= 1
        if self.refcount[b] == 0:
            self.free_blocks.append(b)
```

------

## 🧠 15.3 PagedKVCache（KV 缓存池）

```python
# kv_cache.py
import torch

class PagedKVCache:
    def __init__(self, num_blocks, block_size, head_dim):
        self.block_size = block_size
        self.key_cache = torch.zeros(num_blocks, block_size, head_dim, device='cuda')
        self.val_cache = torch.zeros(num_blocks, block_size, head_dim, device='cuda')

    def write(self, block_id, offset, key, val):
        self.key_cache[block_id, offset] = key
        self.val_cache[block_id, offset] = val

    def gather(self, block_ids, valid_lengths):
        keys, vals = [], []
        for b, l in zip(block_ids, valid_lengths):
            keys.append(self.key_cache[b, :l])
            vals.append(self.val_cache[b, :l])
        return torch.cat(keys, dim=0), torch.cat(vals, dim=0)
```

------

## ⚙️ 15.4 简化版 PagedAttention Forward

```python
# paged_attn.py
import torch
import torch.nn.functional as F

def paged_attention(query, key_cache, val_cache, mask=None):
    attn_scores = torch.matmul(query, key_cache.T) / (key_cache.size(-1) ** 0.5)
    if mask is not None:
        attn_scores += mask
    attn_weights = F.softmax(attn_scores, dim=-1)
    return attn_weights @ val_cache
```

------

## 🧪 15.5 运行入口：run_demo.py

```python
# run_demo.py
import torch
from allocator import BlockAllocator
from kv_cache import PagedKVCache
from paged_attn import paged_attention

BLOCK_SIZE = 4
NUM_BLOCKS = 8
HEAD_DIM = 16

alloc = BlockAllocator(NUM_BLOCKS)
cache = PagedKVCache(NUM_BLOCKS, BLOCK_SIZE, HEAD_DIM)

# 模拟两个序列共享前缀
seq1_blocks = alloc.alloc(2)  # blk0, blk1
seq2_blocks = seq1_blocks[:]  # 共享前缀
alloc.inc_ref(seq1_blocks[0])
alloc.inc_ref(seq1_blocks[1])

# 各自扩展新页
seq1_blocks += alloc.alloc(1)
seq2_blocks += alloc.alloc(1)

# 写入数据
for blk in seq1_blocks + seq2_blocks:
    for i in range(BLOCK_SIZE):
        k = torch.randn(HEAD_DIM, device='cuda')
        v = torch.randn(HEAD_DIM, device='cuda')
        cache.write(blk, i, k, v)

# 构造 query 并执行 paged attention
q = torch.randn(HEAD_DIM, device='cuda')
k_cat, v_cat = cache.gather(seq1_blocks, [BLOCK_SIZE]*len(seq1_blocks))
out = paged_attention(q, k_cat, v_cat)
print("Attention output shape:", out.shape)
```

输出示例：

```
Attention output shape: torch.Size([16])
```

------

## 💡 15.6 Demo 解读与扩展

| 功能          | 对应代码             | 启发                       |
| ------------- | -------------------- | -------------------------- |
| 页分配 / 释放 | `BlockAllocator`     | 模拟内存池                 |
| 共享前缀      | `inc_ref`            | 模拟 refcount 机制         |
| 分块写入      | `PagedKVCache.write` | 模拟 prefill               |
| 拼接读取      | `gather()`           | 模拟 decode 阶段           |
| Softmax 计算  | `paged_attention()`  | 模拟 FlashAttention 简化版 |

------

## 🧩 15.7 进阶扩展方向

1. **加上 Copy-On-Write**

   ```python
   if refcount[blk] > 1:
       new_blk = alloc.alloc(1)[0]
       cache.copy_block(blk, new_blk)
   ```

2. **支持 Offload**

   - 使用 `torch.pin_memory()` 创建 CPU 缓冲；
   - 模拟页迁移（GPU ↔ CPU）。

3. **添加 Sliding Window**

   - 限定 gather 只取最近 N 个 block。

4. **多 Query 并发**

   - 在 batch 维度上并行 paged_attention。

------

## ✅ 15.8 小结

| 模块              | 功能          | 对应真实系统   |
| ----------------- | ------------- | -------------- |
| `BlockAllocator`  | 分页显存分配  | GPU 内存管理器 |
| `PagedKVCache`    | 统一 KV 存储  | 显存页池       |
| `paged_attention` | 注意力 kernel | CUDA Kernel    |
| `run_demo`        | 流程调度      | 推理 Engine    |

> **一句话总结：**
>  这个 mini demo 就是 vLLM PagedAttention 的 1:100 缩影 ——
>  它用极简逻辑模拟了「页分配 + 前缀复用 + 动态 gather + 注意力计算」的全流程。

------

下一章（第 16 章）我们将作为 **总结篇**：

> 🧭 回顾整个 PagedAttention 架构的核心思想、工程价值与未来方向，
>  以及如何从中迁移到其他高性能系统（KV 规划、RAG 缓存、MoE 调度等）。

是否继续进入第 16 章（终章）？