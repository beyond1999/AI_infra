太好了 👍
 那我们正式进入 **第 3 章：PagedAttention 的分配器与内存管理机制**。
 这章是整个系统的“灵魂”部分：PagedAttention 如何做到既快又省显存，靠的就是一个**极简高效的 Block Allocator（分页分配器）**。

------

# 💾 第 3 章：Block Allocator 与内存管理

------

## 🧠 3.1 为什么要自己写分配器？

GPU 显存不像 CPU 内存那样支持 malloc/free 等复杂机制。
 如果你直接在 CUDA 里频繁 `cudaMalloc`，性能会炸裂（几十微秒起跳）。

PagedAttention 必须自己管理显存池：

- 提前一次性分配好一整块大显存；
- 在这块显存中自己“切片”出若干固定大小的块（block）；
- 自己记录哪些块被占用、哪些是空闲的。

因此它实现了一个**轻量级固定大小分配器**，大概结构如下👇

------

## 🧩 3.2 内存池总体结构

```
┌────────────────────────────────────────────────────────┐
│                     Block Pool (GPU)                  │
├────────┬────────┬────────┬────────┬────────┬────────┤
│ Block0 │ Block1 │ Block2 │ Block3 │ Block4 │ Block5 │ ...
├────────┴────────┴────────┴────────┴────────┴────────┤
│ 每个 Block 的大小：layers × heads × block_size × head_dim │
└────────────────────────────────────────────────────────┘
```

配套的元数据结构：

```python
class BlockManager:
    def __init__(self, num_blocks):
        self.free_blocks = set(range(num_blocks))  # 空闲块列表
        self.active_blocks = {}  # seq_id -> list of block_ids
```

------

## ⚙️ 3.3 分配（alloc）流程

### 场景：Prefill 阶段新建请求

当用户输入一个新序列：

1. 根据 token 长度计算所需 block 数；
2. 从 free_blocks 中取出对应数量；
3. 绑定到 page_table；
4. 更新 active_blocks。

伪代码：

```python
def allocate_sequence(seq_id, num_tokens):
    num_blocks = ceil(num_tokens / BLOCK_SIZE)
    allocated = []
    for _ in range(num_blocks):
        if not free_blocks:
            raise OOMError()
        blk = free_blocks.pop()
        allocated.append(blk)
    active_blocks[seq_id] = allocated
    return allocated
```

------

## 🔄 3.4 回收（free）流程

### 场景：序列结束或 beam search 分支被淘汰

1. 找到该序列在 active_blocks 中的所有 block；
2. 释放回 free_blocks 集合；
3. 删除其 page_table 记录。

伪代码：

```python
def free_sequence(seq_id):
    for blk in active_blocks[seq_id]:
        free_blocks.add(blk)
    del active_blocks[seq_id]
```

------

## ♻️ 3.5 复用机制：动态扩展 + 即时回收

- **动态扩展**：当序列生成新 token（decode 阶段），若当前 block 已满 → 立即分配新 block。
- **即时回收**：当请求结束（或被取消） → 立即归还 block 到池中。

> 🚀 因为 block_size 固定，管理非常简单，不需要像 malloc 那样做碎片整理（compaction）。

------

## 🧱 3.6 碎片问题与解决策略

在分页机制中：

- 每个 block 是固定大小 → **外部碎片几乎为零**；
- 最多存在少量 **内部碎片**：最后一个 block 未被完全填满。

例如：

```
block_size = 16
prompt_len = 34
→ 实际使用 3 个 block，最后一个只用 2/16 token 空间
```

这种浪费率 ≤ 1/block_size
 通常 < 5%，可以忽略。

------

## 🧩 3.7 Prefix Sharing（页共享）与写时复制（COW）

**关键优化：前缀复用**
 在多序列生成中（尤其是 Beam Search 或多用户共享同一 Prompt），不同序列会共享相同的前缀。

PagedAttention 支持：

- 让多个序列的 PageTable 指向相同的物理 block；
- 当其中某个序列写入新 token 时，再复制（Copy-On-Write）。

示意图：

```
SeqA: [blk#12][blk#45][blk#78]
SeqB: [blk#12][blk#45][blk#79]   ← 前两块共享，最后分叉
```

逻辑：

```python
def append_token(seq_id, token):
    last_blk = page_table[seq_id][-1]
    if is_full(last_blk):
        # 若写入会修改共享块，执行 COW
        if refcount[last_blk] > 1:
            new_blk = alloc_block()
            copy_block(new_blk, last_blk)
            page_table[seq_id][-1] = new_blk
            refcount[last_blk] -= 1
        else:
            new_blk = alloc_block()
            page_table[seq_id].append(new_blk)
```

这样能节省大量显存，同时保持正确性。

------

## 📊 3.8 分配与回收示意图

```
Step 1: 全局空闲池
  free_blocks = {0,1,2,3,4,5,...}

Step 2: 用户 A 请求分配 3 块
  A -> [0,1,2]
  free_blocks = {3,4,5,...}

Step 3: 用户 B 请求分配 2 块
  B -> [3,4]
  free_blocks = {5,6,...}

Step 4: 用户 A 生成新 token，需扩容
  alloc 1 块 → A -> [0,1,2,5]
  free_blocks = {6,7,...}

Step 5: 用户 A 结束，回收
  free_blocks = {0,1,2,5,6,7,...}
```

------

## 📈 3.9 vLLM 的实际实现

vLLM 的核心类：
 🔹 `BlockAllocator`
 🔹 `BlockTable`
 🔹 `PagedKVCache`

它在 GPU 上有一张**block_ref_count 表**，管理每个 block 的引用数（支持共享前缀）。
 主要函数：

| 函数                                | 作用             |
| ----------------------------------- | ---------------- |
| `allocate(num_blocks)`              | 分配多个空闲块   |
| `free(block_ids)`                   | 释放块           |
| `refcount_inc/dec()`                | 管理共享页引用   |
| `get_kv_ptr(layer, head, block_id)` | 返回 GPU KV 地址 |

这样整个内存生命周期就完整闭环了。

------

## ✅ 小结

| 功能 | 机制                     | 效果            |
| ---- | ------------------------ | --------------- |
| 分配 | 固定大小 block，从池中取 | O(1) 时间，稳定 |
| 回收 | 即时回收回 free list     | 无需整理        |
| 扩展 | decode 阶段按需追加      | 动态灵活        |
| 共享 | 前缀复用 + 写时复制      | 节省显存        |
| 碎片 | 仅有内部碎片             | 可忽略          |

PagedAttention 用一个简单的**固定块分配器 + 页表机制**，
 实现了传统连续 KV 难以达到的「动态性 + 并发性 + 空间复用性」。

------

下一章（第 4 章）我们讲：

> 🚀 **推理流程中的分页访问**：
>  从 Prefill 到 Decode 再到 Beam Search / Speculative Decoding，
>  分页结构是如何在运行时工作的（含时间线图和数据流）。

是否继续进入第 4 章？