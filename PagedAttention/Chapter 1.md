非常好 👍
 那我们进入 **第 1 章：PagedAttention 的核心理念与内存抽象**。

------

# 🧱 第 1 章：PagedAttention 的核心理念

PagedAttention 的关键创新点在于：
 **把每个序列的 KV Cache 从“连续数组”抽象为“分页（block）结构”**，
 像操作系统那样，通过页表（page table）实现逻辑序列与物理显存的映射。

------

## 1.1 基本思想：逻辑序列与物理块解耦

设想传统 KV Cache：

```text
Request A: [token1 token2 token3 token4 token5 ... tokenN]
```

连续的一大块内存：

```
[------------------------- N tokens -------------------------]
```

PagedAttention 的做法是：

```
Request A: [block0][block1][block2][block3]...
每个 block 存固定数量的 token（例如 16 或 32）
```

内存池中每个 block 就像一个“页”：

```
Block Pool:
| Block#0 | Block#1 | Block#2 | Block#3 | Block#4 | Block#5 | ...
```

而每个请求维护自己的 **页表（page table）**：

```
Page Table (for Request A):
logical_seq_block_id -> physical_block_id

0 -> 12
1 -> 45
2 -> 78
```

这样：

- **逻辑顺序**由页表决定；
- **物理分配**可以随意复用；
- **不同请求共享一个大显存池**，按页分配。

------

## 1.2 Block 粒度的选择

block_size 是关键超参。

| block_size       | 优点                | 缺点                  |
| ---------------- | ------------------- | --------------------- |
| 小（如16token）  | 灵活，碎片少        | gather 多，访存开销大 |
| 大（如128token） | 高吞吐，locality 好 | 碎片多，复用难        |

👉 vLLM 默认 **block_size=16**，经验上较优：
 兼顾灵活性与 GPU 访存效率。

------

## 1.3 三个核心结构体

我们可以抽象出三个核心组件：

### ① BlockPool

全局显存池，维护所有空闲与已分配的块。

```python
class BlockPool:
    def __init__(self, num_blocks):
        self.free_blocks = set(range(num_blocks))
    def alloc_block(self):
        return self.free_blocks.pop()
    def free_block(self, block_id):
        self.free_blocks.add(block_id)
```

### ② PageTable

每个请求维护自己的页表，记录逻辑序列到物理 block 的映射。

```python
class PageTable:
    def __init__(self):
        self.entries = []  # list of block_ids
    def add_block(self, block_id):
        self.entries.append(block_id)
    def get_block(self, block_index):
        return self.entries[block_index]
```

### ③ KVStore / KVCache

负责存储实际的 KV 数据，每个 block 对应固定形状的 tensor：

```python
KV: [num_layers, num_heads, num_blocks, block_size, head_dim]
```

------

## 1.4 分配流程（Prefill 阶段）

当用户输入 prompt：

1. 计算需要多少个 block
    `num_blocks = ceil(prompt_length / block_size)`
2. 从 BlockPool 中分配这些 block
3. 写入 KV 数据
4. PageTable 记录这些映射

伪代码：

```python
def prefill_sequence(seq_id, tokens):
    num_blocks = ceil(len(tokens) / BLOCK_SIZE)
    for i in range(num_blocks):
        b = block_pool.alloc_block()
        page_table[seq_id].add_block(b)
        write_kv_to_block(b, tokens[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE])
```

------

## 1.5 Decode 阶段：追加写入

生成阶段，每步只写 1 个新 token：

- 定位该序列的最后一个 block；
- 若该 block 未满 → 直接写；
- 若已满 → 从池中再分配一个新 block。

这样避免了“整块复制”的麻烦。

------

## 1.6 Page Table 可视化

```
Logical Sequence (token index)
 |--------------------------------------------------------------|
 0-15     16-31     32-47     48-63 ...
   ↓         ↓         ↓         ↓
PageTable:
 [  12 ,     45 ,     78 ,     91  ]
   ↓         ↓         ↓         ↓
Physical Blocks in GPU Memory
| blk#12 | blk#45 | blk#78 | blk#91 | ...
```

------

## 1.7 带来的优势

✅ 动态扩展：
 可随时增加页，无需重新分配大块显存。

✅ 显存复用：
 不同请求从共享池中分配页，空闲页可即时复用。

✅ 前缀共享（下一章展开）：
 多个序列共享相同的前缀页，节省显存。

------

## ✅ 小结

| 关键词         | 解释                                     |
| -------------- | ---------------------------------------- |
| **BlockPool**  | 全局页池，统一显存管理                   |
| **PageTable**  | 每个序列的页表                           |
| **Block Size** | 固定粒度的 token 分配单位                |
| **解耦**       | 逻辑序列与物理存储完全独立               |
| **收益**       | 动态扩展、减少碎片、支持前缀共享与多并发 |

------

如果你想，我们下一章（第 2 章）就讲：

> 🔹 数据布局（KV tensor 排布）
>  🔹 索引路径（token idx → block idx → 物理地址）
>  🔹 为什么这种布局能兼容 GPU 高效 gather。

是否继续进入第 2 章？