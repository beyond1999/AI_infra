太棒了👏
 那我们正式进入 **第 4 章：PagedAttention 在推理流程中的分页访问**。
 这章我们从运行时视角出发，看整个「从输入到输出」的时间线中——PagedAttention 是如何动态分配、写入、读取、共享和回收 KV 页块的。

------

# 🚀 第 4 章：PagedAttention 在推理流程中的运行机制

PagedAttention 的设计目标是：

> **在 LLM 推理过程中实现高并发、动态扩展、最小显存占用的 KV 管理。**

整个流程可以分为三大阶段：

1. **Prefill 阶段**：批量处理 prompt
2. **Decode 阶段**：逐步生成新 token
3. **分支与共享阶段**：如 beam search、speculative decoding

------

## 🧩 4.1 Prefill 阶段：批量分配与顺序写入

**Prefill = 模型读取 prompt 并生成首个 KV Cache。**

### 流程图：

```
用户输入 → Tokenize → 模型前向计算 → 生成 K/V → 写入 BlockPool
```

### 细节：

1. **计算所需 block 数**

   ```python
   num_blocks = ceil(prompt_len / block_size)
   ```

2. **从 BlockPool 分配**

   ```python
   blocks = block_allocator.allocate(num_blocks)
   page_table[seq_id] = blocks
   ```

3. **写入 KV Cache**

   - 每个 block 顺序写入 `block_size` 个 token 的 KV 向量；
   - 多个请求可并行 prefill；
   - 通过“批处理合并（batch prefill）”提升 GPU 利用率。

### 核心特征：

- 一次性写入，访问模式顺序；
- 页表写满一批；
- Prefill 结束时，所有请求都有初始的 page_table。

------

## 🔄 4.2 Decode 阶段：逐步扩展与随机访问

**Decode = 模型逐步生成新 token。**

对于每一步解码：

1. 模型读取历史所有 K/V（通过页表间接索引）；
2. 计算下一个 token 的 K/V；
3. 写入当前序列的最后一个 block；
4. 若该 block 已满 → 分配新 block。

伪代码：

```python
def decode_step(seq_id, new_token):
    last_blk = page_table[seq_id][-1]
    if block_is_full(last_blk):
        new_blk = block_allocator.allocate(1)
        page_table[seq_id].append(new_blk)
    write_token_kv(seq_id, new_token)
```

### 内部工作机制：

- **读路径**：通过 page_table gather 全部 K/V block；
- **写路径**：只写最后一个 block 的末尾；
- **kernel 批量 gather**：多序列一起做注意力计算；
- **动态 batching**：调度器可动态加入新请求或移除已完成请求。

> 💡 动态批处理（Dynamic Batching）是 vLLM 性能爆炸的关键：
>  不同阶段的请求都能共享 GPU batch，一切靠 PagedAttention 的页表解耦实现。

------

## 🧭 4.3 注意力计算中的分页访问（PagedAttention kernel）

假设模型正在为多个序列解码下一 token：

```
Batch = [SeqA, SeqB, SeqC]
```

每个序列的 page_table：

```
SeqA: [12, 45, 78]
SeqB: [13, 46]
SeqC: [14, 47, 79]
```

Kernel 会：

1. 构建全局 gather 索引：

   ```
   all_block_ids = [12,45,78,13,46,14,47,79]
   ```

2. 从全局 KV Pool 批量 gather 对应的 block；

3. 对每个序列执行 attention：

   ```
   q @ k.T → softmax → value加权
   ```

4. 结果按序列还原。

> 🧮 FlashAttention + PagedAttention =
>  在 tile 级别上优化访存（Flash） + 在页级别上优化布局（Paged）。

------

## 🌱 4.4 Beam Search：前缀共享与写时复制

在 Beam Search 中，多个候选序列往往共享同一段前缀：

```
         ┌─> SeqA1
SeqA ────┤
         └─> SeqA2
```

传统做法：每个 beam 都复制整份 KV Cache（高昂）。
 PagedAttention 的改进：

- 所有 beam 共享同一页表；
- 分叉时只复制需要写入的新页。

示意：

```
Before fork:
SeqA: [blk#12][blk#45][blk#78]
RefCount: {12:1, 45:1, 78:1}

After fork:
SeqA1: [12][45][78]
SeqA2: [12][45][78]
RefCount: {12:2, 45:2, 78:2}
```

当 SeqA2 生成下一个 token：

- 若 block#78 仍共享，则执行 COW：

  ```
  new_blk = alloc()
  copy_block(new_blk, blk#78)
  SeqA2.page_table[-1] = new_blk
  refcount[78] -= 1
  refcount[new_blk] = 1
  ```

结果：显存几乎按分叉宽度“线性放大”一小点，而不是全复制。

------

## 🧮 4.5 Speculative Decoding：页表回滚与重用

Speculative Decoding 是一种并行推测生成策略：
 先由一个“小模型”快速生成一批 token，然后大模型验证其中部分。
 这会导致“生成-撤销-继续生成”的来回。

PagedAttention 支持：

- 临时分配页；
- 若撤销则直接回收这些页；
- 页表可快速回滚（undo）。

示意：

```
Step 1: 小模型生成 16 tokens
        分配 block#90, #91

Step 2: 大模型验证前 8 个 token 合法 → 保留
        后 8 个丢弃 → 回收 block#91
```

这种结构的灵活性离不开固定大小 block + 页表映射机制。

------

## 🧹 4.6 序列结束与回收

当生成结束（或被取消）：

1. 模型标记该序列结束；
2. scheduler 调用 block_allocator.free(block_ids)；
3. 所有页立即回收进 free_list；
4. 页表清空。

显存立刻可复用，无需等待 compaction。

------

## 🧭 4.7 全流程时间线概览

```
┌────────────┬───────────────┬──────────────┬────────────┐
│ 阶段       │ 操作           │ 页表变化      │ Block池变化 │
├────────────┼───────────────┼──────────────┼────────────┤
│ Prefill    │ 一次分配N页     │ 写满一批       │ 减少N页      │
│ Decode     │ 逐token写入     │ 追加/替换页    │ 小量波动     │
│ Fork       │ 前缀共享/COW    │ 新页映射更新   │ 分配少量页    │
│ Speculative│ 回滚/撤销页     │ 截断页表       │ 回收页        │
│ Free       │ 释放全部页      │ 清空映射       │ 归还页        │
└────────────┴───────────────┴──────────────┴────────────┘
```

------

## ✅ 小结

| 阶段        | 核心操作             | PagedAttention 带来的收益 |
| ----------- | -------------------- | ------------------------- |
| Prefill     | 批量分配、顺序写入   | 避免碎片、高速初始化      |
| Decode      | 逐步扩展、动态 batch | 高并发吞吐、O(1) 扩展     |
| Beam Search | COW                  | 前缀共享、省显存          |
| Speculative | 页表回滚             | 快速撤销/重用             |
| Free        | 即时回收             | GPU 利用率最大化          |

------

接下来在 **第 5 章**，我们就会进入最核心的部分：

> 🧮 **PagedAttention Kernel 的实现原理**
>  ——也就是它在 GPU 上如何通过「分页化 gather」实现注意力计算，
>  如何与 FlashAttention 协同（tile IO vs page IO）。

是否继续进入第 5 章？