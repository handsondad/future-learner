# DeepSeek V3 详解

以下是通过 DeepSeek Chat来理解其原理，我们可以基于它自身来学习。

<p align="center">
  <img src="figures/DeepSeek V3 Architecture.png">
</p>

## 关于 Transformer

```python
class Transformer(nn.Module):
    """
    Transformer 模型，包含位置嵌入、多层 transformer 块和输出投影层。

    Attributes:
        max_seq_len (int): 模型支持的最大序列长度。
        embed (nn.Module): 输入 token 的嵌入层。
        layers (torch.nn.ModuleList): 包含所有 transformer 块的模块列表。
        norm (nn.Module): 所有 transformer 块后的层归一化。
        head (nn.Module): 输出投影层，映射到词表大小。
        freqs_cis (torch.Tensor): 预计算的旋转位置嵌入的复指数值。
    """
    def __init__(self, args: ModelArgs):
        """
        初始化 Transformer 模型。

        Args:
            args (ModelArgs): 包含模型参数的 ModelArgs 对象。
        """
        global world_size, rank
        # 获取分布式环境中的 world_size 和 rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        # 根据模型参数设置线性层的精度
        Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        # 调用父类初始化函数
        super().__init__()
        # 设置最大序列长度
        self.max_seq_len = args.max_seq_len
        # 初始化嵌入层
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        # 初始化多层 transformer 块
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))
        # 初始化层归一化
        self.norm = RMSNorm(args.dim)
        # 初始化输出投影层
        self.head = ColumnParallelLinear(args.dim, args.vocab_size, dtype=torch.get_default_dtype())
        # 注册缓冲区，存储预计算的旋转位置嵌入的复指数值
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        """
        Transformer 模型的前向传播。

        Args:
            tokens (torch.Tensor): 输入 token ID 张量，形状为 (batch_size, seq_len)。
            start_pos (int, optional): 旋转位置嵌入的起始位置，默认为 0。

        Returns:
            torch.Tensor: 形状为 (batch_size, vocab_size) 的 logits 张量。
        """
        # 获取输入序列长度
        seqlen = tokens.size(1)
        # 将输入 token 转换为嵌入向量
        h = self.embed(tokens)
        # 获取从 start_pos 开始的旋转位置嵌入的复指数值
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        # 如果序列长度大于 1，生成掩码（上三角为负无穷，防止未来信息泄漏）
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
        # 逐层通过 transformer 块
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        # 对最后一层的输出进行归一化，并取最后一个位置的向量
        h = self.norm(h)[:, -1]
        # 通过输出投影层得到 logits
        logits = self.head(h)
        # 如果启用了分布式，收集所有 GPU 的 logits
        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        # 返回最终的 logits
        return logits

class Block(nn.Module):
    """
    Transformer 块，结合了注意力层和前馈网络层。

    Attributes:
        attn (nn.Module): 注意力层 (MLA)。
        ffn (nn.Module): 前馈网络 (MLP 或 MoE)。
        attn_norm (nn.Module): 注意力层的层归一化。
        ffn_norm (nn.Module): 前馈网络的层归一化。
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        初始化 Transformer 块。

        Args:
            layer_id (int): Transformer 中的层索引。
            args (ModelArgs): 包含块参数的 ModelArgs 对象。
        """
        # 调用父类初始化函数
        super().__init__()
        # 初始化注意力层 (MLA)
        self.attn = MLA(args)
        # 根据层索引选择前馈网络：前几层使用 MLP，后续层使用 MoE
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        # 初始化注意力层的层归一化
        self.attn_norm = RMSNorm(args.dim)
        # 初始化前馈网络的层归一化
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Transformer 块的前向传播。

        Args:
            x (torch.Tensor): 输入张量。
            start_pos (int): 序列中的起始位置。
            freqs_cis (torch.Tensor): 预计算的旋转位置嵌入的复指数值。
            mask (Optional[torch.Tensor]): 注意力掩码张量，用于排除某些位置。

        Returns:
            torch.Tensor: 块计算后的输出张量。
        """
        # 残差连接：先对输入进行层归一化，然后通过注意力层
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        # 残差连接：先对输入进行层归一化，然后通过前馈网络
        x = x + self.ffn(self.ffn_norm(x))
        # 返回输出张量
        return x

```

## 关于 MLA

Multi-Head Latent Attention（多头潜在注意力）**是一种改进的注意力机制，通常用于深度学习和自然语言处理（NLP）任务中。它结合了**多头注意力机制（Multi-Head Attention）**和**潜在表示（Latent Representation）**的思想，旨在更有效地捕获输入数据中的复杂依赖关系和潜在结构。

### 概念和介绍

以下是 Multi-Head Latent Attention 的核心概念和介绍：

#### **多头注意力机制（Multi-Head Attention）**
- 多头注意力是 Transformer 模型的核心组件，由 Vaswani 等人在 2017 年提出。
- 通过将输入数据映射到多个子空间（即“头”），多头注意力可以并行地捕获输入的不同特征。
- 每个头独立地计算注意力权重，然后将结果拼接在一起进行融合。

#### **潜在表示（Latent Representation）**
- 潜在表示是指数据在隐空间中的抽象表示，通常通过深度学习模型（如自编码器或生成模型）学习得到。
- 潜在表示可以捕捉数据的高层次语义特征，并减少噪声。

#### **Multi-Head Latent Attention 的核心思想**
Multi-Head Latent Attention 在传统的多头注意力机制的基础上，引入了潜在表示的思想。具体来说：
- 它通过将输入数据映射到一个潜在空间（latent space）来提取更高层次的语义特征。
- 在潜在空间中计算多头注意力，从而更有效地捕获输入数据中的全局和局部依赖关系。
- 这种方法可以提高模型对复杂模式的建模能力，同时减少计算复杂度。

#### **工作原理**
Multi-Head Latent Attention 的工作流程通常包括以下步骤：
1. **输入映射**：
   - 将输入数据（如词嵌入或特征向量）通过一个非线性变换（如全连接层或卷积层）映射到潜在空间。
2. **多头注意力计算**：
   - 在潜在空间中，将输入数据分为多个“头”，每个头独立地计算注意力权重。
   - 使用点积注意力（Scaled Dot-Product Attention）或更复杂的注意力机制计算每个头的权重。
3. **结果融合**：
   - 将所有头的输出拼接在一起，并通过一个线性变换将其映射回原始空间。
4. **输出**：
   - 输出融合后的结果，可以用于下游任务（如分类、生成或预测）。

#### **优势**
- **更好的特征提取**：通过潜在表示，能够捕捉输入数据的高层次语义特征。
- **更高的效率**：在潜在空间中计算注意力可以减少计算复杂度，尤其是在处理高维数据时。
- **更强的建模能力**：多头机制和潜在表示的结合，使得模型能够同时捕获全局和局部的依赖关系。

#### **应用场景**
Multi-Head Latent Attention 可以广泛应用于以下领域：
- **自然语言处理（NLP）**：如文本分类、机器翻译、问答系统等。
- **计算机视觉（CV）**：如图像分类、目标检测、生成对抗网络（GAN）等。
- **多模态学习**：如图文对齐、视频理解等。
- **时间序列分析**：如股票预测、语音识别等。

#### **与传统多头注意力的区别**

| **特性**         | **传统多头注意力** | **多头潜在注意力**               |
| ---------------- | ------------------ | -------------------------------- |
| **输入空间**     | 原始输入空间       | 潜在空间                         |
| **特征提取能力** | 直接捕获输入特征   | 捕获高层次语义特征               |
| **计算复杂度**   | 较高               | 可能降低（取决于潜在空间的维度） |
| **建模能力**     | 有限于原始输入空间 | 更强，适合复杂依赖关系           |

#### **相关研究**
Multi-Head Latent Attention 的灵感来自以下研究领域：
- **Transformer 模型**：Vaswani 等人在 2017 年提出的注意力机制。
- **变分自编码器（VAE）**：通过潜在表示学习数据分布。
- **生成对抗网络（GAN）**：在潜在空间中生成高质量样本。

#### 总结
Multi-Head Latent Attention 是一种结合多头注意力和潜在表示的先进注意力机制，能够更有效地建模复杂数据中的依赖关系。它在 NLP、CV 和多模态学习等领域具有广泛的应用前景，是深度学习研究中的一个重要方向。

### 代码详解

```python
class MLA(nn.Module):
    """
    Multi-Headed Attention Layer (MLA).

    Attributes:
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        n_local_heads (int): Number of local attention heads for distributed systems.
        q_lora_rank (int): Rank for low-rank query projection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        qk_head_dim (int): Total dimensionality of query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for softmax in attention computation.
    """
    def __init__(self, args: ModelArgs):
        # 调用父类的初始化方法
        super().__init__()
        
        # 从args中获取模型的维度、注意力头数等参数
        self.dim = args.dim  # 模型的维度
        self.n_heads = args.n_heads  # 总的注意力头数
        self.n_local_heads = args.n_heads // world_size  # 本地注意力头数，world_size可能是分布式训练中的进程数
        self.q_lora_rank = args.q_lora_rank  # 查询（Query）LoRA的秩
        self.kv_lora_rank = args.kv_lora_rank  # 键值（Key-Value）LoRA的秩
        self.qk_nope_head_dim = args.qk_nope_head_dim  # 不使用位置编码的注意力头的维度
        self.qk_rope_head_dim = args.qk_rope_head_dim  # 使用位置编码的注意力头的维度
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim  # 总注意力头的维度
        self.v_head_dim = args.v_head_dim  # 值（Value）头的维度

        # 根据q_lora_rank的值，选择不同的查询权重初始化方式
        if self.q_lora_rank == 0:
            # 如果q_lora_rank为0，直接使用ColumnParallelLinear初始化查询权重
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            # 否则，使用LoRA结构初始化查询权重
            self.wq_a = Linear(self.dim, self.q_lora_rank)  # LoRA的第一层
            self.q_norm = RMSNorm(self.q_lora_rank)  # 对LoRA的输出进行归一化
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)  # LoRA的第二层

        # 初始化键值（Key-Value）权重
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)  # LoRA的第一层
        self.kv_norm = RMSNorm(self.kv_lora_rank)  # 对LoRA的输出进行归一化
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))  # LoRA的第二层

        # 初始化输出权重
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)

        # 设置softmax的缩放因子
        self.softmax_scale = self.qk_head_dim ** -0.5

        # 如果最大序列长度大于原始序列长度，调整softmax的缩放因子
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        # 根据注意力实现方式的不同，初始化不同的缓存
        if attn_impl == "naive":
            # 如果是"naive"实现，初始化键和值的缓存
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)
        else:
            # 否则，初始化键值和位置编码的缓存
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        Forward pass for the Multi-Headed Attention Layer (MLA).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for caching.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        # 获取输入张量的 batch size 和序列长度
        bsz, seqlen, _ = x.size()
        # 计算结束位置
        end_pos = start_pos + seqlen

        # 根据 q_lora_rank 的值选择不同的查询（Query）计算方式
        if self.q_lora_rank == 0:
            # 如果 q_lora_rank 为 0，直接使用 wq 计算查询
            q = self.wq(x)
        else:
            # 否则，使用 LoRA 结构计算查询
            q = self.wq_b(self.q_norm(self.wq_a(x)))

        # 调整查询张量的形状，以适应多头注意力机制
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        # 将查询张量分为不使用位置编码和使用位置编码的部分
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        # 对使用位置编码的部分应用旋转嵌入
        q_pe = apply_rotary_emb(q_pe, freqs_cis)

        # 计算键值（Key-Value）张量
        kv = self.wkv_a(x)
        # 将键值张量分为 LoRA 部分和位置编码部分
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        # 对键值的位置编码部分应用旋转嵌入
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)

        # 根据注意力实现方式选择不同的计算逻辑
        if attn_impl == "naive":
            # 如果不使用 LoRA 优化，直接拼接查询的不使用位置编码部分和使用位置编码部分
            q = torch.cat([q_nope, q_pe], dim=-1)
            # 对键值张量应用归一化并通过 wkv_b 进行线性变换
            kv = self.wkv_b(self.kv_norm(kv))
            # 调整键值张量的形状，以适应多头注意力机制
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            # 将键值张量分为键的不使用位置编码部分和值部分
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            # 将键的不使用位置编码部分和位置编码部分拼接起来
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            # 更新缓存中的键和值
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v
            # 计算注意力分数，通过矩阵乘法（einsum）计算查询和键的点积，并乘以 softmax 缩放因子
            scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
        else:
            # 如果使用 LoRA 优化，获取 wkv_b 的权重并进行反量化（如果 scale 存在）
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size)
            # 调整 wkv_b 的形状，以适应多头注意力机制
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            # 计算不使用位置编码部分的查询与 wkv_b 的线性变换结果
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            # 更新缓存中的键值部分和位置编码部分
            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
            # 计算注意力分数，分别计算不使用位置编码部分和位置编码部分的分数，然后相加并乘以 softmax 缩放因子
            scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                    torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale

        # 如果提供了 mask 张量，将其添加到注意力分数中（用于屏蔽某些位置）
        if mask is not None:
            scores += mask.unsqueeze(1)

        # 对注意力分数进行 softmax 归一化，以确保分数在 0 到 1 之间，并转换为输入张量的数据类型
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)

        # 根据注意力实现方式选择不同的输出计算逻辑
        if attn_impl == "naive":
            # 如果不使用 LoRA 优化，直接通过矩阵乘法计算输出
            x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
        else:
            # 如果使用 LoRA 优化，先计算中间结果，然后通过 wkv_b 进行线性变换
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])

        # 将多头注意力的输出展平，并通过 wo 进行线性变换，得到最终输出
        x = self.wo(x.flatten(2))

        # 返回输出张量
        return x

```

这段代码定义了一个 **Multi-Headed Attention Layer (MLA)**，用于处理多注意力机制的任务。以下是对代码的总结，方便理解：

#### **核心目标**
- **MLA** 是一个多头注意力层，用于在深度神经网络中实现注意力机制，特别是在处理序列数据时。
- 它支持低秩投影（LoRA）和位置编码（Rotary Embedding），并能在分布式系统中处理局部注意力。

#### **主要功能与结构**
1. **初始化 (`__init__`)**
   - **参数配置**：根据 `ModelArgs` 配置维度和注意力头的数量。
   - **低秩投影 (`q_lora_rank`, `kv_lora_rank`)**：如果启用，使用低秩投影来降低计算复杂度。
   - **线性层**：
     - `wq`, `wkv_a`, `wkv_b`, `wo`：分别用于查询（Query）、键/值（Key/Value）和输出（Output）的线性变换。
   - **缓存机制**：根据 `attn_impl` 的不同实现方式，初始化键/值缓存（`k_cache`, `v_cache` 或 `kv_cache`）和位置编码缓存（`pe_cache`）。

2. **前向传播 (`forward`)**
   - **输入**：
     - `x`：输入张量，形状为 `(batch_size, seq_len, dim)`。
     - `start_pos`：序列中的起始位置，用于缓存。
     - `freqs_cis`：预计算的旋转位置编码（Rotary Embedding）值。
     - `mask`：可选掩码张量，用于屏蔽某些位置的注意力。
   - **核心步骤**：
     1. **查询（Query）**：通过 `wq` 或低秩投影 (`wq_a`, `wq_b`) 计算查询向量。
     2. **键/值（Key/Value）**：通过 `wkv_a` 和 `wkv_b` 计算键和值向量，并应用位置编码。
     3. **缓存更新**：将计算得到的键/值和位置编码存入缓存。
     4. **注意力分数计算**：通过点积计算注意力分数，并应用缩放因子和掩码。
     5. **输出计算**：根据注意力分数对值进行加权求和，最终通过 `wo` 线性层输出结果。

3. **实现细节**
   - **Naive 实现**：直接计算完整的键/值缓存，适用于较短序列。
   - **高效实现**：通过低秩投影和缓存机制优化计算，适用于长序列和分布式系统。

#### **关键特点**
- **低秩投影**：通过 `q_lora_rank` 和 `kv_lora_rank` 支持低秩投影，减少计算开销。
- **位置编码**：使用旋转位置编码 (`Rotary Embedding`) 增强模型对序列位置信息的捕捉能力。
- **缓存机制**：在 `forward` 过程中缓存键/值和位置编码，避免重复计算。
- **分布式支持**：通过 `n_local_heads` 支持分布式系统中的局部注意力计算。

#### **总结**
这段代码实现了一个高效的多头注意力层，支持低秩投影、位置编码和分布式计算。通过灵活的缓存机制和优化实现，它能够处理长短不一的序列数据，并适用于多种硬件和分布式环境。

## 关于 RoPE

RoPE，全称为“Rotary Positional Embedding”（旋转位置嵌入），是自然语言处理领域中，特别是在处理序列数据的Transformer模型中使用的一种创新的位置编码技术。它由Alexander H. Wu、Kaiwen Wu等人在2021年的论文《RoFormer: Enhanced Transformer with Rotary Positional Encoding for Long-Sequence Text Summarization》中首次提出。RoPE的提出是为了改进标准的固定位置编码方法，以解决长序列文本处理中的位置依赖性问题，同时减缓模型在序列长度变化时的表现下降。

### 概念和介绍

#### 核心思想
传统的Transformer模型通常使用Sinusoidal位置编码（如Bert），将位置信息嵌入到模型输入中。然而，这种方法存在局限性，尤其是在处理长度可变的输入序列时，固定的编码可能难以适应序列长度的变化，影响模型的泛化能力。

RoPE通过引入旋转操作，将位置信息以一种动态且参数化的方式嵌入到Transformer模型的注意力机制中。具体而言，RoPE对查询和键（Query和Key）向量的位置信息进行编码，然后通过矩阵旋转来更新这些向量，而不是直接加法或乘法操作。这种方法保留了绝对位置信息，同时增强了模型对相对位置的敏感性，从而提高了模型处理长序列数据的能力。

#### 运作机制
在RoPE中，每个位置的Query和Key向量首先通过相应的旋转矩阵进行编码。旋转矩阵是基于正弦和余弦函数构建的，与位置相关联，允许模型捕捉相对位置关系。在计算注意力机制时，这些旋转后的向量会参与到注意力权重的计算中，从而影响最终的注意力分布和价值（Value）向量的加权和。

#### 主要优势
- **位置不变性**：RoPE使得模型在处理不同长度的序列时能够更好地保持一致性，因为它不依赖于特定的序列长度。
- **相对位置编码**：通过更新位置信息的编码方式，RoPE增强了模型对相对位置的感知，有助于处理长距离依赖。
- **参数高效性**：与固定位置编码相比，RoPE的引入通常不需要增加额外的模型参数，有助于控制模型的规模和复杂度。
- **序列长度独立性**：RoPE的动态编码机制使得模型在处理变量长度的输入时更加灵活，无需为每一种可能的序列长度预计算位置编码。

RoPE自提出以来，已在多种NLP任务中显示出其有效性和实用性，包括文本摘要、对话系统、机器阅读理解等，为处理长序列数据提供了新的视角和解决方案。

### 代码详解

理解下RoPE (Rotary Positional Embedding)的代码实现：
```python
def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)
```

这段代码实现了RoPE (Rotary Positional Embedding)在给定输入张量上的应用，它是处理序列数据时用于增强模型对相对位置的理解的一种技术，特别在Transformer架构中常见。下面是对代码的逐行详细解释：

```python
def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
```
定义了函数 `apply_rotary_emb`，它接受两个参数：张量 `x`（输入序列），以及 `freqs_cis`（用于旋转的位置编码张量）。

```python
dtype = x.dtype
```
存储了输入张量 `x` 的原始数据类型，这通常是为了在转换类型进行计算后，能将结果转换回原始类型，确保与模型的其他部分兼容。

```python
x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
```
这一行代码做了几件事：
1. 首先将张量 `x` 转换为浮点数类型，这是因为复数张量在PyTorch中需要以浮点数形式操作。
2. 然后，对 `x` 的形状进行重塑，将最后一维的元素通过 `-1, 2` 重塑为复数的实部和虚部。这相当于将最后一维的一对元素解释为一个复数，即最后一维的每个元素被拆分为实部和虚部。
3. 最终，使用 `torch.view_as_complex` 将重塑的张量转换为复数张量。

```python
freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
```
这里，`freqs_cis` 张量被重塑，以匹配张量 `x` 的各个维度。这是为了后续的元素级乘法做准备，确保维度对齐。`freqs_cis` 张量的形状调整，使得它可以在批量处理和头数维度上进行广播。

```python
y = torch.view_as_real(x * freqs_cis).flatten(3)
```
执行旋转操作：
1. `x * freqs_cis` 在复数域中进行元素级乘法。由于 `x` 和 `freqs_cis` 都被视为复数张量，这实质上是进行了复数乘法，实现了旋转操作。
2. 然后，使用 `torch.view_as_real` 将乘法结果，即复数张量转换回实数张量表示。
3. 最后，使用 `.flatten(3)` 将第四维及之后的维度展平，恢复原始的张量结构，除了将旋转操作中最后两个维度的“2”及之前由旋转引入的维度回缩到原本最后一维的位置。这样，结果张量的形状将与输入张量 `x` 相匹配，只除了在旋转操作中被“压缩”再“扩展”的维度部分。

```python
return y.to(dtype)
```
在进行完所有计算后，结果张量 `y` 被转换回与输入张量 `x` 相同的数据类型，然后返回。

总之，这段代码的核心目的是通过一个旋转操作应用位置编码。这一旋转操作有助于模型在处理长序列或需要对相对位置敏感的任务时，更好地捕获序列中元素间的依赖关系。

### 理解下复数乘法的数学本质

在解释为什么`x * freqs_cis`在复数域中的元素级乘法能实现旋转操作之前，我们首先需要理解复数乘法的数学本质以及其如何与旋转操作关联。

#### 复数乘法的本质
复数乘法的本质在于它不仅改变了复数的“长度”（模），还改变了它的“角度”（幅角）。数学上，如果两个复数分别表示为
$$z_1 = r_1(\cos\theta_1 + i\sin\theta_1)$$
和
$$z_2 = r_2(\cos\theta_2 + i\sin\theta_2)$$
其中$r_1$和$r_2$是它们的模，$\theta_1$和$\theta_2$是它们的幅角，那么它们的乘积$z_1z_2$可以表示为
$$
z_1z_2 = r_1r_2(\cos(\theta_1+\theta_2) + i\sin(\theta_1+\theta_2))
$$
这表明复数乘法实际上结合了两者的“长度”乘积（模的乘积）和两个复数的角度相加。

#### 旋转操作与复数乘法
在二维平面上，一个复数可以被视作一个向量，它的实部和虚部分别对应于向量在x轴和y轴上的分量。从几何角度来看，复数的幅角则代表了这个向量与正x轴的夹角。因此，当我们乘以一个模为1（即“长度”为1）的复数时，实际上是在原地旋转这个向量而不改变其长度。

#### RoPE中的旋转操作
在RoPE中使用的`freqs_cis`实际上是基于位置的复数值，这些值是周期函数（余弦和正弦）的输出，构造为
$$
freqs_{cis}[t] = \cos(\omega t) + i\sin(\omega t)
$$
其中，$t$是位置索引，而$\omega$是与频率相关的参数。由于$freqs_{cis}$的模为1，当其与复数表示的向量（即变换后的`x`）相乘时，实际上是在对向量进行旋转。

#### RoPE中的应用
当输入张量`x`通过某些方法转换为复数表示，并与`freqs_cis`进行元素级乘法时，每一位置上的向量都会经历一个基于位置的独立旋转。这种旋转保留了向量的模不变，同时调整了它的方向，反映了时间（位置）信息。由于旋转的角度与位置索引直接相关，这种操作本质上就编码了位置信息到查询（Query）和键（Key）向量中，有助于模型学习序列中不同位置元素之间的相对关系。

因此，通过复数乘法，`x * freqs_cis`实际上实现了基于位置的旋转操作，这是一种非常巧妙的方法来显式地将位置信息嵌入到Transformer的注意力机制中，特别适合处理如长序列文本或语音数据等任务。

## 关于 DeepSeekMoE

### **DeepSeekMoE 简介**

**DeepSeekMoE** 是一种基于 **Mixture of Experts (MoE)** 的深度学习架构，专门设计用于高效处理大规模任务，尤其是在计算资源有限的情况下。MoE 的核心思想是将模型分解为多个专家（Experts），每个专家负责处理特定的子任务，而门控机制（Gating Network）则动态地决定如何组合这些专家的输出。DeepSeekMoE 结合了 MoE 的优势，同时通过一系列优化技术提高了模型的效率和性能。

<p align="center">
  <img src="figures/MoE Layer.png">
</p>

#### **核心设计理念**
1. **Mixture of Experts (MoE)**:
   - 模型由多个专家组成，每个专家是一个小的神经网络。
   - 门控机制根据输入动态分配权重，决定哪些专家参与计算。

2. **动态路由（Dynamic Routing）**:
   - 输入数据通过门控网络分配，只有部分专家被激活，从而减少计算量。
   - 这种稀疏激活机制使得模型能够扩展到更大的规模，同时保持高效性。

3. **高效训练与推理**:
   - 通过分布式训练和参数共享技术，优化大规模模型的训练效率。
   - 支持低资源环境下的推理，例如边缘计算设备。

#### **关键技术特点**
1. **稀疏激活**:
   - 只有少数专家被激活，减少计算开销。
   - 通过门控机制实现动态选择，适应不同的输入数据。

2. **参数共享与专家专业化**:
   - 专家之间可以共享部分参数，减少模型的总体参数量。
   - 每个专家专注于特定的任务或特征，提升模型的表达能力。

3. **分布式训练**:
   - 支持多 GPU 或多节点训练，适应大规模模型的需求。
   - 通过高效的通信机制减少分布式训练的开销。

4. **自适应门控**:
   - 门控网络能够根据输入数据的特性动态调整专家的权重。
   - 支持多种门控策略，例如 Top-K 选择或软注意力机制。

5. **资源优化**:
   - 针对低资源环境（如移动设备或嵌入式设备）进行优化，降低内存和计算需求。

#### **优势**
1. **高效性**:
   - 稀疏激活和动态路由技术显著减少计算开销。
2. **扩展性**:
   - 支持大规模模型的训练与部署。
3. **灵活性**:
   - 适用于多种任务和领域，具有广泛的应用潜力。
4. **资源节约**:
   - 在低资源环境下仍能高效运行，降低硬件需求。

#### **总结**
DeepSeekMoE 是一种创新的基于 MoE 的深度学习架构，通过稀疏激活、动态路由和分布式训练等技术，实现了高效、灵活和可扩展的模型设计。它特别适合处理大规模任务和低资源环境，在 NLP、CV 和推荐系统等领域具有广泛的应用前景。

### 代码详解

```python
class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.dim = args.dim  # 输入特征的维度
        # 确保路由专家的数量可以被 world_size 整除
        assert args.n_routed_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"
        self.n_routed_experts = args.n_routed_experts  # 路由专家的总数
        self.n_local_experts = args.n_routed_experts // world_size  # 每个设备上的本地专家数量
        self.n_activated_experts = args.n_activated_experts  # 每个样本激活的专家数量
        self.experts_start_idx = rank * self.n_local_experts  # 当前设备上专家的起始索引
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts  # 当前设备上专家的结束索引
        self.gate = Gate(args)  # 门控模块，用于决定输入分配给哪些专家
        # 初始化专家列表，仅当前设备上的专家被实例化
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)  # 共享专家模块

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        shape = x.size()  # 保存输入张量的原始形状
        x = x.view(-1, self.dim)  # 将输入张量展平为二维张量，便于处理
        weights, indices = self.gate(x)  # 通过门控模块获取权重和路由索引
        y = torch.zeros_like(x)  # 初始化输出张量
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()  # 统计每个专家被分配到的样本数量
        # 遍历当前设备上的专家，计算输出
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:  # 如果当前专家未被分配到任何样本，则跳过
                continue
            expert = self.experts[i]  # 获取当前专家
            idx, top = torch.where(indices == i)  # 获取分配到当前专家的样本索引
            y[idx] += expert(x[idx]) * weights[idx, top, None]  # 计算当前专家的输出并加权累加
        z = self.shared_experts(x)  # 计算共享专家的输出
        if world_size > 1:
            dist.all_reduce(y)  # 如果使用多设备，则对输出进行全局求和
        return (y + z).view(shape)  # 合并专家输出并将输出恢复为原始形状并返回

class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = Linear(dim, inter_dim)  # 输入到隐藏层的线性变换
        self.w2 = Linear(inter_dim, dim)  # 隐藏层到输出层的线性变换
        self.w3 = Linear(dim, inter_dim)  # 输入到隐藏层的额外线性变换

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        # 计算 w1(x) 并使用 SiLU 激活函数【x * torch.sigmoid(x)】(替换ReLU)，然后与 w3(x) 逐元素相乘
        hidden = F.silu(self.w1(x)) * self.w3(x)
        # 将结果通过 w2 线性变换得到最终输出
        return self.w2(hidden)

class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the Gate module.

        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__()
        self.dim = args.dim  # 输入特征的维度
        self.topk = args.n_activated_experts  # 每个输入激活的专家数量
        self.n_groups = args.n_expert_groups  # 路由分组的数量
        self.topk_groups = args.n_limited_groups  # 每个输入路由到的分组数量
        self.score_func = args.score_func  # 评分函数类型（'softmax' 或 'sigmoid'）
        self.route_scale = args.route_scale  # 路由权重的缩放因子
        # 可学习的权重矩阵，形状为 (n_routed_experts, dim)
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        # 可选的偏置项，仅在特定条件下使用
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts)) if self.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 路由权重和选中的专家索引。
        """
        # 计算输入 x 与权重矩阵的线性变换得分
        scores = linear(x, self.weight)
        # 根据评分函数类型对得分进行归一化
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)  # 使用 softmax 归一化
        else:
            scores = scores.sigmoid()  # 使用 sigmoid 归一化
        original_scores = scores  # 保存原始的得分
        # 如果存在偏置项，将其加到得分上
        if self.bias is not None:
            scores = scores + self.bias
        # 如果分组数量大于 1，则进行分组路由
        if self.n_groups > 1:
            # 将得分 reshape 为 (batch_size, n_groups, -1)
            scores = scores.view(x.size(0), self.n_groups, -1)
            # 如果没有偏置项，计算每个组的最大得分
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            # 如果有偏置项，计算每个组的 top-2 得分之和
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            # 选择得分最高的 topk_groups 个组
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            # 创建掩码，保留选中的组
            mask = torch.zeros_like(scores[..., 0]).scatter_(1, indices, True)
            # 根据掩码更新得分，并展平为 (batch_size, n_groups * -1)
            scores = (scores * mask.unsqueeze(-1)).flatten(1)
        # 选择得分最高的 topk 个专家
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        # 从原始得分中提取选中的专家的权重
        weights = original_scores.gather(1, indices)
        # 如果使用 sigmoid，对权重进行归一化
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        # 应用路由缩放因子
        weights *= self.route_scale
        # 返回权重和专家索引，确保权重类型与输入一致
        return weights.type_as(x), indices
```

## 关于 Multi-Token Prediction

<p align="center">
  <img src="figures/DeepSeek V3 Multi-Token Prediction.png">
</p>

### 结果生成

```python
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0
) -> List[List[int]]:
    """
    基于给定的 prompt tokens，使用指定的 Transformer 模型生成新的 tokens。

    Args:
        model (Transformer): 用于生成 tokens 的 Transformer 模型。
        prompt_tokens (List[List[int]]): 包含每个序列的 prompt tokens 的列表。
        max_new_tokens (int): 生成的新 tokens 的最大数量。
        eos_id (int): 序列结束符（End-of-Sequence, EOS）的 token ID。
        temperature (float, optional): 采样温度，控制生成的多样性。默认值为 1.0。

    Returns:
        List[List[int]]: 包含每个序列生成 tokens 的列表。
    """
    # 计算每个 prompt 的长度
    prompt_lens = [len(t) for t in prompt_tokens]
    # 检查 prompt 长度是否超过模型的最大序列长度，如果超过则抛出异常
    assert max(prompt_lens) <= model.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
    # 计算总长度，即 prompt 长度与新生成 tokens 长度的总和，但不超出模型最大序列长度
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))

    # 创建一个形状为 (batch_size, total_len) 的 tokens 张量，初始值为 -1
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device="cuda")
    # 将每个 prompt tokens 填充到 tokens 张量的对应位置
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

    # 记录前一个生成位置的索引
    prev_pos = 0
    # 布尔张量，表示每个序列是否已完成生成（即是否遇到 eos_id）
    finished = torch.tensor([False] * len(prompt_tokens), device="cuda")
    # 布尔张量，表示 tokens 张量中哪些位置是 prompt 部分
    prompt_mask = tokens != -1

    # 从 min(prompt_lens) 到 total_len 进行循环，逐位置生成新 token
    for cur_pos in range(min(prompt_lens), total_len):
        # 调用模型计算当前 tokens 的 logits
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        # 根据 temperature 采样下一个 token，temperature > 0 时使用 sample 函数，否则取概率最大的 token
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        # 确保 prompt 部分的 token 不会被覆盖
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        # 更新 tokens 张量
        tokens[:, cur_pos] = next_token
        # 检查是否遇到 eos_id，标记生成完成的序列
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        # 更新前一个生成位置的索引
        prev_pos = cur_pos
        # 如果所有序列都已生成完成，则提前结束循环
        if finished.all():
            break

    # 提取每个序列生成的新 tokens
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        # 从 tokens 张量中提取生成的 tokens
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        # 如果遇到 eos_id，截断生成结果
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        # 将生成结果添加到列表中
        completion_tokens.append(toks)

    # 返回所有序列的生成结果
    return completion_tokens
```

### 结果采样

这段代码实现了一个带有 **温度缩放（Temperature Scaling）** 的采样函数，通常用于从模型的输出 logits 中采样一个 token。它常用于生成模型（如语言模型或图像生成模型）中，以控制生成结果的多样性和随机性。下面我们逐步解释代码的每一部分。

```python
def sample(logits, temperature: float = 1.0):
    """
    Samples a token from the logits using temperature scaling.

    Args:
        logits (torch.Tensor): The logits tensor for token predictions.
        temperature (float, optional): Temperature for scaling logits. Defaults to 1.0.

    Returns:
        torch.Tensor: The sampled token.
    """
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)
```

#### **函数功能**
`sample` 函数的作用是根据输入的 logits，通过温度缩放和随机采样，得到一个 token 索引。

#### **参数说明**
- `logits` (`torch.Tensor`): 模型的输出 logits，形状通常为 `(batch_size, vocab_size)` 或 `(vocab_size,)`，表示每个 token 的未归一化分数。
- `temperature` (`float`, 默认值为 1.0): 温度参数，用于调整 logits 的分布。温度越高，采样结果越随机；温度越低，采样结果越倾向于概率最高的 token。

#### **代码解释**

* 1. 温度缩放
```python
logits = logits / max(temperature, 1e-5)
```
- 将 logits 除以温度值 `temperature`。
- `max(temperature, 1e-5)` 确保温度不会为 0 或负数，否则会导致数值不稳定（分母过小可能导致 logits 值过大，甚至溢出）。
- 当 `temperature > 1` 时，logits 的分布变得更平坦，采样结果更随机。
- 当 `temperature < 1` 时，logits 的分布更尖锐，采样结果更倾向于概率最高的 token。

* 2. 计算概率分布
```python
probs = torch.softmax(logits, dim=-1)
```
- 使用 `softmax` 函数将缩放后的 logits 转换为概率分布。
- `dim=-1` 表示在最后一个维度（通常是 token 的维度）上进行 softmax。

* 3. Gumbel-Max Trick 采样
```python
probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)
```
- 这里实现了 **Gumbel-Max Trick**，一种从离散分布中采样的方法。
- 步骤如下：
  1. 生成服从指数分布 `Exponential(1)` 的随机噪声：`torch.empty_like(probs).exponential_(1)`。
  2. 将概率分布 `probs` 除以这些随机噪声：`probs.div_(...)`。
  3. 对处理后的值取 `argmax`，即找到最大值的索引。这一步实现了从分布中采样一个 token。
- 使用 Gumbel-Max Trick 可以在保证采样随机性的同时，避免直接使用 `torch.multinomial` 函数的计算开销。

* 返回值
- `torch.Tensor`: 采样得到的 token 的索引，形状为 `(batch_size,)` 或 `()`（标量）。

#### **温度参数的作用**
- **`temperature = 1.0`**: 保持 logits 的原始分布，采样结果直接基于 softmax 后的概率。
- **`temperature > 1.0`**: 使 logits 的分布更平坦，增加随机性，生成结果更多样。
- **`temperature < 1.0`**: 使 logits 的分布更尖锐，减少随机性，生成结果更倾向于概率最高的 token。

#### **总结**
该代码实现了基于温度缩放的 token 采样方法，通过调整温度参数 `temperature`，可以控制采样结果的随机性和多样性。Gumbel-Max Trick 的使用使得采样过程更加高效和稳定。

## 关于训练成本

<p align="center">
  <img src="figures/DeepSeek V3 成本.png">
</p>
从论文中的公布细节可以得到它的训练成本估算：

- 以 H800 GPU 小时为单位。H800 GPU 的租赁价格假定为每小时 2 美元。
- 训练分为三个阶段：预训练、上下文扩展和后期训练：
- 预训练：使用了 2664K（266.4 万）GPU 小时，成本约为 532.8 万美元。
- 上下文扩展：使用了 119K（11.9 万）GPU 小时，成本约为 23.8 万美元。
- 后期训练：使用了 5K GPU 小时，成本约为 1,000 美元。
- 总成本：2788K（278.8 万）GPU 小时，总费用为 557.6 万美元。

比起动辄几百亿人民币都训练不出来一个好用的大模型，DeepSeek V3的训练简直颠覆了大家的想象。可能有如下原因：

### 浮点精度降低

这里训练这么省钱当然主要是因为该模型原生就是FP8，还有在模型架构上做了一些优化导致模型训练成本很低。

```python
def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor):
    """
    Perform a matrix multiplication using FP8 precision.

    Args:
        a (torch.Tensor): The first input matrix, must be contiguous.
        a_s (torch.Tensor): The scaling factor for the first input matrix, must be contiguous.
        b (torch.Tensor): The second input matrix, must be contiguous.
        b_s (torch.Tensor): The scaling factor for the second input matrix, must be contiguous.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """
    assert a.is_contiguous() and b.is_contiguous(), 'Input tensors must be contiguous'
    assert a_s.is_contiguous() and b_s.is_contiguous(), 'Scaling factor tensors must be contiguous'
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    fp8_gemm_kernel[grid](a, b, c, a_s, b_s, M, N, K)
    return c
```

### 注意力机制优化

#### 引入 Latent Features

DeepSeek V3除了使用了FP8之外，还有一些其他的模型细节。比如它继续采用了多头潜在注意力（MLA）来实现高效推理。它在传统多头注意力机制（Multi-Head Attention）的基础上，引入了潜在【隐】特征（Latent Features）概念，进一步提高了对复杂关系的建模能力。

```python
if self.q_lora_rank == 0:
    self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
else:
    self.wq_a = Linear(self.dim, self.q_lora_rank)
    self.q_norm = RMSNorm(self.q_lora_rank)
    self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
```

也就是先把token的特征压缩成一个小维度的latent vector，然后再通过一些简单的变换把它扩展到各个头需要的Key和Value空间。对于一些重要的信息，比如旋转位置编码RoPE，会进行单独处理，这样网络仍然可以保留时间和位置的信息。

#### 缓存 Key-Value

```python
if attn_impl == "naive":
    self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), persistent=False)
    self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)
else:
    self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
    self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)
```

### MoE 优化

在MOE架构中，引入了路由专家 (Routed Experts) 和共享专家 (Shared Experts) 。主要是用来激活那些参数需要被更新。

路由专家中主要是用来选择参数进行激活。对于每个输入的token，只有一部分路由专家会被选中来参与计算。这个选择过程是由一个门控机制决定的，比如DeepSeekMoE中用的那种根据亲和度分数来选的Top-K方式。

而共享专家始终参与所有输入的处理。无论输入是什么，所有共享专家都会贡献它们的力量。

### MTP 技术

还用到了一个MTP（多个tokens预测）技术，MTP的核心理念在于训练时，模型不仅要预测下一个token（就像传统语言模型那样），还要同时预测序列后面的几个token。这样一来，模型就能获得更丰富的训练信息，有助于它更深入地理解上下文以及长距离的依赖关系。

### 并行计算

```python
class ColumnParallelLinear(Linear):
    """
    Linear layer with column parallelism, splitting output features across distributed processes.

    Args:
        in_features (int): Number of input features.
        out_features (int): Total number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for column parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with column-parallel computation.
        """
        y = linear(x, self.weight, self.bias)
        return y


class RowParallelLinear(Linear):
    """
    Linear layer with row parallelism, splitting input features across distributed processes.

    Args:
        in_features (int): Total number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for row parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with row-parallel computation.
        """
        y = linear(x, self.weight)
        if world_size > 1:
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y
```