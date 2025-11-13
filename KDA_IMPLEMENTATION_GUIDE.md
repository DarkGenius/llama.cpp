# KDA (Kimi Delta Attention) Implementation Guide

## Overview

This document provides a detailed guide for implementing KDA (Kimi Delta Attention) layers in llama.cpp. KDA is a linear-complexity attention mechanism used in 20 out of 27 layers in the Kimi-Linear-48B model.

## Current Status (Updated: Commit 81f64fb)

- ✅ **KDA Implementation**: Enhanced with all major components (see below)
- ✅ **MLA Implementation**: Fully working (7/27 layers)
- ✅ **MoE FFN**: Fully working (all layers)

### What's Implemented in KDA (Commits e3b0a53, 81f64fb):

✅ **Tensor Creation** (llama-model.cpp, llama-model.h):
- All 9 KDA tensor fields in llama_layer structure
- Conv1d weights (wq_conv1d, wk_conv1d, wv_conv1d)
- Delta gating projections (attn_f_a, attn_f_b, attn_dt_b)
- Output gating projections (attn_g_a, attn_g_b)
- Output normalization (attn_o_norm)

✅ **Graph Builder** (kimi-linear.cpp):
- Q/K/V projections and reshaping
- Short convolution (differentiated single-token vs multi-token)
- Q/K rope/nope splitting
- RoPE application to positional encodings
- Delta gating (LoRA projections + GELU activation)
- Causal attention (O(N²) standard attention)
- Output gating (sigmoid activation)
- Output normalization (RMS norm)

### What's Still Missing for True O(N) Linear Attention:

❌ **Custom GGML Operators**:
- Stateful conv1d with cross-batch state persistence
- Recurrent linear attention: `S[t] = exp(A_log * dt) * S[t-1] + K[t] ⊗ V[t]`
- CPU/CUDA/Metal kernel implementations

❌ **Recurrent State Management**:
- KV cache for conv states (currently no cross-batch persistence)
- KV cache for attention states (recurrent accumulator)
- State update and retrieval infrastructure

❌ **A_log Decay Parameter**:
- Tensor not created (would need to add to tensor creation)
- Integration with delta gating for temporal modulation

**Key Difference:** Current implementation has all KDA components but uses
standard O(N²) causal attention instead of O(N) recurrent linear attention.
This provides functional inference with correct causality but without the
linear complexity benefit.

**Estimated effort for full O(N) implementation:** 6-10 weeks (see roadmap below)

## KDA Architecture

### Components

1. **Short Convolution (kernel size 4)**
   - Applied to Q, K, V projections
   - Maintains conv state across time steps
   - Input: `[n_tokens, hidden_dim]`
   - Output: `[n_tokens, hidden_dim]`
   - Conv state: `[kernel_size-1, hidden_dim]` = `[3, hidden_dim]`

2. **Delta Attention**
   - Linear attention with exponential decay
   - Formula: `A[t] = exp(log_A) * A[t-1] + K[t] ⊗ V[t]`
   - Maintains recurrent state across sequence
   - State shape: `[n_head, d_head, d_value]`

3. **Delta Gating**
   - Two-stage LoRA projection for temporal control
   - `dt = softplus(f_b @ f_a @ x + dt_bias)`
   - Controls information flow across time

4. **Output Gating**
   - `gate = sigmoid(g_b @ g_a @ x)`
   - `output = gate * attention_output`
   - Final layer norm on output

## Required Tensors

### Per KDA Layer Tensors

```python
# Q/K/V projections (standard)
wq: [n_embd, n_head * d_head]                    # Query projection
wk: [n_embd, n_head_kv * d_head]                 # Key projection
wv: [n_embd, n_head_kv * d_value]                # Value projection
wo: [n_head * d_value, n_embd]                    # Output projection

# Short convolution (kernel_size=4)
q_conv1d: [kernel_size, n_head * d_head]         # Q conv weights
k_conv1d: [kernel_size, n_head_kv * d_head]      # K conv weights
v_conv1d: [kernel_size, n_head_kv * d_value]     # V conv weights

# Delta attention
a_log: [n_head, d_head]                           # Log decay parameter A

# Delta gating (LoRA style)
f_a_proj: [n_embd, lora_rank]                     # First projection for dt
f_b_proj: [lora_rank, n_head]                     # Second projection for dt
dt_bias: [n_head]                                 # Bias for dt
b_proj: [n_head, d_head]                          # Projection for B matrix

# Output gating (LoRA style)
g_a_proj: [n_embd, lora_rank]                     # First projection for gate
g_b_proj: [lora_rank, n_head * d_value]           # Second projection for gate
o_norm: [n_head * d_value]                        # Output normalization
```

### Tensor Creation Code

Add to `src/llama-model.cpp` around line 6402:

```cpp
// KDA (Kimi Delta Attention) tensors
const int64_t kda_lora_rank = 256;  // Typical LoRA rank for gating

// Q/K/V conv1d weights (kernel_size = 4)
layer.wq_conv1d = create_tensor(tn(LLM_TENSOR_ATTN_Q_CONV1D, "weight", i),
    {hparams.n_shortconv_l_cache, n_head * n_embd_head_k_full}, 0);
layer.wk_conv1d = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV1D, "weight", i),
    {hparams.n_shortconv_l_cache, n_head_kv * n_embd_head_k_full}, 0);
layer.wv_conv1d = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV1D, "weight", i),
    {hparams.n_shortconv_l_cache, n_head_kv * n_embd_head_v}, 0);

// Delta attention decay parameter
layer.attn_a_log = create_tensor(tn(LLM_TENSOR_ATTN_A_LOG, "weight", i),
    {n_head, n_embd_head_k_full}, 0);

// Delta gating (dt computation)
layer.attn_f_a_proj = create_tensor(tn(LLM_TENSOR_ATTN_F_A_PROJ, "weight", i),
    {n_embd, kda_lora_rank}, 0);
layer.attn_f_b_proj = create_tensor(tn(LLM_TENSOR_ATTN_F_B_PROJ, "weight", i),
    {kda_lora_rank, n_head}, 0);
layer.attn_dt_bias = create_tensor(tn(LLM_TENSOR_ATTN_DT_BIAS, "weight", i),
    {n_head}, 0);
layer.attn_b_proj = create_tensor(tn(LLM_TENSOR_ATTN_B_PROJ, "weight", i),
    {n_head, n_embd_head_k_full}, 0);

// Output gating
layer.attn_g_a_proj = create_tensor(tn(LLM_TENSOR_ATTN_G_A_PROJ, "weight", i),
    {n_embd, kda_lora_rank}, 0);
layer.attn_g_b_proj = create_tensor(tn(LLM_TENSOR_ATTN_G_B_PROJ, "weight", i),
    {kda_lora_rank, n_head * n_embd_head_v}, 0);
layer.attn_o_norm = create_tensor(tn(LLM_TENSOR_ATTN_O_NORM, "weight", i),
    {n_head * n_embd_head_v}, 0);
```

## GGML Operators Needed

### 1. Short Convolution with State

```c
// Short 1D convolution with state management
struct ggml_tensor * ggml_conv_1d_stateful(
    struct ggml_context * ctx,
    struct ggml_tensor  * input,        // [n_tokens, channels]
    struct ggml_tensor  * conv_weights, // [kernel_size, channels]
    struct ggml_tensor  * conv_state    // [kernel_size-1, channels] (in/out)
);
```

**Implementation notes:**
- Slide conv_state by 1 position
- Concatenate new input: `[conv_state, input]`
- Apply conv weights
- Update conv_state with latest inputs
- Return convolved output

**CPU kernel:** Use sliding window + dot product
**CUDA kernel:** Parallel per-channel convolution with shared memory

### 2. Delta Attention (Linear Attention)

```c
// Linear attention with delta rule and recurrent state
struct ggml_tensor * ggml_delta_attention(
    struct ggml_context * ctx,
    struct ggml_tensor  * Q,            // [n_head, d_head, n_tokens]
    struct ggml_tensor  * K,            // [n_head, d_head, n_tokens]
    struct ggml_tensor  * V,            // [n_head, d_value, n_tokens]
    struct ggml_tensor  * A_log,        // [n_head, d_head] decay param
    struct ggml_tensor  * dt,           // [n_head, n_tokens] time delta
    struct ggml_tensor  * recur_state   // [n_head, d_head, d_value] (in/out)
);
```

**Algorithm:**
```python
for t in range(n_tokens):
    A = exp(A_log * dt[t])              # [n_head, d_head]
    recur_state = A * recur_state + K[t] ⊗ V[t]  # Update state
    output[t] = Q[t] @ recur_state      # Query state
```

**Implementation notes:**
- Process tokens sequentially (recurrence dependency)
- Use einsum for outer product: `K[t] ⊗ V[t]`
- Efficient matmul for `Q[t] @ recur_state`

**CPU kernel:** Sequential loop with BLAS
**CUDA kernel:** Parallel per-head with sequential tokens

### 3. Delta Gating

```c
// Compute delta gating values
struct ggml_tensor * ggml_delta_gate(
    struct ggml_context * ctx,
    struct ggml_tensor  * x,            // [n_tokens, n_embd]
    struct ggml_tensor  * f_a_proj,     // [n_embd, lora_rank]
    struct ggml_tensor  * f_b_proj,     // [lora_rank, n_head]
    struct ggml_tensor  * dt_bias       // [n_head]
);
```

**Algorithm:**
```python
h = x @ f_a_proj                        # [n_tokens, lora_rank]
dt = h @ f_b_proj + dt_bias             # [n_tokens, n_head]
dt = softplus(dt)                       # Ensure positive
```

**Implementation:** Compose from existing matmul + softplus

## Graph Builder Implementation

Add to `src/models/kimi-linear.cpp` around line 123:

```cpp
// ============ KDA (Kimi Delta Attention) ============

// 1. Q/K/V projections
ggml_tensor * Q = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
ggml_tensor * K = ggml_mul_mat(ctx0, model.layers[il].wk, cur);
ggml_tensor * V = ggml_mul_mat(ctx0, model.layers[il].wv, cur);

// 2. Short convolutions (with state)
Q = ggml_conv_1d_stateful(ctx0, Q, model.layers[il].wq_conv1d,
                          kv_self.conv_states_q[il]);
K = ggml_conv_1d_stateful(ctx0, K, model.layers[il].wk_conv1d,
                          kv_self.conv_states_k[il]);
V = ggml_conv_1d_stateful(ctx0, V, model.layers[il].wv_conv1d,
                          kv_self.conv_states_v[il]);

// 3. Apply RoPE to RoPE portion of Q/K
ggml_tensor * Q_rope = ggml_view_3d(ctx0, Q, n_embd_head_qk_rope, n_head, n_tokens, ...);
ggml_tensor * K_rope = ggml_view_3d(ctx0, K, n_embd_head_qk_rope, n_head_kv, n_tokens, ...);
Q_rope = ggml_rope_ext(ctx0, Q_rope, inp_pos, ...);
K_rope = ggml_rope_ext(ctx0, K_rope, inp_pos, ...);

// 4. Compute delta gating (dt)
ggml_tensor * dt = ggml_delta_gate(ctx0, cur,
                                   model.layers[il].attn_f_a_proj,
                                   model.layers[il].attn_f_b_proj,
                                   model.layers[il].attn_dt_bias);

// 5. Delta attention (linear attention with state)
ggml_tensor * attn_out = ggml_delta_attention(ctx0, Q, K, V,
                                              model.layers[il].attn_a_log,
                                              dt,
                                              kv_self.recur_states[il]);

// 6. Output gating
ggml_tensor * gate_h = ggml_mul_mat(ctx0, model.layers[il].attn_g_a_proj, cur);
ggml_tensor * gate = ggml_mul_mat(ctx0, model.layers[il].attn_g_b_proj, gate_h);
gate = ggml_sigmoid(ctx0, gate);
attn_out = ggml_mul(ctx0, gate, attn_out);

// 7. Output norm
attn_out = build_norm(attn_out, model.layers[il].attn_o_norm, nullptr, LLM_NORM_RMS, il);

// 8. Output projection
cur = ggml_mul_mat(ctx0, model.layers[il].wo, attn_out);
```

## KV Cache for KDA

### Required State

```cpp
struct llama_kv_cache_kda {
    // Conv states for Q, K, V
    std::vector<struct ggml_tensor *> conv_states_q;  // [n_layer][3, hidden_dim]
    std::vector<struct ggml_tensor *> conv_states_k;  // [n_layer][3, hidden_dim]
    std::vector<struct ggml_tensor *> conv_states_v;  // [n_layer][3, hidden_dim]

    // Recurrent states for delta attention
    std::vector<struct ggml_tensor *> recur_states;   // [n_layer][n_head, d_head, d_value]
};
```

Add to `llama_kv_cache` structure in `src/llama-kv-cache.h`:

```cpp
struct llama_kv_cache {
    // ... existing fields ...

    // KDA-specific states
    llama_kv_cache_kda kda_states;
};
```

## Implementation Roadmap

### Phase 1: GGML Operators (2-3 weeks)
1. Implement `ggml_conv_1d_stateful` CPU kernel
2. Implement `ggml_delta_attention` CPU kernel
3. Implement `ggml_delta_gate` (or use existing ops)
4. Add operator registration and compute functions
5. Write unit tests for each operator

### Phase 2: Integration (1 week)
1. Add KDA tensor creation
2. Implement KDA KV cache
3. Update graph builder with KDA implementation
4. Test with small synthetic data

### Phase 3: GPU Kernels (2-3 weeks)
1. Implement CUDA kernels for conv1d_stateful
2. Implement CUDA kernels for delta_attention
3. Implement Metal kernels (optional)
4. Benchmark and optimize

### Phase 4: Testing & Validation (1 week)
1. Convert full Kimi-Linear model
2. Compare outputs with PyTorch reference
3. Accuracy testing
4. Performance benchmarking

**Total estimated time: 6-10 weeks**

## Testing Strategy

### Unit Tests

```cpp
// Test short convolution with state
void test_conv_1d_stateful() {
    // Create input, weights, state
    // Run convolution
    // Verify output shape and values
    // Verify state update
}

// Test delta attention
void test_delta_attention() {
    // Create Q, K, V, A_log, dt, state
    // Run attention
    // Verify output shape
    // Verify state evolution
}
```

### Integration Tests

```python
# Compare with PyTorch reference
import torch
from transformers import AutoModel

# Load PyTorch model
model_pt = AutoModel.from_pretrained("moonshotai/Kimi-Linear-48B-A3B-Instruct")

# Load llama.cpp model
model_cpp = load_model("kimi-linear-48b.gguf")

# Generate same input
input_ids = torch.randint(0, 32000, (1, 128))

# Run both models
output_pt = model_pt(input_ids)
output_cpp = model_cpp.forward(input_ids)

# Compare outputs
assert torch.allclose(output_pt, output_cpp, atol=1e-3)
```

## References

- **KDA Paper**: https://arxiv.org/pdf/2510.26692
- **KDA Implementation**: https://github.com/fla-org/flash-linear-attention/pull/621/
- **Delta Rule**: Linear attention with exponential decay
- **State Space Models**: Similar recurrent mechanisms in Mamba/RWKV

## Contact

For questions or contributions to KDA implementation:
- Open an issue in llama.cpp repository
- Reference this guide in implementation PRs
