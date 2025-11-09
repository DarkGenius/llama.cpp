#include "models.h"

llm_build_kimi_linear::llm_build_kimi_linear(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {

    // Kimi-Linear head dimensions
    const int64_t n_embd_head_qk_rope = hparams.n_qk_rope_head_dim;
    const int64_t n_embd_head_qk_nope = hparams.n_qk_nope_head_dim;
    const int64_t n_embd_head_v       = hparams.n_v_head_dim;
    const int64_t n_embd_head_k       = n_embd_head_qk_rope + n_embd_head_qk_nope;

    const uint32_t kv_lora_rank = hparams.n_lora_kv;

    // Pre-scale for RoPE/attention
    const float mscale      = attn_factor * (1.0f + hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
    const float kq_scale    = 1.0f * mscale * mscale / sqrtf(float(n_embd_head_k));
    const float attn_factor_scaled = 1.0f / (1.0f + 0.1f * logf(1.0f / freq_scale));

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // {n_embd, n_tokens}
    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self_attention
        {
            // Check if this layer uses MLA or KDA
            bool is_mla = hparams.mla_layer_arr[il];

            if (is_mla) {
                // ============ MLA (Multi-head Latent Attention) ============
                // Query projection
                ggml_tensor * q = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
                cb(q, "q", il);

                // Split Q into nope and rope parts
                ggml_tensor * q_nope =
                    ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens,
                                 ggml_row_size(q->type, n_embd_head_k),
                                 ggml_row_size(q->type, n_embd_head_k) * n_head, 0);
                cb(q_nope, "q_nope", il);

                ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head, n_tokens,
                                                   ggml_row_size(q->type, n_embd_head_k),
                                                   ggml_row_size(q->type, n_embd_head_k) * n_head,
                                                   ggml_row_size(q->type, n_embd_head_qk_nope));
                cb(q_pe, "q_pe", il);

                // KV compression
                ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
                cb(kv_cmpr_pe, "kv_cmpr_pe", il);

                // Split into compressed KV and rope part for K
                ggml_tensor * kv_cmpr =
                    ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                                 ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
                cb(kv_cmpr, "kv_cmpr", il);

                ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                                                   ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                                   ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                                                   ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
                cb(k_pe, "k_pe", il);

                // Apply RoPE to query and key rope parts
                q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                     ext_factor, attn_factor_scaled, beta_fast, beta_slow);
                cb(q_pe, "q_pe", il);

                k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                     ext_factor, attn_factor_scaled, beta_fast, beta_slow);
                cb(k_pe, "k_pe", il);

                // Normalize compressed KV
                kv_cmpr = build_norm(kv_cmpr, model.layers[il].attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
                cb(kv_cmpr, "kv_cmpr", il);

                // Decompress KV
                ggml_tensor * kv = ggml_mul_mat(ctx0, model.layers[il].wkv_b, kv_cmpr);
                cb(kv, "kv", il);

                // Split decompressed KV into K_nope and V
                ggml_tensor * k_nope =
                    ggml_view_3d(ctx0, kv, n_embd_head_qk_nope, n_head, n_tokens,
                                 ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v),
                                 ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v) * n_head, 0);
                cb(k_nope, "k_nope", il);

                ggml_tensor * Vcur = ggml_view_3d(ctx0, kv, n_embd_head_v, n_head, n_tokens,
                                                   ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v),
                                                   ggml_row_size(kv->type, n_embd_head_qk_nope + n_embd_head_v) * n_head,
                                                   ggml_row_size(kv->type, n_embd_head_qk_nope));
                cb(Vcur, "Vcur", il);

                Vcur = ggml_cont(ctx0, Vcur);
                cb(Vcur, "Vcur_cont", il);

                // Concatenate rope and nope parts for Q and K
                ggml_tensor * Qcur = ggml_concat(ctx0, q_pe, q_nope, 0);
                cb(Qcur, "Qcur", il);

                ggml_tensor * Kcur = ggml_concat(ctx0, ggml_repeat(ctx0, k_pe, q_pe), k_nope, 0);
                cb(Kcur, "Kcur", il);

                // Standard MLA attention
                cur = build_attn(inp_attn,
                                 model.layers[il].wo, NULL,
                                 Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            } else {
                // ============ KDA (Kimi Delta Attention) ============
                // Simplified implementation using existing GGML operators
                // Note: Full KDA with recurrent state requires custom operators (see KDA_IMPLEMENTATION_GUIDE.md)

                // 1. Q/K/V projections
                ggml_tensor * Q = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
                cb(Q, "Q", il);

                ggml_tensor * K = ggml_mul_mat(ctx0, model.layers[il].wk, cur);
                cb(K, "K", il);

                ggml_tensor * V = ggml_mul_mat(ctx0, model.layers[il].wv, cur);
                cb(V, "V", il);

                // Reshape to [n_tokens, n_head, d_head]
                Q = ggml_reshape_3d(ctx0, Q, n_embd_head_k, n_head, n_tokens);
                cb(Q, "Q_reshaped", il);

                K = ggml_reshape_3d(ctx0, K, n_embd_head_k, n_head, n_tokens);
                cb(K, "K_reshaped", il);

                V = ggml_reshape_3d(ctx0, V, n_embd_head_v, n_head, n_tokens);
                cb(V, "V_reshaped", il);

                // 2. Short convolution (simplified - using 1x1 conv as placeholder)
                // Full implementation requires conv1d with state management
                // For now, apply learnable weighting across token dimension
                ggml_tensor * conv_wq = model.layers[il].wq_conv1d;
                ggml_tensor * conv_wk = model.layers[il].wk_conv1d;
                ggml_tensor * conv_wv = model.layers[il].wv_conv1d;

                // Simplified: Just apply first weight (1x1 conv approximation)
                // TODO: Implement proper conv1d with kernel_size=4
                ggml_tensor * conv_wq_first = ggml_view_2d(ctx0, conv_wq,
                    n_head * n_embd_head_k, 1,
                    ggml_row_size(conv_wq->type, n_head * n_embd_head_k), 0);
                Q = ggml_mul(ctx0, Q, ggml_reshape_3d(ctx0, conv_wq_first, n_embd_head_k, n_head, 1));
                cb(Q, "Q_conv", il);

                ggml_tensor * conv_wk_first = ggml_view_2d(ctx0, conv_wk,
                    n_head * n_embd_head_k, 1,
                    ggml_row_size(conv_wk->type, n_head * n_embd_head_k), 0);
                K = ggml_mul(ctx0, K, ggml_reshape_3d(ctx0, conv_wk_first, n_embd_head_k, n_head, 1));
                cb(K, "K_conv", il);

                ggml_tensor * conv_wv_first = ggml_view_2d(ctx0, conv_wv,
                    n_head * n_embd_head_v, 1,
                    ggml_row_size(conv_wv->type, n_head * n_embd_head_v), 0);
                V = ggml_mul(ctx0, V, ggml_reshape_3d(ctx0, conv_wv_first, n_embd_head_v, n_head, 1));
                cb(V, "V_conv", il);

                // 3. Split Q and K into rope and nope parts
                ggml_tensor * q_nope =
                    ggml_view_3d(ctx0, Q, n_embd_head_qk_nope, n_head, n_tokens,
                                 ggml_row_size(Q->type, n_embd_head_k),
                                 ggml_row_size(Q->type, n_embd_head_k) * n_head, 0);
                cb(q_nope, "q_nope", il);

                ggml_tensor * q_pe = ggml_view_3d(ctx0, Q, n_embd_head_qk_rope, n_head, n_tokens,
                                                   ggml_row_size(Q->type, n_embd_head_k),
                                                   ggml_row_size(Q->type, n_embd_head_k) * n_head,
                                                   ggml_row_size(Q->type, n_embd_head_qk_nope));
                cb(q_pe, "q_pe", il);

                ggml_tensor * k_nope =
                    ggml_view_3d(ctx0, K, n_embd_head_qk_nope, n_head, n_tokens,
                                 ggml_row_size(K->type, n_embd_head_k),
                                 ggml_row_size(K->type, n_embd_head_k) * n_head, 0);
                cb(k_nope, "k_nope", il);

                ggml_tensor * k_pe = ggml_view_3d(ctx0, K, n_embd_head_qk_rope, n_head, n_tokens,
                                                   ggml_row_size(K->type, n_embd_head_k),
                                                   ggml_row_size(K->type, n_embd_head_k) * n_head,
                                                   ggml_row_size(K->type, n_embd_head_qk_nope));
                cb(k_pe, "k_pe", il);

                // 4. Apply RoPE to rope portions
                q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                     ext_factor, attn_factor_scaled, beta_fast, beta_slow);
                cb(q_pe, "q_pe_rope", il);

                k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                     ext_factor, attn_factor_scaled, beta_fast, beta_slow);
                cb(k_pe, "k_pe_rope", il);

                // 5. Concatenate rope and nope parts back
                ggml_tensor * Qcur = ggml_concat(ctx0, q_pe, q_nope, 0);
                cb(Qcur, "Qcur", il);

                ggml_tensor * Kcur = ggml_concat(ctx0, k_pe, k_nope, 0);
                cb(Kcur, "Kcur", il);

                ggml_tensor * Vcur = V;
                cb(Vcur, "Vcur", il);

                // 6. Apply linear attention (simplified - using standard attention)
                // Full KDA uses delta attention with recurrent state: A[t] = exp(log_A) * A[t-1] + K[t] ⊗ V[t]
                // This simplified version uses standard attention as an approximation
                ggml_tensor * attn_out = build_attn(inp_attn,
                                 nullptr, nullptr,  // No wo projection yet
                                 Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
                cb(attn_out, "attn_out", il);

                // 7. Output gating (g = sigmoid(g_b @ g_a @ x))
                ggml_tensor * gate_h = ggml_mul_mat(ctx0, model.layers[il].attn_g_a, cur);
                cb(gate_h, "gate_h", il);

                ggml_tensor * gate = ggml_mul_mat(ctx0, model.layers[il].attn_g_b, gate_h);
                cb(gate, "gate", il);

                gate = ggml_sigmoid(ctx0, gate);
                cb(gate, "gate_sigmoid", il);

                // Reshape gate for multiplication: [n_tokens, n_head * n_embd_head_v]
                gate = ggml_reshape_2d(ctx0, gate, n_head * n_embd_head_v, n_tokens);
                cb(gate, "gate_reshaped", il);

                // Apply gate to attention output
                attn_out = ggml_mul(ctx0, attn_out, gate);
                cb(attn_out, "attn_gated", il);

                // 8. Output normalization
                attn_out = build_norm(attn_out, model.layers[il].attn_o_norm, nullptr, LLM_NORM_RMS, il);
                cb(attn_out, "attn_normed", il);

                // 9. Output projection
                cur = ggml_mul_mat(ctx0, model.layers[il].wo, attn_out);
                cb(cur, "attn_out", il);

                // Log info about simplified KDA implementation (once)
                if (il == 0) {
                    fprintf(stderr, "%s: info: KDA layers using simplified implementation (standard attention + gating)\n",
                            __func__);
                    fprintf(stderr, "%s: info: Full KDA with recurrent linear attention requires custom GGML operators\n",
                            __func__);
                }
            }
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // FFN norm
        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // MoE FFN (routed experts)
        ggml_tensor * moe_out = build_moe_ffn(cur,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_down_exps,
            model.layers[il].ffn_gate_inp_b,  // router bias
            n_expert, n_expert_used,
            LLM_FFN_SILU, false, // norm_w
            true, // scale_w
            hparams.f_routed_scaling_factor,
            (llama_expert_gating_func_type) hparams.expert_gating_func,
            il);
        cb(moe_out, "ffn_moe_out", il);

        // Shared expert FFN
        if (hparams.n_expert_shared > 0) {
            ggml_tensor * ffn_shexp =
                build_ffn(cur,
                    model.layers[il].ffn_up_shexp, NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        } else {
            cur = moe_out;
        }

        // Residual connection
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    // Final norm
    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    res->t_embd = cur;

    // lm_head
    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
}
