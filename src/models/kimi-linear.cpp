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
                // TODO: KDA requires custom GGML operators:
                // 1. Short convolution (conv1d with kernel size 4)
                // 2. Linear attention with recurrent state management
                // 3. Delta gating mechanism
                //
                // For now, use a simple placeholder that prevents compilation errors
                // but will fail at runtime if KDA layers are used.

                // Placeholder: simple Q projection (will not work correctly)
                ggml_tensor * Qcur = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur_placeholder", il);

                // Create dummy K and V to satisfy build_attn
                ggml_tensor * Kcur = Qcur;  // Placeholder
                ggml_tensor * Vcur = Qcur;  // Placeholder

                // This will not work correctly without proper KDA implementation
                cur = build_attn(inp_attn,
                                 model.layers[il].wo, NULL,
                                 Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);

                // Log warning that KDA is not implemented (using fprintf to stderr)
                if (il == 0) {  // Only warn once for first KDA layer
                    fprintf(stderr, "%s: warning: KDA layers are using placeholder implementation - output will be incorrect!\n",
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
