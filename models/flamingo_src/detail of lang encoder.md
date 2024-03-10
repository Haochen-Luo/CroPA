```python
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32003, 4096)
    (layers): ModuleList(
      (0-2): 3 x FlamingoLayer(
        (decoder_layer): LlamaDecoderLayer(
          (self_attn): LlamaAttention(
            (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (rotary_emb): LlamaRotaryEmbedding()
          )
          (mlp): LlamaMLP(
            (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
            (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (act_fn): SiLUActivation()
          )
          (input_layernorm): LlamaRMSNorm()
          (post_attention_layernorm): LlamaRMSNorm()
        )
      )
      (3): FlamingoLayer(
        (gated_cross_attn_layer): GatedCrossAttentionBlock(
          (attn): MaskedCrossAttention(
            (norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
            (to_q): Linear(in_features=4096, out_features=512, bias=False)
            (to_kv): Linear(in_features=1024, out_features=1024, bias=False)
            (to_out): Linear(in_features=512, out_features=4096, bias=False)
          )
          (ff): Sequential(
            (0): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
            (1): Linear(in_features=4096, out_features=16384, bias=False)
            (2): GELU(approximate='none')
            (3): Linear(in_features=16384, out_features=4096, bias=False)
          )
        )
        (decoder_layer): LlamaDecoderLayer(
          (self_attn): LlamaAttention(
            (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (rotary_emb): LlamaRotaryEmbedding()
          )
          (mlp): LlamaMLP(
            (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
            (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (act_fn): SiLUActivation()
          )
          (input_layernorm): LlamaRMSNorm()
          (post_attention_layernorm): LlamaRMSNorm()
        )
      )
      (4-6): 3 x FlamingoLayer(
        (decoder_layer): LlamaDecoderLayer(
          (self_attn): LlamaAttention(
            (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (rotary_emb): LlamaRotaryEmbedding()
          )
          (mlp): LlamaMLP(
            (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
            (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (act_fn): SiLUActivation()
          )
          (input_layernorm): LlamaRMSNorm()
          (post_attention_layernorm): LlamaRMSNorm()
        )
      )
      (7): FlamingoLayer(
        (gated_cross_attn_layer): GatedCrossAttentionBlock(
          (attn): MaskedCrossAttention(
            (norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
            (to_q): Linear(in_features=4096, out_features=512, bias=False)
            (to_kv): Linear(in_features=1024, out_features=1024, bias=False)
            (to_out): Linear(in_features=512, out_features=4096, bias=False)
          )
          (ff): Sequential(
            (0): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
            (1): Linear(in_features=4096, out_features=16384, bias=False)
            (2): GELU(approximate='none')
            (3): Linear(in_features=16384, out_features=4096, bias=False)
          )
        )
        (decoder_layer): LlamaDecoderLayer(
          (self_attn): LlamaAttention(
            (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (rotary_emb): LlamaRotaryEmbedding()
          )
          (mlp): LlamaMLP(
            (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
            (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (act_fn): SiLUActivation()
          )
          (input_layernorm): LlamaRMSNorm()
          (post_attention_layernorm): LlamaRMSNorm()
        )
      )
      (8-10): 3 x FlamingoLayer(
        (decoder_layer): LlamaDecoderLayer(
          (self_attn): LlamaAttention(
            (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (rotary_emb): LlamaRotaryEmbedding()
          )
          (mlp): LlamaMLP(
            (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
            (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (act_fn): SiLUActivation()
          )
          (input_layernorm): LlamaRMSNorm()
          (post_attention_layernorm): LlamaRMSNorm()
        )
      )
      (11): FlamingoLayer(
        (gated_cross_attn_layer): GatedCrossAttentionBlock(
          (attn): MaskedCrossAttention(
            (norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
            (to_q): Linear(in_features=4096, out_features=512, bias=False)
            (to_kv): Linear(in_features=1024, out_features=1024, bias=False)
            (to_out): Linear(in_features=512, out_features=4096, bias=False)
          )
          (ff): Sequential(
            (0): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
            (1): Linear(in_features=4096, out_features=16384, bias=False)
            (2): GELU(approximate='none')
            (3): Linear(in_features=16384, out_features=4096, bias=False)
          )
        )
        (decoder_layer): LlamaDecoderLayer(
          (self_attn): LlamaAttention(
            (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (rotary_emb): LlamaRotaryEmbedding()
          )
          (mlp): LlamaMLP(
            (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
            (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (act_fn): SiLUActivation()
          )
          (input_layernorm): LlamaRMSNorm()
          (post_attention_layernorm): LlamaRMSNorm()
        )
      )
      (12-14): 3 x FlamingoLayer(
        (decoder_layer): LlamaDecoderLayer(
          (self_attn): LlamaAttention(
            (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (rotary_emb): LlamaRotaryEmbedding()
          )
          (mlp): LlamaMLP(
            (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
            (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (act_fn): SiLUActivation()
          )
          (input_layernorm): LlamaRMSNorm()
          (post_attention_layernorm): LlamaRMSNorm()
        )
      )
      (15): FlamingoLayer(
        (gated_cross_attn_layer): GatedCrossAttentionBlock(
          (attn): MaskedCrossAttention(
            (norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
            (to_q): Linear(in_features=4096, out_features=512, bias=False)
            (to_kv): Linear(in_features=1024, out_features=1024, bias=False)
            (to_out): Linear(in_features=512, out_features=4096, bias=False)
          )
          (ff): Sequential(
            (0): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
            (1): Linear(in_features=4096, out_features=16384, bias=False)
            (2): GELU(approximate='none')
            (3): Linear(in_features=16384, out_features=4096, bias=False)
          )
        )
        (decoder_layer): LlamaDecoderLayer(
          (self_attn): LlamaAttention(
            (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (rotary_emb): LlamaRotaryEmbedding()
          )
          (mlp): LlamaMLP(
            (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
            (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (act_fn): SiLUActivation()
          )
          (input_layernorm): LlamaRMSNorm()
          (post_attention_layernorm): LlamaRMSNorm()
        )
      )
      (16-18): 3 x FlamingoLayer(
        (decoder_layer): LlamaDecoderLayer(
          (self_attn): LlamaAttention(
            (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (rotary_emb): LlamaRotaryEmbedding()
          )
          (mlp): LlamaMLP(
            (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
            (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (act_fn): SiLUActivation()
          )
          (input_layernorm): LlamaRMSNorm()
          (post_attention_layernorm): LlamaRMSNorm()
        )
      )
      (19): FlamingoLayer(
        (gated_cross_attn_layer): GatedCrossAttentionBlock(
          (attn): MaskedCrossAttention(
            (norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
            (to_q): Linear(in_features=4096, out_features=512, bias=False)
            (to_kv): Linear(in_features=1024, out_features=1024, bias=False)
            (to_out): Linear(in_features=512, out_features=4096, bias=False)
          )
          (ff): Sequential(
            (0): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
            (1): Linear(in_features=4096, out_features=16384, bias=False)
            (2): GELU(approximate='none')
            (3): Linear(in_features=16384, out_features=4096, bias=False)
          )
        )
        (decoder_layer): LlamaDecoderLayer(
          (self_attn): LlamaAttention(
            (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (rotary_emb): LlamaRotaryEmbedding()
          )
          (mlp): LlamaMLP(
            (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
            (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (act_fn): SiLUActivation()
          )
          (input_layernorm): LlamaRMSNorm()
          (post_attention_layernorm): LlamaRMSNorm()
        )
      )
      (20-22): 3 x FlamingoLayer(
        (decoder_layer): LlamaDecoderLayer(
          (self_attn): LlamaAttention(
            (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (rotary_emb): LlamaRotaryEmbedding()
          )
          (mlp): LlamaMLP(
            (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
            (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (act_fn): SiLUActivation()
          )
          (input_layernorm): LlamaRMSNorm()
          (post_attention_layernorm): LlamaRMSNorm()
        )
      )
      (23): FlamingoLayer(
        (gated_cross_attn_layer): GatedCrossAttentionBlock(
          (attn): MaskedCrossAttention(
            (norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
            (to_q): Linear(in_features=4096, out_features=512, bias=False)
            (to_kv): Linear(in_features=1024, out_features=1024, bias=False)
            (to_out): Linear(in_features=512, out_features=4096, bias=False)
          )
          (ff): Sequential(
            (0): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
            (1): Linear(in_features=4096, out_features=16384, bias=False)
            (2): GELU(approximate='none')
            (3): Linear(in_features=16384, out_features=4096, bias=False)
          )
        )
        (decoder_layer): LlamaDecoderLayer(
          (self_attn): LlamaAttention(
            (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (rotary_emb): LlamaRotaryEmbedding()
          )
          (mlp): LlamaMLP(
            (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
            (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (act_fn): SiLUActivation()
          )
          (input_layernorm): LlamaRMSNorm()
          (post_attention_layernorm): LlamaRMSNorm()
        )
      )
      (24-26): 3 x FlamingoLayer(
        (decoder_layer): LlamaDecoderLayer(
          (self_attn): LlamaAttention(
            (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (rotary_emb): LlamaRotaryEmbedding()
          )
          (mlp): LlamaMLP(
            (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
            (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (act_fn): SiLUActivation()
          )
          (input_layernorm): LlamaRMSNorm()
          (post_attention_layernorm): LlamaRMSNorm()
        )
      )
      (27): FlamingoLayer(
        (gated_cross_attn_layer): GatedCrossAttentionBlock(
          (attn): MaskedCrossAttention(
            (norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
            (to_q): Linear(in_features=4096, out_features=512, bias=False)
            (to_kv): Linear(in_features=1024, out_features=1024, bias=False)
            (to_out): Linear(in_features=512, out_features=4096, bias=False)
          )
          (ff): Sequential(
            (0): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
            (1): Linear(in_features=4096, out_features=16384, bias=False)
            (2): GELU(approximate='none')
            (3): Linear(in_features=16384, out_features=4096, bias=False)
          )
        )
        (decoder_layer): LlamaDecoderLayer(
          (self_attn): LlamaAttention(
            (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (rotary_emb): LlamaRotaryEmbedding()
          )
          (mlp): LlamaMLP(
            (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
            (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (act_fn): SiLUActivation()
          )
          (input_layernorm): LlamaRMSNorm()
          (post_attention_layernorm): LlamaRMSNorm()
        )
      )
      (28-30): 3 x FlamingoLayer(
        (decoder_layer): LlamaDecoderLayer(
          (self_attn): LlamaAttention(
            (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (rotary_emb): LlamaRotaryEmbedding()
          )
          (mlp): LlamaMLP(
            (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
            (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (act_fn): SiLUActivation()
          )
          (input_layernorm): LlamaRMSNorm()
          (post_attention_layernorm): LlamaRMSNorm()
        )
      )
      (31): FlamingoLayer(
        (gated_cross_attn_layer): GatedCrossAttentionBlock(
          (attn): MaskedCrossAttention(
            (norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
            (to_q): Linear(in_features=4096, out_features=512, bias=False)
            (to_kv): Linear(in_features=1024, out_features=1024, bias=False)
            (to_out): Linear(in_features=512, out_features=4096, bias=False)
          )
          (ff): Sequential(
            (0): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
            (1): Linear(in_features=4096, out_features=16384, bias=False)
            (2): GELU(approximate='none')
            (3): Linear(in_features=16384, out_features=4096, bias=False)
          )
        )
        (decoder_layer): LlamaDecoderLayer(
          (self_attn): LlamaAttention(
            (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (rotary_emb): LlamaRotaryEmbedding()
          )
          (mlp): LlamaMLP(
            (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
            (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (act_fn): SiLUActivation()
          )
          (input_layernorm): LlamaRMSNorm()
          (post_attention_layernorm): LlamaRMSNorm()
        )
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=32003, bias=False)
  (gated_cross_attn_layers): ModuleList(
    (0-2): 3 x None
    (3): GatedCrossAttentionBlock(
      (attn): MaskedCrossAttention(
        (norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (to_q): Linear(in_features=4096, out_features=512, bias=False)
        (to_kv): Linear(in_features=1024, out_features=1024, bias=False)
        (to_out): Linear(in_features=512, out_features=4096, bias=False)
      )
      (ff): Sequential(
        (0): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (1): Linear(in_features=4096, out_features=16384, bias=False)
        (2): GELU(approximate='none')
        (3): Linear(in_features=16384, out_features=4096, bias=False)
      )
    )
    (4-6): 3 x None
    (7): GatedCrossAttentionBlock(
      (attn): MaskedCrossAttention(
        (norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (to_q): Linear(in_features=4096, out_features=512, bias=False)
        (to_kv): Linear(in_features=1024, out_features=1024, bias=False)
        (to_out): Linear(in_features=512, out_features=4096, bias=False)
      )
      (ff): Sequential(
        (0): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (1): Linear(in_features=4096, out_features=16384, bias=False)
        (2): GELU(approximate='none')
        (3): Linear(in_features=16384, out_features=4096, bias=False)
      )
    )
    (8-10): 3 x None
    (11): GatedCrossAttentionBlock(
      (attn): MaskedCrossAttention(
        (norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (to_q): Linear(in_features=4096, out_features=512, bias=False)
        (to_kv): Linear(in_features=1024, out_features=1024, bias=False)
        (to_out): Linear(in_features=512, out_features=4096, bias=False)
      )
      (ff): Sequential(
        (0): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (1): Linear(in_features=4096, out_features=16384, bias=False)
        (2): GELU(approximate='none')
        (3): Linear(in_features=16384, out_features=4096, bias=False)
      )
    )
    (12-14): 3 x None
    (15): GatedCrossAttentionBlock(
      (attn): MaskedCrossAttention(
        (norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (to_q): Linear(in_features=4096, out_features=512, bias=False)
        (to_kv): Linear(in_features=1024, out_features=1024, bias=False)
        (to_out): Linear(in_features=512, out_features=4096, bias=False)
      )
      (ff): Sequential(
        (0): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (1): Linear(in_features=4096, out_features=16384, bias=False)
        (2): GELU(approximate='none')
        (3): Linear(in_features=16384, out_features=4096, bias=False)
      )
    )
    (16-18): 3 x None
    (19): GatedCrossAttentionBlock(
      (attn): MaskedCrossAttention(
        (norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (to_q): Linear(in_features=4096, out_features=512, bias=False)
        (to_kv): Linear(in_features=1024, out_features=1024, bias=False)
        (to_out): Linear(in_features=512, out_features=4096, bias=False)
      )
      (ff): Sequential(
        (0): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (1): Linear(in_features=4096, out_features=16384, bias=False)
        (2): GELU(approximate='none')
        (3): Linear(in_features=16384, out_features=4096, bias=False)
      )
    )
    (20-22): 3 x None
    (23): GatedCrossAttentionBlock(
      (attn): MaskedCrossAttention(
        (norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (to_q): Linear(in_features=4096, out_features=512, bias=False)
        (to_kv): Linear(in_features=1024, out_features=1024, bias=False)
        (to_out): Linear(in_features=512, out_features=4096, bias=False)
      )
      (ff): Sequential(
        (0): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (1): Linear(in_features=4096, out_features=16384, bias=False)
        (2): GELU(approximate='none')
        (3): Linear(in_features=16384, out_features=4096, bias=False)
      )
    )
    (24-26): 3 x None
    (27): GatedCrossAttentionBlock(
      (attn): MaskedCrossAttention(
        (norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (to_q): Linear(in_features=4096, out_features=512, bias=False)
        (to_kv): Linear(in_features=1024, out_features=1024, bias=False)
        (to_out): Linear(in_features=512, out_features=4096, bias=False)
      )
      (ff): Sequential(
        (0): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (1): Linear(in_features=4096, out_features=16384, bias=False)
        (2): GELU(approximate='none')
        (3): Linear(in_features=16384, out_features=4096, bias=False)
      )
    )
    (28-30): 3 x None
    (31): GatedCrossAttentionBlock(
      (attn): MaskedCrossAttention(
        (norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (to_q): Linear(in_features=4096, out_features=512, bias=False)
        (to_kv): Linear(in_features=1024, out_features=1024, bias=False)
        (to_out): Linear(in_features=512, out_features=4096, bias=False)
      )
      (ff): Sequential(
        (0): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (1): Linear(in_features=4096, out_features=16384, bias=False)
        (2): GELU(approximate='none')
        (3): Linear(in_features=16384, out_features=4096, bias=False)
      )
    )
  )
)
```