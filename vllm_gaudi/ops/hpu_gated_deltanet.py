# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""HPU-specific Qwen3.5 GatedDeltaNet layer and Qwen3Next MoE patch."""

import torch
from vllm.model_executor.models.qwen3_5 import Qwen3_5GatedDeltaNet
from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.distributed import tensor_model_parallel_all_gather

from einops import rearrange
from vllm.forward_context import ForwardContext, get_forward_context

from vllm_gaudi.ops.causal_conv1d_pytorch import (
    hpu_causal_conv1d_fn,
    hpu_causal_conv1d_update,
)
from vllm_gaudi.ops.hpu_gdn_pytorch import (
    hpu_chunk_gated_delta_rule,
    hpu_fused_gdn_gating,
    hpu_fused_recurrent_gated_delta_rule,
)
from vllm_gaudi.v1.attention.backends.hpu_attn import HPUAttentionMetadataV1


# ---------------------------------------------------------------------------
# Qwen3NextSparseMoeBlock forward patch
# ---------------------------------------------------------------------------
# HPU requires 3D->2D reshape before routing and handles SharedFusedMoE
# tuple returns (shared_out, fused_out).
def _hpu_qwen3next_sparse_moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    orig_shape = hidden_states.shape
    hidden_dim = orig_shape[-1]
    hidden_states = hidden_states.reshape(-1, hidden_dim)
    num_tokens = hidden_states.shape[0]

    if self.is_sequence_parallel:
        hidden_states = sequence_parallel_chunk(hidden_states)

    if self.experts.is_internal_router:
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=hidden_states
        )
    else:
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

    if self.shared_expert is not None:
        final_hidden_states = final_hidden_states[0] + final_hidden_states[1]

    if self.is_sequence_parallel:
        final_hidden_states = tensor_model_parallel_all_gather(final_hidden_states, 0)
        final_hidden_states = final_hidden_states[:num_tokens]
    elif self.tp_size > 1:
        final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
            final_hidden_states
        )

    return final_hidden_states.reshape(orig_shape)


if not getattr(Qwen3NextSparseMoeBlock, "_hpu_shape_patch_applied", False):
    Qwen3NextSparseMoeBlock.forward = _hpu_qwen3next_sparse_moe_forward
    Qwen3NextSparseMoeBlock._hpu_shape_patch_applied = True


# ---------------------------------------------------------------------------
# HPU GatedDeltaNet layer
# ---------------------------------------------------------------------------
class HPUQwen3_5GatedDeltaNet(Qwen3_5GatedDeltaNet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Assigned by model runner per KV cache group for hybrid GDN models.
        self.cache_group_idx: int | None = None
        # Use configured chunk size when explicitly set; otherwise default to
        # 128 to match HPU prompt bucket alignment.
        hf_text_config = getattr(self.model_config, "hf_text_config", None)
        has_explicit_chunk_size = (
            hf_text_config is not None
            and (
                getattr(hf_text_config, "mamba_chunk_size", None) is not None
                or getattr(hf_text_config, "chunk_size", None) is not None
            )
        )
        self.mamba_chunk_size = (
            self.model_config.get_mamba_chunk_size()
            if has_explicit_chunk_size
            else 128
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        num_tokens = hidden_states.size(0)

        # Prompt buckets on HPU can include padded tokens. Mask once before
        # projections so q/k/v/b/a/z for padded rows are all zero.
        attn_metadata = get_forward_context().attn_metadata

        if attn_metadata is not None and bool(getattr(attn_metadata, "is_prompt", False)):
            padding_mask_flat = getattr(attn_metadata, "padding_mask_flat", None)
            if (
                padding_mask_flat is not None
                and padding_mask_flat.numel() == hidden_states.size(0)
            ):
                hidden_mask = padding_mask_flat.view(-1, 1).to(dtype=hidden_states.dtype)
                hidden_states = hidden_states * hidden_mask

        # Part 1: Input Projection
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        z_size = self.value_dim // self.tp_size
        mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        ba, _ = self.in_proj_ba(hidden_states)

        b, a = ba.chunk(2, dim=-1)
        b = b.contiguous()
        a = a.contiguous()

        # Part 2: Core Attention (Custom Op)
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self.gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            self.prefix,
        )

        # Part 3: Output Projection
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output[:num_tokens], _ = self.out_proj(core_attn_out)

    def gdn_attention_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        layer_name: str,
    ) -> None:
        forward_context: ForwardContext = get_forward_context()
        self_layer = forward_context.no_compile_layers[layer_name]

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            # V1 profile run
            return

        self._forward_core_hpu(
            self_layer,
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            core_attn_out=core_attn_out,
            attn_metadata=attn_metadata,
        )

    def _forward_core_hpu(
        self,
        self_layer,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: HPUAttentionMetadataV1,
    ):
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if attn_metadata is None:
            return

        has_initial_state = attn_metadata.has_initial_states_p
        non_spec_query_start_loc = attn_metadata.query_start_loc_p
        padding_mask_flat = getattr(attn_metadata, "padding_mask_flat", None)
        # Speculative decode placeholders (not implemented in phase 1)
        spec_sequence_masks = None
        non_spec_state_indices_tensor = attn_metadata.state_indices_tensor

        self_kv_cache = self.kv_cache[forward_context.virtual_engine]
        conv_state = self_kv_cache[0]

        if non_spec_state_indices_tensor is not None and non_spec_state_indices_tensor.dim() > 1:
            cache_group_idx = getattr(self, "cache_group_idx", None)
            assert cache_group_idx is not None, (
                "HPUQwen3_5GatedDeltaNet requires linear_attn.cache_group_idx when "
                "state_indices_tensor is 2D; ensure model runner assigns "
                "layer.linear_attn.cache_group_idx per KV cache group."
            )
            assert 0 <= int(cache_group_idx) < non_spec_state_indices_tensor.size(0), (
                f"Invalid cache_group_idx={cache_group_idx} for "
                f"state_indices_tensor rows={non_spec_state_indices_tensor.size(0)}"
            )
            non_spec_state_indices_tensor = non_spec_state_indices_tensor[int(cache_group_idx)]

        ssm_state = self_kv_cache[1]

        num_actual_tokens = mixed_qkv.size(0)
        is_prompt = bool(attn_metadata.is_prompt)
        token_mask_flat: torch.Tensor | None = None
        chunk_query_start_loc = non_spec_query_start_loc

        if is_prompt and non_spec_query_start_loc is not None:
            try:
                num_actual_tokens = int(non_spec_query_start_loc[-1].item())
            except Exception:
                pass

        if is_prompt and padding_mask_flat is not None:
            token_mask_flat = padding_mask_flat.view(-1, 1).to(dtype=mixed_qkv.dtype)

            if non_spec_state_indices_tensor is not None:
                num_rows = int(non_spec_state_indices_tensor.numel())
                total_tokens = int(mixed_qkv.size(0))
                if num_rows > 0 and total_tokens % num_rows == 0:
                    padded_seq_len = total_tokens // num_rows
                    chunk_query_start_loc = torch.arange(
                        0,
                        (num_rows + 1) * padded_seq_len,
                        padded_seq_len,
                        device=mixed_qkv.device,
                        dtype=non_spec_query_start_loc.dtype if non_spec_query_start_loc is not None else torch.int32,
                    )

        num_prefills = 1 if is_prompt else 0
        num_decodes = 0 if is_prompt else 1

        if not is_prompt:
            mixed_qkv = mixed_qkv[:num_actual_tokens]
            b = b[:num_actual_tokens]
            a = a[:num_actual_tokens]

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        mixed_qkv_spec = None
        mixed_qkv_non_spec = mixed_qkv

        assert self.cache_config is not None
        mamba_block_size = self.cache_config.mamba_block_size

        if num_prefills > 0:
            mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)

            mixed_qkv_non_spec = hpu_causal_conv1d_fn(
                x=mixed_qkv_non_spec_T,
                weight=conv_weights,
                bias=self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                block_size_to_align=mamba_block_size,
                metadata=attn_metadata,
                is_prompt=True,
            ).transpose(0, 1)
            if token_mask_flat is not None:
                mixed_qkv_non_spec = mixed_qkv_non_spec * token_mask_flat
        elif num_decodes > 0:
            mixed_qkv_non_spec = hpu_causal_conv1d_update(
                x=mixed_qkv_non_spec,
                conv_state=conv_state,
                weight=conv_weights,
                bias=self.conv1d.bias,
                activation=self.activation,
                conv_state_indices=non_spec_state_indices_tensor[
                    :num_actual_tokens
                ],
                query_start_loc=non_spec_query_start_loc,
                validate_data=False,
            )
        else:
            mixed_qkv_non_spec = None

        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
            mixed_qkv_non_spec
        )

        g, beta = hpu_fused_gdn_gating(self.A_log, a, b, self.dt_bias)
        if token_mask_flat is not None:
            token_mask_h = token_mask_flat.view(1, -1, 1).to(dtype=g.dtype)
            g = g * token_mask_h
            beta = beta * token_mask_h

        # Phase 1: no speculative decode, use non-spec tensors directly
        g_non_spec = g
        beta_non_spec = beta

        # 2. Recurrent attention
        if is_prompt:
            initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()
            initial_state[~has_initial_state, ...] = 0

            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = hpu_chunk_gated_delta_rule(
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                g=g_non_spec,
                beta=beta_non_spec,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=chunk_query_start_loc,
                use_qk_l2norm_in_kernel=True,
                chunk_size=self.mamba_chunk_size,
            )
            ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(
                device=ssm_state.device,
                dtype=ssm_state.dtype,
            )
        elif num_decodes > 0:
            core_attn_out_non_spec, last_recurrent_state = (
                hpu_fused_recurrent_gated_delta_rule(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g_non_spec,
                    beta=beta_non_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=non_spec_query_start_loc[:num_decodes + 1],
                    ssm_state_indices=non_spec_state_indices_tensor,
                    use_qk_l2norm_in_kernel=True,
                )
            )
        else:
            core_attn_out_non_spec, last_recurrent_state = None, None

        # 3. Merge core attention output
        if core_attn_out_non_spec is not None:
            non_spec_out = core_attn_out_non_spec.squeeze(0)
            if non_spec_out.shape[0] == core_attn_out.shape[0]:
                core_attn_out.copy_(non_spec_out)
            else:
                n = min(num_actual_tokens, non_spec_out.shape[0], core_attn_out.shape[0])
                core_attn_out[:n] = non_spec_out[:n]

        if token_mask_flat is not None:
            core_attn_out.mul_(token_mask_flat.view(-1, 1, 1))
