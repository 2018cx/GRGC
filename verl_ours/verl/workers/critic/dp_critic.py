# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement a multiprocess PPOCritic
"""

import itertools
import logging
import os
import numpy as np
import torch
import torch.distributed
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import time

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.critic import BasePPOCritic

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def _numpy_to_torch(x, device, dtype):
    # x can be np.ndarray or np.scalar
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    else:
        # np.float64 / Python float / etc.
        t = torch.tensor(x)
    return t.to(device=device, dtype=dtype)

def beta_ppf_transform(values: torch.Tensor, q: float = 0.01, vmin: float = -10, vmax: float = 20, eps: float = 1e-12) -> torch.Tensor:
    """
    values: arbitrary shape tensor
    Returns: same shape tensor after Beta PPF transform:
        Beta.ppf(q, a=0.5 + values - vmin, b=0.5 - values + vmax)
    """
    if not torch.is_tensor(values):
        raise TypeError("values must be a torch.Tensor")

    # a,b construction
    a = 0.5 + (values - vmin)
    b = 0.5 + (vmax - values)

    # numerical safety: clamp to strictly positive
    a = a.clamp_min(eps)
    b = b.clamp_min(eps)

    q_t = torch.full_like(values, float(q))

    try:
        beta_dist = torch.distributions.Beta(a, b)
        out = beta_dist.icdf(q_t)
        out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
        return out
    except Exception:
        values_cpu = values.detach().float().cpu()
        a_cpu = a.detach().float().cpu()
        b_cpu = b.detach().float().cpu()
        from scipy.stats import beta as sp_beta
        out_np = sp_beta.ppf(float(q), a_cpu.numpy(), b_cpu.numpy())
        out_np = np.nan_to_num(out_np, nan=0.0, posinf=1.0, neginf=0.0)
        return _numpy_to_torch(out_np, device=values.device, dtype=values.dtype)


def apply_reward_transform(
    values: torch.Tensor,
    transform: str = "beta",
    index: torch.Tensor = None,
    **kwargs,
) -> torch.Tensor:
    """
    Apply configurable transform to critic outputs before GRPO.
    Aims to provide stronger learning signal for multi-round RL.

    transform options:
      - "identity": no transform, raw critic output (GRPO will normalize)
      - "linear": (x - vmin) / (vmax - vmin), batch-adaptive vmin/vmax
      - "linear_fixed": same with fixed vmin/vmax
      - "beta": Beta PPF transform (original behavior)
      - "beta_logit": Beta PPF then logit-tail expansion
      - "temperature": x / temperature (temp<1 sharpens, temp>1 softens)
      - "group_zscore": per-group z-score (needs index)
      - "group_power": per-group z-score then sign(z)*|z|^gamma (needs index)
      - "group_sinh": per-group z-score then sinh(alpha*z) (needs index)
      - "top1_margin_boost": only boost group-best by lambda*(top1-top2) (needs index)
      - "rank_gaussian": replace by group rank mapped to Gaussian quantiles (needs index)
    """
    eps = kwargs.get("eps", 1e-8)
    work = values.float()

    def _group_zscore_1d(scores_1d: torch.Tensor, gidx: torch.Tensor):
        out = torch.zeros_like(scores_1d)
        for gid in torch.unique(gidx):
            mask = gidx == gid
            s = scores_1d[mask]
            if s.numel() < 2:
                out[mask] = 0.0
            else:
                mean_g = s.mean()
                std_g = s.std(unbiased=False).clamp_min(eps)
                out[mask] = (s - mean_g) / std_g
        return out

    def _normal_quantiles(k: int, device, dtype):
        if k <= 0:
            return torch.empty((0,), device=device, dtype=dtype)
        i = torch.arange(1, k + 1, device=device, dtype=dtype)
        p = (i - 0.5) / k
        z = (2.0**0.5) * torch.erfinv(2.0 * p - 1.0)
        return z

    if transform == "identity":
        return values

    if transform == "linear" or transform == "linear_fixed":
        vmin = kwargs.get("vmin", -10.0)
        vmax = kwargs.get("vmax", 20.0)
        if transform == "linear":
            vmin = float(work.amin().item())
            vmax = float(work.amax().item())
        span = vmax - vmin + eps
        out = (work - vmin) / span
        return out.to(values.dtype)

    if transform == "beta":
        q = kwargs.get("beta_q", 0.01)
        vmin = kwargs.get("vmin", -10.0)
        vmax = kwargs.get("vmax", 20.0)
        out = beta_ppf_transform(work, q=q, vmin=vmin, vmax=vmax, eps=eps)
        baseline = beta_ppf_transform(
            torch.zeros((), device=values.device, dtype=work.dtype),
            q=q, vmin=vmin, vmax=vmax, eps=eps
        )
        return (out - baseline).to(values.dtype)

    if transform == "beta_logit":
        q = kwargs.get("beta_q", 0.01)
        vmin = kwargs.get("vmin", -10.0)
        vmax = kwargs.get("vmax", 20.0)
        logit_eps = kwargs.get("beta_logit_eps", 1e-5)
        scale = kwargs.get("beta_logit_scale", 1.0)
        out = beta_ppf_transform(work, q=q, vmin=vmin, vmax=vmax, eps=eps)
        out = out.clamp(logit_eps, 1.0 - logit_eps)
        out = torch.log(out) - torch.log1p(-out)
        return (out * float(scale)).to(values.dtype)

    if transform == "temperature":
        temp = kwargs.get("temperature", 0.5)
        return (work / max(float(temp), eps)).to(values.dtype)

    if transform == "group_zscore":
        if index is None:
            return values  # fallback
        scores_1d = work if work.dim() == 1 else work.sum(dim=-1)
        out = _group_zscore_1d(scores_1d, index.long())
        return out.to(values.dtype)

    if transform == "group_power":
        if index is None:
            return values
        gamma = float(kwargs.get("power_gamma", 1.5)) #2.0
        scores_1d = work if work.dim() == 1 else work.sum(dim=-1)
        z = _group_zscore_1d(scores_1d, index.long())
        out = torch.sign(z) * torch.pow(torch.abs(z), gamma)
        return out.to(values.dtype)

    if transform == "group_sinh":
        if index is None:
            return values
        alpha = float(kwargs.get("sinh_alpha", 1.5))
        scores_1d = work if work.dim() == 1 else work.sum(dim=-1)
        z = _group_zscore_1d(scores_1d, index.long())
        out = torch.sinh(alpha * z)
        return out.to(values.dtype)

    if transform == "top1_margin_boost":
        if index is None:
            return values
        boost_lambda = float(kwargs.get("top1_boost_lambda", 1.0))
        scores_1d = (work if work.dim() == 1 else work.sum(dim=-1)).clone()
        gidx = index.long()
        for gid in torch.unique(gidx):
            mask = gidx == gid
            s = scores_1d[mask]
            if s.numel() < 2:
                continue
            top2_vals, top2_idx = torch.topk(s, k=2, largest=True)
            margin = (top2_vals[0] - top2_vals[1]).clamp_min(0.0)
            global_idx = torch.where(mask)[0][top2_idx[0]]
            scores_1d[global_idx] = scores_1d[global_idx] + boost_lambda * margin
        return scores_1d.to(values.dtype)

    if transform == "rank_gaussian":
        if index is None:
            return values
        rank_scale = float(kwargs.get("rank_scale", 1.0))
        scores_1d = work if work.dim() == 1 else work.sum(dim=-1)
        out = torch.zeros_like(scores_1d)
        gidx = index.long()
        for gid in torch.unique(gidx):
            mask = gidx == gid
            s = scores_1d[mask]
            k = int(s.numel())
            if k <= 1:
                out[mask] = 0.0
                continue
            order = torch.argsort(s, dim=0)
            q = _normal_quantiles(k, device=s.device, dtype=s.dtype) * rank_scale
            out_group = torch.empty_like(s)
            out_group[order] = q
            out[mask] = out_group
        return out.to(values.dtype)

    return values

class DataParallelPPOCritic(BasePPOCritic):
    def __init__(self, config, critic_module: nn.Module, critic_optimizer: optim.Optimizer):
        super().__init__(config=config)
        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer
        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        print(f"Critic use_remove_padding={self.use_remove_padding}")

        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        self.device_name = get_device_name()

    def _forward_micro_batch(self, micro_batch, compute_teacher):
        if compute_teacher:
            response_length = micro_batch["teacher_response"].size(-1)
        else:
            response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            if compute_teacher:
                input_ids = micro_batch["teacher_input_ids"]
                batch, seqlen = input_ids.shape
                attention_mask = micro_batch["teacher_attention_mask"]
                position_ids = micro_batch["teacher_position_ids"]
            else:
                input_ids = micro_batch["input_ids"]
                batch, seqlen = input_ids.shape
                attention_mask = micro_batch["attention_mask"]
                position_ids = micro_batch["position_ids"]
            
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.critic_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating

                if hasattr(self.critic_module, "v_head"):
                    # For trl.AutoModelForCausalLMWithValueHead
                    values_rmpad = output[2].squeeze(0).unsqueeze(-1)
                else:
                    values_rmpad = output.logits
                    values_rmpad = values_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    values_rmpad = gather_outpus_and_unpad(values_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # pad it back
                values = pad_input(values_rmpad, indices=indices, batch=batch, seqlen=seqlen).squeeze(-1)
                
                values = values[:, -response_length:] # NOTE: the line above is for critic usage, that we predict the value of *next* token. but since we hack it for reward model, we change it to predict the value of the *current* token.
                response_mask = attention_mask[:, -response_length:]
                response_lengths = response_mask.sum(dim=1).long()
                last_token_indices = response_lengths - 1
                last_token_mask = torch.zeros_like(response_mask, dtype=torch.bool)
                batch_indices = torch.arange(response_mask.size(0), device=response_mask.device)
                last_token_mask[batch_indices, last_token_indices] = True
                values = values * last_token_mask.type_as(values)
            else:
                output = self.critic_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                if hasattr(self.critic_module, "v_head"):
                    # For trl.AutoModelForCausalLMWithValueHead
                    values = output[2]
                else:
                    values = output.logits
                values = values[:, -response_length:].squeeze(-1) # NOTE: the line above is for critic usage, that we predict the value of *next* token. but since we hack it for reward model, we change it to predict the value of the *current* token.
                response_mask = attention_mask[:, -response_length:]
                response_lengths = response_mask.sum(dim=1).long()
                last_token_indices = response_lengths - 1
                last_token_mask = torch.zeros_like(response_mask, dtype=torch.bool)
                batch_indices = torch.arange(response_mask.size(0), device=response_mask.device)
                last_token_mask[batch_indices, last_token_indices] = True
                values = values * last_token_mask.type_as(values)
            return values

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.critic_module, FSDP):
            grad_norm = self.critic_module.clip_grad_norm_(self.config.grad_clip)
        elif isinstance(self.critic_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.critic_optimizer.zero_grad()
        else:
            self.critic_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp critic", logger=logger)
    def compute_values(self, data: DataProto) -> torch.Tensor:
        compute_teacher = False # currently we only compute student values
        self.critic_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        if compute_teacher:
            select_keys = ["teacher_response", "teacher_input_ids", "teacher_attention_mask", "teacher_position_ids"]
        else:
            select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "index"]
        batch = data.select(batch_keys=select_keys).batch
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len, compute_teacher=compute_teacher)
        else:
            micro_batches = batch.split(micro_batch_size)

        values_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            with torch.no_grad():
                values = self._forward_micro_batch(micro_batch, compute_teacher=compute_teacher)
            values_lst.append(values)
        values = torch.concat(values_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == values.size(0), f"{len(indices)} vs. {values.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            values = values[revert_indices]

        if compute_teacher:
            responses = data.batch["teacher_response"]
            attention_mask = data.batch["teacher_attention_mask"]
        else:
            responses = data.batch["responses"]
            attention_mask = data.batch["attention_mask"]
        response_length = responses.size(1)
        response_mask = attention_mask[:, -response_length:]
        values = values * response_mask  # Only action tokens have values

        # Reward transformation before GRPO. Keep current default behavior.
        reward_transform = getattr(self.config, "reward_transform", "beta")
        if reward_transform is not None and reward_transform != "none":
            response_lengths = response_mask.sum(dim=1).long()
            valid = response_lengths > 0
            seq_scores = masked_sum(values, response_mask, axis=-1)
            group_index = data.batch["index"].to(seq_scores.device).long() if "index" in data.batch.keys() else None
            seq_scores_t = apply_reward_transform(
                seq_scores,
                transform=reward_transform,
                index=group_index,
                beta_q=float(getattr(self.config, "reward_transform_beta_q", 0.01)),
                vmin=float(getattr(self.config, "reward_transform_vmin", -10.0)),
                vmax=float(getattr(self.config, "reward_transform_vmax", 20.0)),
                temperature=float(getattr(self.config, "reward_transform_temperature", 0.5)),
                power_gamma=float(getattr(self.config, "reward_transform_power_gamma", 1.2)), #1.5
                sinh_alpha=float(getattr(self.config, "reward_transform_sinh_alpha", 1.5)),
                top1_boost_lambda=float(getattr(self.config, "reward_transform_top1_boost_lambda", 1.0)),
                rank_scale=float(getattr(self.config, "reward_transform_rank_scale", 1.0)),
                beta_logit_eps=float(getattr(self.config, "reward_transform_beta_logit_eps", 1e-5)),
                beta_logit_scale=float(getattr(self.config, "reward_transform_beta_logit_scale", 1.0)),
            )
            seq_scores_t = torch.where(valid, seq_scores_t, torch.zeros_like(seq_scores_t))

            # write transformed sequence score back to the last valid token
            values = torch.zeros_like(values)
            bidx = torch.arange(values.size(0), device=values.device)
            last_idx = (response_lengths - 1).clamp_min(0)
            values[bidx, last_idx] = seq_scores_t
            values = values * response_mask
        return values

    @GPUMemoryLogger(role="dp critic", logger=logger)
    def update_critic(self, data: DataProto):
        # make sure we are in training mode
        self.critic_module.train()
        metrics = {}

        tensor_select_keys = [
            "input_ids", "responses", "attention_mask", "position_ids",
            "teacher_input_ids", "teacher_response", "teacher_attention_mask", "teacher_position_ids",
            "index",  # <<< NEW: prompt-group id tensor
        ]

        non_tensor_select_keys = []
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")

        ot_coef = float(getattr(self.config, "critic_ot_coef", 0.01))
        ot_target_mode = getattr(self.config, "critic_ot_target_mode", "gaussian")
        ot_target_alpha = float(getattr(self.config, "critic_ot_target_alpha", 1.0))
        ot_loss_type = getattr(self.config, "critic_ot_loss_type", "w2")
        ot_min_group_size = int(getattr(self.config, "critic_ot_min_group_size", 2))
        bsz = int(data.batch.batch_size[0])
        mini = int(self.config.ppo_mini_batch_size)
        num_mini_batches = max(1, (bsz + mini - 1) // mini)

        if len(non_tensor_select_keys) > 0:
            dataloader = data.select(tensor_select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = data.select(batch_keys=tensor_select_keys).chunk(num_mini_batches)

        for epoch in range(self.config.ppo_epochs):
            for batch_idx, mini_dp in enumerate(dataloader):

                if has_multi_modal_inputs:
                    micro = int(self.config.ppo_micro_batch_size_per_gpu)
                    mb = int(mini_dp.batch.batch_size[0])
                    num_micro_batches = max(1, (mb + micro - 1) // micro)
                    micro_batches = mini_dp.chunk(num_micro_batches)
                    self.gradient_accumulation = max(1, int(self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu))
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(
                        batch=mini_dp.batch.select(*tensor_select_keys),
                        max_token_len=max_token_len,
                        compute_teacher=False,
                    )
                    self.gradient_accumulation = 1  
                else:
                    micro = int(self.config.ppo_micro_batch_size_per_gpu)
                    mb = int(mini_dp.batch.batch_size[0])
                    num_micro_batches = max(1, (mb + micro - 1) // micro)
                    micro_batches = mini_dp.chunk(num_micro_batches)
                    self.gradient_accumulation = max(1, int(self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu))

                self.critic_optimizer.zero_grad()

                for micro_item in micro_batches:
                    if isinstance(micro_item, DataProto):
                        micro = {**micro_item.batch.to(get_device_id()), **micro_item.non_tensor_batch}
                    else:
                        micro = micro_item.to(get_device_id())

                    # student tensors
                    responses = micro["responses"]
                    attention_mask = micro["attention_mask"]
                    response_length = responses.size(1)
                    response_mask = attention_mask[:, -response_length:]

                    # teacher tensors
                    teacher_response = micro["teacher_response"]
                    teacher_attention_mask = micro["teacher_attention_mask"]
                    teacher_response_length = teacher_response.size(1)
                    teacher_response_mask = teacher_attention_mask[:, -teacher_response_length:]

                    student_vpreds = self._forward_micro_batch(micro, compute_teacher=False)
                    teacher_vpreds = self._forward_micro_batch(micro, compute_teacher=True)

                    # outcome reward (scalar per response)
                    student_reward = masked_sum(student_vpreds, response_mask, axis=-1)          # (B,)
                    teacher_reward = masked_sum(teacher_vpreds, teacher_response_mask, axis=-1)  # (B,)

                    d_acc = (teacher_reward > student_reward).float().mean().detach().item()

                    d_loss = core_algos.compute_discriminator_loss(
                        student_vpreds=student_vpreds,
                        teacher_vpreds=teacher_vpreds,
                        response_mask=response_mask,
                        teacher_response_mask=teacher_response_mask,
                    )

                    ot_loss = None
                    if ot_coef > 0.0:
                        if "index" not in micro:
                            print("error")
                            input()
                        else:
                            group_index = micro["index"].to(student_reward.device).long()
                            ot_loss = core_algos.groupwise_1d_ot_calibration_loss(
                                scores=student_reward,
                                group_index=group_index,
                                target_mode=ot_target_mode,
                                target_alpha=ot_target_alpha,
                                loss_type=ot_loss_type,
                                min_group_size=ot_min_group_size,
                            )

                    anchor_loss = None

                    d_loss_total = d_loss if ot_loss is None else (d_loss + ot_coef * ot_loss)
                    if anchor_loss is not None:
                        d_loss_total = d_loss_total + anchor_coef * anchor_loss

                    # -----------------------------
                    # (H) backward & step
                    # -----------------------------
                    if self.config.use_dynamic_bsz:
                        loss = d_loss_total * (len(micro_item) / self.config.ppo_mini_batch_size)
                    else:
                        loss = d_loss_total / self.gradient_accumulation

                    loss.backward()

                    log_data = {
                        "critic/d_loss": float(d_loss.detach().item()),
                        "critic/d_acc": float(d_acc),
                        "critic/student_value_mean": float(student_reward.mean().detach().item()),
                        "critic/teacher_value_mean": float(teacher_reward.mean().detach().item()),
                    }
                    if ot_loss is not None:
                        log_data["critic/ot_loss"] = float(ot_loss.detach().item())
                        log_data["critic/d_loss_total"] = float(d_loss_total.detach().item())
                    
                    if anchor_loss is not None:
                        log_data["critic/anchor_loss"] = float(anchor_loss.detach().item())

                    append_to_dict(metrics, log_data)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"critic/grad_norm": float(grad_norm.detach().item())})

        self.critic_optimizer.zero_grad()
        return metrics

