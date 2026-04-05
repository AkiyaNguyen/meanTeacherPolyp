import math
import re

import torch
import torch.nn as nn


class VectorLoRA_QKV(nn.Module):
	"""
	LoRA wrapper for a fused qkv projection layer.

	This module keeps the original qkv path intact and adds low-rank updates
	only to Q and V. K remains strictly frozen (no LoRA branch).
	"""

	def __init__(self, qkv: nn.Linear, rank: int, alpha: float = 16.0) -> None:
		super().__init__()
		if rank <= 0:
			raise ValueError(f"rank must be > 0, got {rank}")
		if not isinstance(qkv, nn.Linear):
			raise TypeError("qkv must be an nn.Linear layer")
		if qkv.out_features % 3 != 0:
			raise ValueError(
				f"qkv.out_features must be divisible by 3, got {qkv.out_features}"
			)

		self.qkv = qkv
		self.rank = int(rank)
		self.alpha = float(alpha)
		self.scale = self.alpha / self.rank

		embed_dim = qkv.in_features
		qkv_chunk = qkv.out_features // 3
		if qkv_chunk != embed_dim:
			raise ValueError(
				"Expected fused qkv output layout [Q, K, V] with each chunk equal "
				f"to embed_dim. Got in_features={embed_dim}, out_features={qkv.out_features}."
			)

		self.embed_dim = embed_dim
		base_device = qkv.weight.device
		base_dtype = qkv.weight.dtype

		# Freeze original qkv parameters.
		for p in self.qkv.parameters():
			p.requires_grad = False

		# LoRA for Q: x @ A_q^T @ B_q^T
		self.lora_q_A = nn.Parameter(torch.empty(self.rank, self.embed_dim, device=base_device, dtype=base_dtype))
		self.lora_q_B = nn.Parameter(torch.empty(self.embed_dim, self.rank, device=base_device, dtype=base_dtype))

		# LoRA for V: x @ A_v^T @ B_v^T
		self.lora_v_A = nn.Parameter(torch.empty(self.rank, self.embed_dim, device=base_device, dtype=base_dtype))
		self.lora_v_B = nn.Parameter(torch.empty(self.embed_dim, self.rank, device=base_device, dtype=base_dtype))

		self.reset_parameters()

	def reset_parameters(self) -> None:
		nn.init.kaiming_uniform_(self.lora_q_A, a=math.sqrt(5))
		nn.init.zeros_(self.lora_q_B)
		nn.init.kaiming_uniform_(self.lora_v_A, a=math.sqrt(5))
		nn.init.zeros_(self.lora_v_B)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# Base fused qkv projection: [..., 3 * embed_dim]
		qkv_out = self.qkv(x)

		# Low-rank updates for Q and V only.
		q_update = (x @ self.lora_q_A.t()) @ self.lora_q_B.t()
		v_update = (x @ self.lora_v_A.t()) @ self.lora_v_B.t()

		qkv_out[..., : self.embed_dim] += self.scale * q_update
		qkv_out[..., 2 * self.embed_dim :] += self.scale * v_update

		return qkv_out


class VectorLoRA_Linear(nn.Module):
	"""LoRA wrapper for a single linear projection (used for query/value fallback)."""

	def __init__(self, linear: nn.Linear, rank: int, alpha: float = 16.0) -> None:
		super().__init__()
		if rank <= 0:
			raise ValueError(f"rank must be > 0, got {rank}")
		if not isinstance(linear, nn.Linear):
			raise TypeError("linear must be an nn.Linear layer")

		self.linear = linear
		self.rank = int(rank)
		self.alpha = float(alpha)
		self.scale = self.alpha / self.rank
		self.in_dim = linear.in_features
		self.out_dim = linear.out_features
		base_device = linear.weight.device
		base_dtype = linear.weight.dtype

		for p in self.linear.parameters():
			p.requires_grad = False

		self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_dim, device=base_device, dtype=base_dtype))
		self.lora_B = nn.Parameter(torch.empty(self.out_dim, self.rank, device=base_device, dtype=base_dtype))
		self.reset_parameters()

	def reset_parameters(self) -> None:
		nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
		nn.init.zeros_(self.lora_B)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		base = self.linear(x)
		update = (x @ self.lora_A.t()) @ self.lora_B.t()
		return base + self.scale * update


def _vector_lora_rank_for_block(block_idx: int) -> int:
	if 0 <= block_idx <= 2:
		return 16
	if 3 <= block_idx <= 5:
		return 8
	if 6 <= block_idx <= 8:
		return 4
	if 9 <= block_idx <= 11:
		return 2
	# Fallback for deeper variants; keep the smallest configured rank.
	return 2


def _resolve_attr_path(root: nn.Module, path: str):
	obj = root
	for token in path.split("."):
		if not hasattr(obj, token):
			return None
		obj = getattr(obj, token)
	return obj


def _find_transformer_blocks(model: nn.Module):
	# Ordered from most common to less common DAv2 wrappers.
	candidate_paths = [
		"blocks",
		"pretrained.blocks",
		"backbone.blocks",
		"backbone.pretrained.blocks",
		"model.blocks",
		"model.pretrained.blocks",
		"module.blocks",
		"module.pretrained.blocks",
	]

	for path in candidate_paths:
		blocks = _resolve_attr_path(model, path)
		if blocks is not None:
			return blocks, path

	# Last-resort scan for a child module exposing "blocks" where blocks contain attn.qkv.
	for module_name, module in model.named_modules():
		if not hasattr(module, "blocks"):
			continue
		blocks = getattr(module, "blocks")
		try:
			if len(blocks) > 0 and hasattr(blocks[0], "attn") and hasattr(blocks[0].attn, "qkv"):
				return blocks, f"{module_name}.blocks" if module_name else "blocks"
		except TypeError:
			continue

	return None, None


def _resolve_path_with_indices(root: nn.Module, path: str):
	obj = root
	for token in path.split("."):
		if token.isdigit():
			obj = obj[int(token)]
		else:
			obj = getattr(obj, token)
	return obj


def _replace_module_by_path(root: nn.Module, module_path: str, new_module: nn.Module) -> None:
	parent_path, leaf = module_path.rsplit(".", 1)
	parent = _resolve_path_with_indices(root, parent_path)
	if leaf.isdigit():
		parent[int(leaf)] = new_module
	else:
		setattr(parent, leaf, new_module)


def _extract_block_idx_from_qkv_path(module_path: str, default_idx: int) -> int:
	match = re.search(r"(?:^|\.)blocks\.(\d+)(?:\.|$)", module_path)
	if match is not None:
		return int(match.group(1))
	match = re.search(r"(?:^|\.)layer\.(\d+)(?:\.|$)", module_path)
	if match is not None:
		return int(match.group(1))
	match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", module_path)
	if match is not None:
		return int(match.group(1))
	return default_idx


def inject_vector_lora_into_dav2(model: nn.Module, alpha: float = 16.0) -> nn.Module:
	"""
	Inject Vector-LoRA wrappers into DAv2 attention qkv layers.

	Block discovery order:
	1) model.blocks
	2) model.pretrained.blocks

	Rank schedule:
	- blocks 0-2  : rank 16
	- blocks 3-5  : rank 8
	- blocks 6-8  : rank 4
	- blocks 9-11 : rank 2
	"""
	blocks, block_path = _find_transformer_blocks(model)

	injected_count = 0
	if blocks is not None:
		for block_idx, block in enumerate(blocks):
			if not hasattr(block, "attn") or not hasattr(block.attn, "qkv"):
				continue

			current_qkv = block.attn.qkv
			if isinstance(current_qkv, VectorLoRA_QKV):
				injected_count += 1
				continue
			if not isinstance(current_qkv, nn.Linear):
				raise TypeError(
					f"Expected block.attn.qkv to be nn.Linear at block {block_idx}, got {type(current_qkv)}"
				)

			rank = _vector_lora_rank_for_block(block_idx)
			block.attn.qkv = VectorLoRA_QKV(current_qkv, rank=rank, alpha=alpha)
			injected_count += 1
	else:
		# Fallback: find all attention qkv projections by module name and wrap them directly.
		qkv_paths = [
			name for name, mod in model.named_modules()
			if name.endswith("attn.qkv") and isinstance(mod, (nn.Linear, VectorLoRA_QKV))
		]
		if len(qkv_paths) > 0:
			block_path = "named_modules(attn.qkv)"
			for default_idx, qkv_path in enumerate(qkv_paths):
				current_qkv = _resolve_path_with_indices(model, qkv_path)
				if isinstance(current_qkv, VectorLoRA_QKV):
					injected_count += 1
					continue
				if not isinstance(current_qkv, nn.Linear):
					raise TypeError(
						f"Expected {qkv_path} to be nn.Linear, got {type(current_qkv)}"
					)
				block_idx = _extract_block_idx_from_qkv_path(qkv_path, default_idx)
				rank = _vector_lora_rank_for_block(block_idx)
				_replace_module_by_path(model, qkv_path, VectorLoRA_QKV(current_qkv, rank=rank, alpha=alpha))
				injected_count += 1
		else:
			# Final fallback for backbones using split attention projections: query/key/value.
			split_paths = [
				name for name, mod in model.named_modules()
				if isinstance(mod, nn.Linear) and (name.endswith(".query") or name.endswith(".value"))
			]
			if len(split_paths) == 0:
				raise AttributeError(
					"Could not find transformer blocks, attn.qkv modules, or split query/value modules for LoRA injection. "
					"Tried paths: blocks, pretrained.blocks, backbone.blocks, backbone.pretrained.blocks, "
					"model.blocks, model.pretrained.blocks."
				)
			block_path = "named_modules(*.query|*.value)"
			for default_idx, linear_path in enumerate(split_paths):
				current_linear = _resolve_path_with_indices(model, linear_path)
				if isinstance(current_linear, VectorLoRA_Linear):
					injected_count += 1
					continue
				if not isinstance(current_linear, nn.Linear):
					raise TypeError(
						f"Expected {linear_path} to be nn.Linear, got {type(current_linear)}"
					)
				block_idx = _extract_block_idx_from_qkv_path(linear_path, default_idx)
				rank = _vector_lora_rank_for_block(block_idx)
				_replace_module_by_path(model, linear_path, VectorLoRA_Linear(current_linear, rank=rank, alpha=alpha))
				injected_count += 1

	# Freeze everything first.
	for param in model.parameters():
		param.requires_grad = False

	# Re-enable LoRA params only in wrapped QKV layers.
	for module in model.modules():
		if not isinstance(module, (VectorLoRA_QKV, VectorLoRA_Linear)):
			continue
		for name, param in module.named_parameters(recurse=False):
			if name.startswith("lora_") and (name.endswith("_A") or name.endswith("_B")):
				param.requires_grad = True

	total_params = sum(p.numel() for p in model.parameters())
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	trainable_pct = 100.0 * trainable_params / total_params if total_params > 0 else 0.0

	print(f"[Vector-LoRA] Blocks source: {block_path}")
	print(f"[Vector-LoRA] Injected wrappers into {injected_count} blocks")
	print(
		f"[Vector-LoRA] Trainable params: {trainable_params:,} / {total_params:,} "
		f"({trainable_pct:.4f}%)"
	)

	return model
