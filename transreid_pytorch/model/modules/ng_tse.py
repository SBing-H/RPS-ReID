import torch
import torch.nn as nn
import torch.nn.functional as F


class NormGuidedTokenStructureEnhancement(nn.Module):
    def __init__(
        self,
        dim,
        grid_size,
        kernel_size=5,
        beta=0.25,
        gate_type='q',
        detach_gate=True,
        residual_gate=True,
        weighted_structure=True,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError('NG-TSE kernel_size must be odd.')
        if gate_type not in ('none', 'q', 'qk'):
            raise ValueError("NG-TSE gate_type must be one of 'none', 'q', or 'qk'.")

        self.dim = dim
        self.grid_size = tuple(grid_size)
        self.kernel_size = kernel_size
        self.beta = beta
        self.gate_type = gate_type
        self.detach_gate = detach_gate
        self.residual_gate = residual_gate
        self.weighted_structure = weighted_structure

        gate_dim = 3 if gate_type == 'qk' else 1
        if gate_type == 'none':
            self.gate_mlp = None
        else:
            hidden_dim = max(4, gate_dim * 4)
            self.gate_mlp = nn.Sequential(
                nn.Linear(gate_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )

    @staticmethod
    def _normalize_token_score(score):
        mean = score.mean(dim=1, keepdim=True)
        std = score.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        return (score - mean) / std

    def _build_gate(self, q_norm, k_norm=None):
        if self.gate_type == 'none':
            return None

        q_score = q_norm[:, 1:]
        if self.detach_gate:
            q_score = q_score.detach()
        q_score = self._normalize_token_score(q_score)

        if self.gate_type == 'q':
            gate_input = q_score.unsqueeze(-1)
        else:
            if k_norm is None:
                raise RuntimeError("NG-TSE gate_type='qk' requires k_norm.")
            k_score = k_norm[:, 1:]
            if self.detach_gate:
                k_score = k_score.detach()
            k_score = self._normalize_token_score(k_score)
            ratio = q_score / (k_score.abs() + 1e-6)
            gate_input = torch.stack([q_score, k_score, ratio], dim=-1)

        return torch.sigmoid(self.gate_mlp(gate_input))

    def forward(self, x, q_norm, k_norm=None):
        if x.dim() != 3:
            raise RuntimeError('NG-TSE expects x with shape [B, 1+N, C].')
        if q_norm.dim() != 2:
            raise RuntimeError('NG-TSE expects q_norm with shape [B, 1+N].')

        b, token_count, c = x.shape
        h, w = self.grid_size
        patch_count = token_count - 1
        if c != self.dim:
            raise RuntimeError('NG-TSE dim mismatch: got {}, expected {}.'.format(c, self.dim))
        if patch_count != h * w:
            raise RuntimeError('NG-TSE grid mismatch: got {} patch tokens, expected {}x{}.'.format(patch_count, h, w))
        if q_norm.shape[0] != b or q_norm.shape[1] != token_count:
            raise RuntimeError('NG-TSE q_norm shape mismatch.')

        patch_tokens = x[:, 1:, :]
        feat_map = patch_tokens.transpose(1, 2).reshape(b, c, h, w)
        neighbors = F.unfold(
            feat_map,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        )
        neighbors = neighbors.view(b, c, self.kernel_size * self.kernel_size, patch_count)
        center = feat_map.reshape(b, c, 1, patch_count)
        grad = F.relu(neighbors - center)

        if self.weighted_structure:
            weight = grad / (1.0 + grad.sum(dim=2, keepdim=True))
            struct = (weight * neighbors).sum(dim=2)
        else:
            struct = grad.mean(dim=2)

        struct = struct.transpose(1, 2)
        gate = self._build_gate(q_norm, k_norm)
        if gate is not None:
            if self.residual_gate:
                struct = (1.0 + gate) * struct
            else:
                struct = gate * struct

        cls_delta = x.new_zeros(b, 1, c)
        patch_delta = self.beta * struct
        return torch.cat([cls_delta, patch_delta], dim=1)
