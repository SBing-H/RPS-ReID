from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhotometricOrderAttentionBias(nn.Module):
    """Build a local patch-to-patch attention prior from luminance order."""

    _NEIGHBOR_INDICES = (0, 1, 2, 3, 5, 6, 7, 8)

    def __init__(
        self,
        patch_rows: int,
        patch_cols: int,
        pixel_mean: Sequence[float],
        pixel_std: Sequence[float],
        scales: Sequence[int] = (1, 2),
        temperature: float = 0.10,
        margin: float = 0.02,
        luminance_domain: str = 'log',
        order_encoding: str = 'soft',
        bias_scale: float = 0.20,
        local_radius: int = 2,
        use_identity_bias: bool = True,
    ) -> None:
        super().__init__()
        if patch_rows <= 0 or patch_cols <= 0:
            raise ValueError('Patch-grid dimensions must be positive.')
        if len(pixel_mean) != 3 or len(pixel_std) != 3:
            raise ValueError('Photometric order expects three-channel RGB input.')
        if temperature <= 0.0:
            raise ValueError('Photometric-order temperature must be positive.')
        if margin < 0.0:
            raise ValueError('Photometric-order margin must be non-negative.')
        if bias_scale < 0.0:
            raise ValueError('Photometric-order attention-bias scale must be non-negative.')
        if local_radius <= 0:
            raise ValueError('Photometric-order attention radius must be positive.')
        luminance_domain = str(luminance_domain).lower()
        order_encoding = str(order_encoding).lower()
        if luminance_domain not in ('log', 'linear'):
            raise ValueError("Photometric-order luminance domain must be 'log' or 'linear'.")
        if order_encoding not in ('soft', 'hard'):
            raise ValueError("Photometric-order encoding must be 'soft' or 'hard'.")

        parsed_scales = tuple(int(scale) for scale in scales)
        if not parsed_scales or any(scale <= 0 for scale in parsed_scales):
            raise ValueError('Photometric-order scales must contain positive integers.')

        self.patch_rows = int(patch_rows)
        self.patch_cols = int(patch_cols)
        self.scales = parsed_scales
        self.temperature = float(temperature)
        self.margin = float(margin)
        self.bias_scale = float(bias_scale)
        self.luminance_domain = luminance_domain
        self.order_encoding = order_encoding
        self.use_identity_bias = bool(use_identity_bias)

        mean = torch.tensor(pixel_mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(pixel_std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer('pixel_mean', mean, persistent=False)
        self.register_buffer('pixel_std', std, persistent=False)
        self.register_buffer(
            'luminance_weights',
            torch.tensor((0.299, 0.587, 0.114), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            'local_attention_mask',
            self._build_local_attention_mask(local_radius),
            persistent=False,
        )

    def _build_local_attention_mask(self, radius: int) -> torch.Tensor:
        rows = torch.arange(self.patch_rows)
        cols = torch.arange(self.patch_cols)
        row_grid = rows[:, None].expand(self.patch_rows, self.patch_cols)
        col_grid = cols[None, :].expand(self.patch_rows, self.patch_cols)
        coordinates = torch.stack((row_grid, col_grid), dim=-1).reshape(-1, 2)
        distance = (
            coordinates[:, None, :] - coordinates[None, :, :]
        ).abs().amax(dim=-1)
        return ((distance > 0) & (distance <= int(radius))).float()

    def _order_relations(self, luminance: torch.Tensor) -> torch.Tensor:
        relations = []
        batch_size, _, height, width = luminance.shape
        for dilation in self.scales:
            padded = F.pad(
                luminance,
                (dilation, dilation, dilation, dilation),
                mode='replicate',
            )
            neighborhoods = F.unfold(
                padded,
                kernel_size=3,
                dilation=dilation,
            ).view(batch_size, 9, height, width)
            center = neighborhoods[:, 4:5]
            neighbors = neighborhoods[:, self._NEIGHBOR_INDICES]
            difference = center - neighbors
            if self.order_encoding == 'hard':
                # Hard ordinality has no tolerance region: {-1, 0, +1}.
                relations.append(torch.sign(difference))
            else:
                positive = torch.sigmoid(
                    (difference - self.margin) / self.temperature
                )
                negative = torch.sigmoid(
                    (-difference - self.margin) / self.temperature
                )
                relations.append(positive - negative)
        return torch.cat(relations, dim=1)

    def _extract_order_maps(self, normalized_rgb: torch.Tensor) -> torch.Tensor:
        if normalized_rgb.dim() != 4 or normalized_rgb.shape[1] != 3:
            raise RuntimeError('Photometric order input must have shape [B, 3, H, W].')
        rgb = normalized_rgb * self.pixel_std + self.pixel_mean
        rgb = rgb.clamp(min=0.0, max=1.0)
        luminance = (rgb * self.luminance_weights).sum(dim=1, keepdim=True)
        patch_luminance = F.adaptive_avg_pool2d(
            luminance,
            output_size=(self.patch_rows, self.patch_cols),
        )
        order_luminance = (
            torch.log(patch_luminance.clamp_min(1e-4))
            if self.luminance_domain == 'log'
            else patch_luminance
        )
        return self._order_relations(order_luminance)

    def forward(self, normalized_rgb: torch.Tensor) -> torch.Tensor:
        order_maps = self._extract_order_maps(normalized_rgb)
        signatures = order_maps.flatten(2).transpose(1, 2)
        # The soft order relation already suppresses near-equal luminance via
        # the margin. Cosine similarity then compares the remaining ordinal
        # patterns without introducing another magnitude gate that could make
        # the configured attention-bias scale ineffective.
        normalized_signatures = F.normalize(signatures, dim=-1, eps=1e-6)
        order_similarity = normalized_signatures @ normalized_signatures.transpose(-2, -1)
        patch_bias = (
            self.bias_scale
            * order_similarity
            * self.local_attention_mask
        )

        # The last Transformer block produces the global ReID descriptor from
        # its CLS token. Patch-to-patch bias alone cannot change that CLS token
        # within the same block, so reliable ordinal evidence also guides the
        # CLS-to-patch aggregation logits. A bounded standardized score keeps
        # this term independent of the absolute image brightness.
        attention_bias = F.pad(patch_bias.unsqueeze(1), (1, 0, 1, 0))
        if self.use_identity_bias:
            order_confidence = signatures.abs().mean(dim=-1)
            confidence_mean = order_confidence.mean(dim=1, keepdim=True)
            confidence_std = order_confidence.std(
                dim=1, keepdim=True, unbiased=False
            ).clamp_min(1e-6)
            cls_patch_bias = self.bias_scale * torch.tanh(
                (order_confidence - confidence_mean) / confidence_std
            )
            attention_bias[:, :, 0, 1:] = cls_patch_bias.unsqueeze(1)
        return attention_bias
