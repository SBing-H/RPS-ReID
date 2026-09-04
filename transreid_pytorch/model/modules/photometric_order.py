from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhotometricOrderTokenEmbedding(nn.Module):
    """Encode local luminance order relations as Structure patch tokens."""

    _NEIGHBOR_INDICES = (0, 1, 2, 3, 5, 6, 7, 8)

    def __init__(
        self,
        embed_dim: int,
        patch_rows: int,
        patch_cols: int,
        pixel_mean: Sequence[float],
        pixel_std: Sequence[float],
        scales: Sequence[int] = (1, 2),
        hidden_dim: int = 64,
        temperature: float = 0.10,
        margin: float = 0.02,
        luminance_domain: str = 'log',
        order_encoding: str = 'soft',
        beta_init: float = 0.01,
        beta_learnable: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim <= 0 or hidden_dim <= 0:
            raise ValueError('Embedding dimensions must be positive.')
        if patch_rows <= 0 or patch_cols <= 0:
            raise ValueError('Patch-grid dimensions must be positive.')
        if len(pixel_mean) != 3 or len(pixel_std) != 3:
            raise ValueError('Photometric order expects three-channel RGB input.')
        if temperature <= 0.0:
            raise ValueError('PHOTOMETRIC_ORDER_TEMPERATURE must be positive.')
        if margin < 0.0:
            raise ValueError('PHOTOMETRIC_ORDER_MARGIN must be non-negative.')
        luminance_domain = str(luminance_domain).lower()
        order_encoding = str(order_encoding).lower()
        if luminance_domain not in ('log', 'linear'):
            raise ValueError("Photometric-order luminance domain must be 'log' or 'linear'.")
        if order_encoding not in ('soft', 'hard'):
            raise ValueError("Photometric-order encoding must be 'soft' or 'hard'.")

        parsed_scales = tuple(int(scale) for scale in scales)
        if not parsed_scales or any(scale <= 0 for scale in parsed_scales):
            raise ValueError('PHOTOMETRIC_ORDER_SCALES must contain positive integers.')

        self.patch_rows = int(patch_rows)
        self.patch_cols = int(patch_cols)
        self.scales = parsed_scales
        self.temperature = float(temperature)
        self.margin = float(margin)
        self.luminance_domain = luminance_domain
        self.order_encoding = order_encoding

        mean = torch.tensor(pixel_mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(pixel_std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer('pixel_mean', mean, persistent=False)
        self.register_buffer('pixel_std', std, persistent=False)
        self.register_buffer(
            'luminance_weights',
            torch.tensor((0.299, 0.587, 0.114), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        order_channels = len(self.scales) * len(self._NEIGHBOR_INDICES)
        self.order_channels = order_channels
        group_count = min(8, int(hidden_dim))
        while int(hidden_dim) % group_count != 0:
            group_count -= 1
        self.projector = nn.Sequential(
            nn.Conv2d(order_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(group_count, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, embed_dim, kernel_size=1, bias=False),
        )
        # This head is used only by paired auxiliary training.  It asks the
        # dark-image order embedding to predict the fixed normal-light order
        # relations, so the consistency objective has a real gradient path.
        self.order_predictor = nn.Conv2d(
            embed_dim,
            order_channels,
            kernel_size=1,
            bias=False,
        )
        self.beta = nn.Parameter(
            torch.tensor(float(beta_init)),
            requires_grad=bool(beta_learnable),
        )
        self._initialize_projector()

    def _initialize_projector(self) -> None:
        for module in self.projector.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        nn.init.normal_(self.order_predictor.weight, std=0.02)

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

    def _project_order_maps(self, order_maps: torch.Tensor) -> torch.Tensor:
        return self.projector(order_maps)

    def forward(
        self,
        normalized_rgb: torch.Tensor,
        apply_beta: bool = True,
    ) -> torch.Tensor:
        order_maps = self._extract_order_maps(normalized_rgb)
        order_features = self._project_order_maps(order_maps)

        order_tokens = order_features.flatten(2).transpose(1, 2)
        if apply_beta:
            order_tokens = self.beta * order_tokens
        return order_tokens

    def paired_order_prediction(
        self,
        dark_rgb: torch.Tensor,
        normal_rgb: torch.Tensor,
    ):
        """Predict normal-light local order from its aligned dark counterpart."""
        dark_order_maps = self._extract_order_maps(dark_rgb)
        dark_order_features = self._project_order_maps(dark_order_maps)
        predicted_normal_order = torch.tanh(
            self.order_predictor(dark_order_features)
        )
        with torch.no_grad():
            target_normal_order = self._extract_order_maps(normal_rgb)
        return predicted_normal_order, target_normal_order
