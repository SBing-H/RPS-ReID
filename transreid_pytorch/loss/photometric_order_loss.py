import torch
import torch.nn.functional as F


def photometric_order_consistency_loss(
    predicted_order: torch.Tensor,
    target_order: torch.Tensor,
    confidence_threshold: float = 0.10,
) -> torch.Tensor:
    """Confidence-weighted soft order loss for aligned normal/dark pairs.

    Strong normal-light ordinal relations receive more weight. Relations near
    equality are excluded because their sign is easily changed by sensor noise.
    """
    if predicted_order.shape != target_order.shape:
        raise ValueError(
            'Photometric-order prediction and target must have identical shapes, '
            'got {} and {}.'.format(predicted_order.shape, target_order.shape)
        )
    if predicted_order.dim() != 4:
        raise ValueError('Photometric-order tensors must have shape [B, C, H, W].')
    if not 0.0 <= confidence_threshold < 1.0:
        raise ValueError('PHOTOMETRIC_ORDER_CONFIDENCE must be in [0, 1).')

    target_order = target_order.detach()
    confidence = (
        (target_order.abs() - confidence_threshold)
        / max(1.0 - confidence_threshold, 1e-6)
    ).clamp_(min=0.0, max=1.0)
    element_loss = F.smooth_l1_loss(
        predicted_order,
        target_order,
        reduction='none',
    )
    normalizer = confidence.sum().clamp_min(1.0)
    return (element_loss * confidence).sum() / normalizer
