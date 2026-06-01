import torch
import torch.nn.functional as F


def _get_gt_mask(logits, target):
    mask = torch.zeros_like(logits, dtype=torch.bool)
    return mask.scatter_(1, target.unsqueeze(1), True)


def _get_other_mask(logits, target):
    mask = torch.ones_like(logits, dtype=torch.bool)
    return mask.scatter_(1, target.unsqueeze(1), False)


def _cat_mask(prob, gt_mask, other_mask):
    gt_prob = (prob * gt_mask).sum(dim=1, keepdim=True)
    other_prob = (prob * other_mask).sum(dim=1, keepdim=True)
    return torch.cat([gt_prob, other_prob], dim=1)


def dkd_loss(logits_student, logits_teacher, target, alpha=1.0, beta=0.8, temperature=4.0):
    logits_student = logits_student.float()
    logits_teacher = logits_teacher.float()
    target = target.long()

    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)

    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = _cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = _cat_mask(pred_teacher, gt_mask, other_mask)
    tckd_loss = F.kl_div(torch.log(pred_student.clamp_min(1e-12)), pred_teacher, reduction='batchmean')

    masked_student = logits_student / temperature
    masked_teacher = logits_teacher / temperature
    masked_student = masked_student.masked_fill(gt_mask, -1000.0)
    masked_teacher = masked_teacher.masked_fill(gt_mask, -1000.0)
    nckd_loss = F.kl_div(
        F.log_softmax(masked_student, dim=1),
        F.softmax(masked_teacher, dim=1),
        reduction='batchmean',
    )

    return (alpha * tckd_loss + beta * nckd_loss) * (temperature ** 2)
