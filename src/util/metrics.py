import torch

def metrics_from_logits(logits, masks, threshold=0.5, eps=1e-7):
    """
    logits: [B, 1, H, W]
    masks:  [B, 1, H, W]
    """

    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    masks = masks.float()

    tp = (preds * masks).sum()
    tn = ((1 - preds) * (1 - masks)).sum()
    fp = (preds * (1 - masks)).sum()
    fn = ((1 - preds) * masks).sum()

    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)

    return {
        "accuracy": accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "iou": iou.item(),
    }