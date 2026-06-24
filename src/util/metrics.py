import torch
import json
import numpy as np
import rasterio

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

def get_high_low_iou_examples(results, n=3):
    """
    Return the n highest and n lowest IoU examples as separate lists.
    """

    sorted_results = sorted(
        results,
        key=lambda sample: float(sample["iou"])
    )

    lowest_iou = sorted_results[:n]
    highest_iou = sorted_results[-n:][::-1]

    return highest_iou, lowest_iou


def get_high_low_flood_pixel_examples(results, n=3):
    """
    Return the n examples with the highest and lowest number of flood pixels
    in the ground-truth mask.
    """

    def flood_pixel_count(sample):
        mask = sample["mask"]

        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu()

        return int((mask == 1).sum().item())

    sorted_results = sorted(
        results,
        key=flood_pixel_count
    )

    lowest_flood_pixels = sorted_results[:n]
    highest_flood_pixels = sorted_results[-n:][::-1]

    return highest_flood_pixels, lowest_flood_pixels

def union_results_by_sample_id(list_a, list_b):
    """
    Combine two lists of dictionaries, removing duplicates by sample_id.
    If the same sample_id appears twice, the version from list_a is kept.
    """
    union = {}

    for sample in list_a + list_b:
        sample_id = sample["sample_id"]

        if sample_id not in union:
            union[sample_id] = sample

    return list(union.values())

def intersect_multiple_results_by_sample_id(*lists):
    """
    Return only samples that appear in all provided lists.
    Keeps the version from the first list.
    """
    common_ids = set(sample["sample_id"] for sample in lists[0])

    for result_list in lists[1:]:
        common_ids &= {sample["sample_id"] for sample in result_list}

    return [
        sample for sample in lists[0]
        if sample["sample_id"] in common_ids
    ]

def results_not_in_other_by_sample_id(list_a, list_b):
    """
    Return samples from list_a that do not appear in list_b.
    Matched by sample_id.
    """
    ids_b = {sample["sample_id"] for sample in list_b}

    return [
        sample for sample in list_a
        if sample["sample_id"] not in ids_b
    ]

def count_mask_pixels(
    mask_path,
    water_classes=(1, 2, 3, 4, 5),
    ignore_classes=(99,),
    non_water_class=0,
):
    """
    Count flood/water and non-flood/non-water pixels in a mask GeoTIFF.

    Args:
        mask_path: Path to mask .tif file.
        water_classes: Classes treated as flood/water.
        ignore_classes: Classes excluded from counting.
        non_water_class: Class treated as non-flood/non-water.

    Returns:
        dict with flood_pixels, non_flood_pixels, ignored_pixels, total_valid_pixels.
    """

    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    flood_mask = np.isin(mask, water_classes)
    non_flood_mask = mask == non_water_class
    ignore_mask = np.isin(mask, ignore_classes)

    flood_pixels = int(np.sum(flood_mask))
    non_flood_pixels = int(np.sum(non_flood_mask))
    ignored_pixels = int(np.sum(ignore_mask))

    return {
        "flood_pixels": flood_pixels,
        "non_flood_pixels": non_flood_pixels,
        "ignored_pixels": ignored_pixels,
        "total_valid_pixels": flood_pixels + non_flood_pixels,
        "flood_pixel_percentage": flood_pixels / (flood_pixels + non_flood_pixels) if (flood_pixels + non_flood_pixels) > 0 else 0.0,
    }