"""Training loop for controlled fetal-heart TransUNet ablations."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from datasets.dataset_fetalheart import FetalHeartNPZDataset, FetalHeartTrainTransform
from losses_fetalheart import CompositeSegmentationLoss


def _state_dict(model):
    return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()


def _worker_init(seed):
    def initialise(worker_id):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    return initialise


def _balanced_sampler(dataset, num_classes, seed):
    present_by_case = []
    frequencies = Counter()
    for case_name in dataset.sample_list:
        path = os.path.join(dataset.base_dir, case_name + ".npz")
        with np.load(path, allow_pickle=False) as data:
            label = np.asarray(data["label"])
        present = [int(value) for value in np.unique(label) if 0 < int(value) < num_classes]
        present_by_case.append(present)
        frequencies.update(present)

    weights = []
    for present in present_by_case:
        if not present:
            weights.append(1.0)
            continue
        # Square-root inverse frequency is deliberately conservative for a
        # multi-label dataset; max ensures rare structures affect sampling.
        weights.append(max((len(dataset) / frequencies[class_id]) ** 0.5 for class_id in present))
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


@torch.no_grad()
def evaluate_present_case_class(model, loader, num_classes, img_size, device):
    """Macro metric over every case/class pair in which GT is present.

    This matches the diagnosis objective: a missed present structure receives
    zero instead of being removed from the average.
    """

    model.eval()
    dice_values = []
    iou_values = []
    recall_values = []
    precision_values = []
    per_class = {class_id: {"dice": [], "iou": []} for class_id in range(1, num_classes)}

    for batch in loader:
        image = batch["image"].float().to(device)
        label = batch["label"].long().to(device)
        if image.ndim == 3:
            image = image.unsqueeze(1)
        original_size = label.shape[-2:]
        resized_image = F.interpolate(image, size=(img_size, img_size), mode="bilinear", align_corners=False)
        logits = model(resized_image)
        prediction = torch.argmax(logits, dim=1, keepdim=True).float()
        prediction = F.interpolate(prediction, size=original_size, mode="nearest").squeeze(1).long()

        for class_id in range(1, num_classes):
            truth = label == class_id
            if not torch.any(truth):
                continue
            estimate = prediction == class_id
            true_positive = torch.logical_and(truth, estimate).sum().item()
            false_positive = torch.logical_and(~truth, estimate).sum().item()
            false_negative = torch.logical_and(truth, ~estimate).sum().item()
            dice = 2.0 * true_positive / max(2.0 * true_positive + false_positive + false_negative, 1.0)
            iou = true_positive / max(true_positive + false_positive + false_negative, 1.0)
            recall = true_positive / max(true_positive + false_negative, 1.0)
            precision = true_positive / max(true_positive + false_positive, 1.0)
            dice_values.append(dice)
            iou_values.append(iou)
            recall_values.append(recall)
            precision_values.append(precision)
            per_class[class_id]["dice"].append(dice)
            per_class[class_id]["iou"].append(iou)

    metrics = {
        "mean_dice": float(np.mean(dice_values)),
        "mean_iou": float(np.mean(iou_values)),
        "mean_recall": float(np.mean(recall_values)),
        "mean_precision": float(np.mean(precision_values)),
        "present_case_class_count": len(dice_values),
        "per_class": {
            str(class_id): {
                "mean_dice": float(np.mean(values["dice"])) if values["dice"] else None,
                "mean_iou": float(np.mean(values["iou"])) if values["iou"] else None,
                "case_count": len(values["dice"]),
            }
            for class_id, values in per_class.items()
        },
    }
    model.train()
    return metrics


def trainer_fetalheart(args, model, snapshot_path):
    from tensorboardX import SummaryWriter

    os.makedirs(snapshot_path, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("arguments: %s", vars(args))

    train_dataset = FetalHeartNPZDataset(
        args.root_path,
        args.list_dir,
        "train",
        transform=FetalHeartTrainTransform([args.img_size, args.img_size], args.augmentation),
    )
    val_dataset = FetalHeartNPZDataset(args.root_path, args.list_dir, "val")
    sampler = None
    shuffle = True
    if args.sampling == "class_balanced":
        sampler = _balanced_sampler(train_dataset, args.num_classes, args.seed)
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size * args.n_gpu,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init(args.seed),
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    logging.info("train cases: %d; validation cases: %d", len(train_dataset), len(val_dataset))

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    device = next(model.parameters()).device
    criterion = CompositeSegmentationLoss(
        num_classes=args.num_classes,
        mode=args.loss_mode,
        ce_weight=args.ce_weight,
        region_weight=args.region_weight,
        alpha_fp=args.tversky_alpha_fp,
        beta_fn=args.tversky_beta_fn,
        tversky_gamma=args.tversky_gamma,
        tversky_fraction=args.tversky_fraction,
    )
    optimizer_mode = getattr(args, "optimizer", "sgd")
    if optimizer_mode == "adamw":
        adamw_weight_decay = getattr(args, "adamw_weight_decay", 0.05)
        decay, no_decay = [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        optimizer = optim.AdamW(
            [
                {"params": decay, "weight_decay": adamw_weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=args.base_lr,
        )
        logging.info(
            "optimizer: AdamW lr %g weight_decay %g (no decay on 1-D params and biases)",
            args.base_lr,
            adamw_weight_decay,
        )
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    writer = SummaryWriter(os.path.join(snapshot_path, "tensorboard"))
    max_iterations = args.max_epochs * len(train_loader)
    warmup_iterations = getattr(args, "warmup_epochs", 0) * len(train_loader)
    iteration = 0
    best_dice = -1.0
    history = []

    model.train()
    for epoch in tqdm(range(args.max_epochs), ncols=80):
        epoch_losses = []
        for batch in train_loader:
            image = batch["image"].float().to(device, non_blocking=True)
            label = batch["label"].long().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(image)
                total_loss, ce_loss, region_loss = criterion(logits, label)
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if warmup_iterations and iteration < warmup_iterations:
                learning_rate = args.base_lr * (iteration + 1) / warmup_iterations
            else:
                progress = (iteration - warmup_iterations) / max(1, max_iterations - warmup_iterations)
                learning_rate = args.base_lr * (1.0 - progress) ** 0.9
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            iteration += 1
            epoch_losses.append(total_loss.item())
            writer.add_scalar("train/total_loss", total_loss.item(), iteration)
            writer.add_scalar("train/ce_loss", ce_loss.item(), iteration)
            writer.add_scalar("train/region_loss", region_loss.item(), iteration)
            writer.add_scalar("train/lr", learning_rate, iteration)

        record = {"epoch": epoch + 1, "train_loss": float(np.mean(epoch_losses))}
        if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.max_epochs:
            metrics = evaluate_present_case_class(
                model, val_loader, args.num_classes, args.img_size, device
            )
            record.update(metrics)
            logging.info(
                "epoch %d loss %.6f val Dice %.6f IoU %.6f recall %.6f precision %.6f",
                epoch + 1,
                record["train_loss"],
                metrics["mean_dice"],
                metrics["mean_iou"],
                metrics["mean_recall"],
                metrics["mean_precision"],
            )
            writer.add_scalar("val/mean_dice", metrics["mean_dice"], epoch + 1)
            writer.add_scalar("val/mean_iou", metrics["mean_iou"], epoch + 1)
            if metrics["mean_dice"] > best_dice:
                best_dice = metrics["mean_dice"]
                torch.save(_state_dict(model), os.path.join(snapshot_path, "best_dice.pth"))
                with open(os.path.join(snapshot_path, "best_metrics.json"), "w", encoding="utf-8") as handle:
                    json.dump(record, handle, indent=2, ensure_ascii=False)
        history.append(record)
        with open(os.path.join(snapshot_path, "history.json"), "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2, ensure_ascii=False)

    torch.save(_state_dict(model), os.path.join(snapshot_path, "last.pth"))
    writer.close()
    return {"best_dice": best_dice, "snapshot_path": snapshot_path}
