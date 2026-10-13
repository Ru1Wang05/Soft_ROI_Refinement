"""Training loop for soft ROI refinement."""
from __future__ import annotations
import json, logging, os, random, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_fetalheart import FetalHeartNPZDataset, FetalHeartTrainTransform
from losses_fetalheart import CompositeSegmentationLoss
from trainer_fetalheart import evaluate_present_case_class

WEAK_IDS = (8, 9, 10, 11)

def _state_dict(model):
    return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

def _worker_init(seed):
    def initialise(worker_id):
        random.seed(seed + worker_id); np.random.seed(seed + worker_id)
    return initialise

def _weak_targets(label):
    return torch.stack([(label == class_id).float() for class_id in WEAK_IDS], dim=1)

def trainer_fetalheart_roi_refiner(args, model, snapshot_path):
    os.makedirs(snapshot_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(snapshot_path, "log.txt"), level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S", force=True)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    train_dataset = FetalHeartNPZDataset(args.root_path, args.list_dir, "train",
        transform=FetalHeartTrainTransform([args.img_size, args.img_size], args.augmentation))
    val_dataset = FetalHeartNPZDataset(args.root_path, args.list_dir, "val")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size * args.n_gpu, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, worker_init_fn=_worker_init(args.seed))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    if args.n_gpu > 1: model = nn.DataParallel(model)
    device = next(model.parameters()).device
    seg_loss = CompositeSegmentationLoss(args.num_classes, mode="dice_ce", ce_weight=0.2, region_weight=0.8)
    # For [B, C, H, W] logits, class weights must broadcast across the two
    # spatial dimensions. A flat [C] tensor incorrectly aligns with W.
    roi_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(args.roi_pos_weights, device=device).view(1, len(WEAK_IDS), 1, 1)
    )
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    writer = SummaryWriter(os.path.join(snapshot_path, "tensorboard"))
    max_iterations = args.max_epochs * len(train_loader); iteration = 0; best_dice = -1.0; history = []
    for epoch in tqdm(range(args.max_epochs), ncols=80):
        model.train(); epoch_losses = []; epoch_roi = []
        for batch in train_loader:
            image = batch["image"].float().to(device, non_blocking=True)
            label = batch["label"].long().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                outputs = model(image, return_aux=True)
                loss_seg, _, _ = seg_loss(outputs["segmentation"], label)
                loss_r = roi_loss(outputs["roi_attention_logits"], _weak_targets(label))
                total_loss = loss_seg + args.roi_loss_weight * loss_r
            scaler.scale(total_loss).backward(); scaler.step(optimizer); scaler.update()
            lr = args.base_lr * (1.0 - iteration / max_iterations) ** 0.9
            for group in optimizer.param_groups: group["lr"] = lr
            iteration += 1; epoch_losses.append(total_loss.item()); epoch_roi.append(loss_r.item())
        record = {"epoch": epoch + 1, "train_loss": float(np.mean(epoch_losses)), "train_roi_loss": float(np.mean(epoch_roi))}
        if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.max_epochs:
            metrics = evaluate_present_case_class(model, val_loader, args.num_classes, args.img_size, device)
            record.update(metrics)
            logging.info("epoch %d loss %.6f roi %.6f val Dice %.6f IoU %.6f", epoch + 1, record["train_loss"], record["train_roi_loss"], metrics["mean_dice"], metrics["mean_iou"])
            if metrics["mean_dice"] > best_dice:
                best_dice = metrics["mean_dice"]
                torch.save(_state_dict(model), os.path.join(snapshot_path, "best_dice.pth"))
                with open(os.path.join(snapshot_path, "best_metrics.json"), "w", encoding="utf-8") as handle: json.dump(record, handle, indent=2, ensure_ascii=False)
        history.append(record)
        with open(os.path.join(snapshot_path, "history.json"), "w", encoding="utf-8") as handle: json.dump(history, handle, indent=2, ensure_ascii=False)
        writer.add_scalar("train/total_loss", record["train_loss"], epoch + 1); writer.add_scalar("train/roi_loss", record["train_roi_loss"], epoch + 1)
    torch.save(_state_dict(model), os.path.join(snapshot_path, "last.pth")); writer.close()
    return {"best_dice": best_dice, "snapshot_path": snapshot_path}
