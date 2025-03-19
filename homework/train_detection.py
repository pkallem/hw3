import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from homework.models import Detector, save_model
from homework.datasets.road_dataset import load_data
from homework.metrics import DetectionMetric

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_train = load_data(split="train")
    ds_val = load_data(split="val")

    loader_train = DataLoader(ds_train, batch_size=16, shuffle=True, num_workers=2)
    loader_val = DataLoader(ds_val, batch_size=16, shuffle=False, num_workers=2)

    model = Detector(in_channels=3, num_classes=3).to(device)

    # Two losses: cross-entropy for segmentation, L1 for depth
    seg_criterion = nn.CrossEntropyLoss()
    depth_criterion = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    metric = DetectionMetric(num_classes=3)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        metric.reset()
        total_loss = 0.0

        for batch in loader_train:
            x = batch["image"].to(device)      # (B, 3, H, W)
            seg_label = batch["track"].to(device)  # (B, H, W)
            depth_label = batch["depth"].to(device) # (B, H, W)

            optimizer.zero_grad()
            logits, raw_depth = model(x)

            # segmentation loss
            seg_loss = seg_criterion(logits, seg_label)

            # depth loss
            depth_loss = depth_criterion(raw_depth, depth_label)

            loss = seg_loss + depth_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # track metrics on training data
            with torch.inference_mode():
                pred_seg = logits.argmax(dim=1)
                metric.add(pred_seg, seg_label, raw_depth, depth_label)

        train_metrics = metric.compute()
        train_iou = train_metrics["iou"]
        train_acc = train_metrics["accuracy"]
        train_depth_err = train_metrics["abs_depth_error"]
        train_tp_depth_err = train_metrics["tp_depth_error"]

        # Validation
        model.eval()
        metric.reset()
        with torch.inference_mode():
            for batch in loader_val:
                x = batch["image"].to(device)
                seg_label = batch["track"].to(device)
                depth_label = batch["depth"].to(device)

                logits, raw_depth = model(x)
                pred_seg = logits.argmax(dim=1)
                metric.add(pred_seg, seg_label, raw_depth, depth_label)

        val_metrics = metric.compute()
        val_iou = val_metrics["iou"]
        val_acc = val_metrics["accuracy"]
        val_depth_err = val_metrics["abs_depth_error"]
        val_tp_depth_err = val_metrics["tp_depth_error"]

        print(
            f"Epoch={epoch} "
            f"TrainLoss={total_loss:.4f} "
            f"TrainIOU={train_iou:.3f} "
            f"TrainAcc={train_acc:.3f} "
            f"TrainDepthErr={train_depth_err:.3f} "
            f"TrainTPDepthErr={train_tp_depth_err:.3f} || "
            f"ValIOU={val_iou:.3f} "
            f"ValAcc={val_acc:.3f} "
            f"ValDepthErr={val_depth_err:.3f} "
            f"ValTPDepthErr={val_tp_depth_err:.3f}"
        )

    # Save final model
    save_model(model)

if __name__ == "__main__":
    main()
