import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import Classifier, save_model
from datasets.classification_dataset import load_data
from metrics import AccuracyMetric

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create datasets
    # 'transform_pipeline' can be 'train' for random flips, etc. 'val' for no augmentation
    ds_train = load_data(split="train", transform_pipeline="train")
    ds_val = load_data(split="val", transform_pipeline="val")

    loader_train = DataLoader(ds_train, batch_size=64, shuffle=True, num_workers=2)
    loader_val = DataLoader(ds_val, batch_size=64, shuffle=False, num_workers=2)

    # Instantiate model, loss, optimizer
    model = Classifier(in_channels=3, num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Metric
    metric = AccuracyMetric()

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        metric.reset()
        for batch in loader_train:
            x, y = batch["image"].to(device), batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            # track training accuracy
            preds = logits.argmax(dim=1)
            metric.add(preds, y)

        train_acc = metric.compute()["accuracy"]

        # Validation
        model.eval()
        metric.reset()
        with torch.inference_mode():
            for batch in loader_val:
                x, y = batch["image"].to(device), batch["label"].to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)
                metric.add(preds, y)
        val_acc = metric.compute()["accuracy"]

        print(f"Epoch={epoch}  Loss={loss.item():.4f}  TrainAcc={train_acc:.3f}  ValAcc={val_acc:.3f}")

    # Save final model
    save_model(model)

if __name__ == "__main__":
    main()
