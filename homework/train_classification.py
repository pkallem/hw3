import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import load_model, save_model
from .datasets.classification_dataset import load_data
from .metrics import AccuracyMetric


def train(
    exp_dir: str = "logs",
    model_name: str = "classifier",
    num_epoch: int = 10,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 2024,
    **kwargs,
):
    """
    Example classification training function that replicates the style of your previous train.py.
    You can call it like:

        from train_classification import train
        train(model_name="classifier", num_epoch=10, lr=1e-3)

    It will:
      - load the classification dataset from 'classification_data/'
      - train a classification model
      - compute and log train/val accuracies
      - save the model to homework/classifier.th (for grading)
      - also save a copy of the model in the timestamped log directory
    """

    # device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA/MPS not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # load model
    model = load_model(model_name=model_name, **kwargs)
    model = model.to(device)
    model.train()

    # load training and validation data
    train_data = load_data(
        dataset_path="classification_data/train",
        transform_pipeline="aug",
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_data = load_data(
        dataset_path="classification_data/val",
        transform_pipeline="default",
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    # loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    metric = AccuracyMetric()
    global_step = 0

    for epoch in range(num_epoch):
        # clear train metric at beginning of epoch
        metric.reset()
        train_losses = []

        model.train()
        for batch in train_data:
            img, label = batch
            img, label = img.to(device), label.to(device)

            optimizer.zero_grad()
            logits = model(img)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            # track train accuracy
            preds = logits.argmax(dim=1)
            metric.add(preds, label)

            global_step += 1

        epoch_train_loss = np.mean(train_losses)
        epoch_train_acc = metric.compute()["accuracy"]

        # validation
        model.eval()
        metric.reset()
        val_losses = []

        with torch.inference_mode():
            for batch in val_data:
                img, label = batch
                img, label = img.to(device), label.to(device)

                logits = model(img)
                loss = criterion(logits, label)
                val_losses.append(loss.item())

                preds = logits.argmax(dim=1)
                metric.add(preds, label)

        epoch_val_loss = np.mean(val_losses)
        epoch_val_acc = metric.compute()["accuracy"]

        # log to tensorboard
        logger.add_scalar("train_loss", epoch_train_loss, global_step)
        logger.add_scalar("train_accuracy", epoch_train_acc, global_step)
        logger.add_scalar("val_loss", epoch_val_loss, global_step)
        logger.add_scalar("val_accuracy", epoch_val_acc, global_step)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d}/{num_epoch:2d}: "
                f"train_loss={epoch_train_loss:.4f} train_acc={epoch_train_acc:.4f} | "
                f"val_loss={epoch_val_loss:.4f} val_acc={epoch_val_acc:.4f}"
            )

    # save model to the homework folder for grading
    save_model(model)

    # also save a copy in our logging folder
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="classifier")
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2024)
    # add any other model hyper-params you need as optional arguments here

    args = parser.parse_args()
    train(**vars(parser.parse_args()))
