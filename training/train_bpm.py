"""Training pipeline for BPM detection models."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_bpm_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-4,
    save_dir: str = "checkpoints",
) -> dict:
    """Train a BPM detection model.

    Args:
        model: PyTorch model with bpm_head and conf_head.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        epochs: Number of training epochs.
        lr: Learning rate.
        save_dir: Directory to save checkpoints.

    Returns:
        Training history dict.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    bpm_criterion = nn.SmoothL1Loss()
    best_val_mae = float("inf")
    history: dict[str, list] = {"train_loss": [], "val_mae": []}

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            mel_specs = batch["mel_spec"].to(device)
            true_bpm = batch["bpm"].to(device)

            bpm_pred, conf_pred = model(mel_specs)

            loss = bpm_criterion(bpm_pred.squeeze(), true_bpm)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        # Validation
        model.eval()
        val_errors = []

        with torch.no_grad():
            for batch in val_loader:
                mel_specs = batch["mel_spec"].to(device)
                true_bpm = batch["bpm"].to(device)

                bpm_pred, _ = model(mel_specs)
                errors = torch.abs(bpm_pred.squeeze() - true_bpm)
                val_errors.extend(errors.cpu().tolist())

        val_mae = sum(val_errors) / max(len(val_errors), 1)
        scheduler.step()

        history["train_loss"].append(epoch_loss / len(train_loader))
        history["val_mae"].append(val_mae)

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss / len(train_loader):.4f} | Val MAE: {val_mae:.2f} BPM")

        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), f"{save_dir}/best_bpm_model.pt")
            print(f"  → Saved best model (MAE: {val_mae:.2f} BPM)")

    return history
