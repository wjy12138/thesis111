import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import ensure_dir
from evaluate import summarize_model_result
from models import CNNLSTMModel, CNNLSTMSeq2SeqModel, CNNModel, LSTMModel


def evaluate_on_loader(model, data_loader, criterion, device):
    """Evaluate legacy forecasters whose batches are shaped as (x, y)."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_x.size(0)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    mean_loss = total_loss / len(data_loader.dataset)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    return mean_loss, all_preds, all_targets


def evaluate_seq2seq_on_loader(model, data_loader, criterion, device):
    """Evaluate seq2seq forecasters whose batches are shaped as (x, decoder_start, y)."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_decoder_start, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_decoder_start = batch_decoder_start.to(device)
            batch_y = batch_y.to(device)

            outputs = model(
                batch_x,
                decoder_start=batch_decoder_start,
                decoder_inputs=None,
                teacher_forcing_ratio=0.0,
            )
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_x.size(0)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    mean_loss = total_loss / len(data_loader.dataset)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    return mean_loss, all_preds, all_targets


def _build_best_model_path(model_dir, config):
    return model_dir / f"best_model_lb{config['LOOKBACK']}_lf{config['LOOK_FORWARD']}.pth"


def _save_best_or_fallback(model, model_name, best_model_path, has_best_checkpoint):
    if not has_best_checkpoint:
        print(
            f"Warning: {model_name} did not produce a finite validation loss. "
            "Using the final epoch weights for test evaluation."
        )
        torch.save(model.state_dict(), best_model_path)


def run_training_loop(model, model_name, data_bundle, config):
    """Train a legacy forecaster and select the best checkpoint by validation loss."""
    device = config["DEVICE"]
    model = model.to(device)
    print("\n模型结构如下:")
    print(model)

    train_loader = data_bundle["train_loader"]
    val_loader = data_bundle["val_loader"]
    test_loader = data_bundle["test_loader"]
    model_dir = ensure_dir(data_bundle["site_dir"] / model_name)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    best_model_path = _build_best_model_path(model_dir, config)
    has_best_checkpoint = False

    print(f"\n开始训练模型: {model_name}")
    start_time = time.time()

    for epoch in range(config["EPOCHS"]):
        model.train()
        running_train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * batch_x.size(0)

        mean_train_loss = running_train_loss / len(train_loader.dataset)
        mean_val_loss, _, _ = evaluate_on_loader(model, val_loader, criterion, device)

        train_losses.append(mean_train_loss)
        val_losses.append(mean_val_loss)

        print(
            f"模型: {model_name} | Epoch [{epoch + 1}/{config['EPOCHS']}] "
            f"| Train Loss: {mean_train_loss:.6f} | Val Loss: {mean_val_loss:.6f}"
        )

        if math.isfinite(mean_val_loss) and mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            torch.save(model.state_dict(), best_model_path)
            has_best_checkpoint = True

    train_time = time.time() - start_time
    print(f"{model_name} 训练完成，耗时: {train_time:.2f} 秒")

    _save_best_or_fallback(model, model_name, best_model_path, has_best_checkpoint)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_preds_scaled, test_targets_scaled = evaluate_on_loader(model, test_loader, criterion, device)
    print(f"{model_name} 最佳权重加载完成，测试集 MSELoss: {test_loss:.6f}")

    overall_metrics, metrics_by_horizon = summarize_model_result(
        model_dir=model_dir,
        display_site_name=data_bundle["display_site_name"],
        model_name=model_name,
        y_times=data_bundle["y_time_test"],
        y_true_scaled=test_targets_scaled,
        y_pred_scaled=test_preds_scaled,
        target_scaler=data_bundle["target_scaler"],
        train_losses=train_losses,
        val_losses=val_losses,
        config=config,
    )

    result_row = {
        "site_name": data_bundle["display_site_name"],
        "model_name": model_name,
        "lookback": config["LOOKBACK"],
        "look_forward": config["LOOK_FORWARD"],
        "batch_size": config["BATCH_SIZE"],
        "learning_rate": config["LEARNING_RATE"],
        "hidden_size": config["HIDDEN_SIZE"],
        "num_layers": config["NUM_LAYERS"],
        "epochs": config["EPOCHS"],
        "dropout": config["DROPOUT"],
        "mse": overall_metrics["mse"],
        "mae": overall_metrics["mae"],
        "r2": overall_metrics["r2"],
        "train_time": train_time,
    }
    return result_row, metrics_by_horizon


def run_seq2seq_training_loop(model, model_name, data_bundle, config):
    """Train a seq2seq forecaster with teacher forcing for training only."""
    device = config["DEVICE"]
    model = model.to(device)
    print("\n模型结构如下:")
    print(model)

    train_loader = data_bundle["train_loader_seq2seq"]
    val_loader = data_bundle["val_loader_seq2seq"]
    test_loader = data_bundle["test_loader_seq2seq"]
    model_dir = ensure_dir(data_bundle["site_dir"] / model_name)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    best_model_path = _build_best_model_path(model_dir, config)
    has_best_checkpoint = False

    print(f"\n开始训练模型: {model_name}")
    start_time = time.time()

    for epoch in range(config["EPOCHS"]):
        model.train()
        running_train_loss = 0.0

        for batch_x, batch_decoder_start, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_decoder_start = batch_decoder_start.to(device)
            batch_y = batch_y.to(device)
            decoder_inputs = torch.cat([batch_decoder_start, batch_y[:, :-1]], dim=1)

            optimizer.zero_grad()
            outputs = model(
                batch_x,
                decoder_start=batch_decoder_start,
                decoder_inputs=decoder_inputs,
                teacher_forcing_ratio=config["SEQ2SEQ_TEACHER_FORCING_RATIO"],
            )
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * batch_x.size(0)

        mean_train_loss = running_train_loss / len(train_loader.dataset)
        mean_val_loss, _, _ = evaluate_seq2seq_on_loader(model, val_loader, criterion, device)

        train_losses.append(mean_train_loss)
        val_losses.append(mean_val_loss)

        print(
            f"模型: {model_name} | Epoch [{epoch + 1}/{config['EPOCHS']}] "
            f"| Train Loss: {mean_train_loss:.6f} | Val Loss: {mean_val_loss:.6f}"
        )

        if math.isfinite(mean_val_loss) and mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            torch.save(model.state_dict(), best_model_path)
            has_best_checkpoint = True

    train_time = time.time() - start_time
    print(f"{model_name} 训练完成，耗时: {train_time:.2f} 秒")

    _save_best_or_fallback(model, model_name, best_model_path, has_best_checkpoint)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_preds_scaled, test_targets_scaled = evaluate_seq2seq_on_loader(model, test_loader, criterion, device)
    print(f"{model_name} 最佳权重加载完成，测试集 MSELoss: {test_loss:.6f}")

    overall_metrics, metrics_by_horizon = summarize_model_result(
        model_dir=model_dir,
        display_site_name=data_bundle["display_site_name"],
        model_name=model_name,
        y_times=data_bundle["y_time_test"],
        y_true_scaled=test_targets_scaled,
        y_pred_scaled=test_preds_scaled,
        target_scaler=data_bundle["target_scaler"],
        train_losses=train_losses,
        val_losses=val_losses,
        config=config,
    )

    result_row = {
        "site_name": data_bundle["display_site_name"],
        "model_name": model_name,
        "lookback": config["LOOKBACK"],
        "look_forward": config["LOOK_FORWARD"],
        "batch_size": config["BATCH_SIZE"],
        "learning_rate": config["LEARNING_RATE"],
        "hidden_size": config["HIDDEN_SIZE"],
        "num_layers": config["NUM_LAYERS"],
        "epochs": config["EPOCHS"],
        "dropout": config["DROPOUT"],
        "mse": overall_metrics["mse"],
        "mae": overall_metrics["mae"],
        "r2": overall_metrics["r2"],
        "train_time": train_time,
    }
    return result_row, metrics_by_horizon


def train_lstm_model(data_bundle, config):
    model = LSTMModel(
        input_size=data_bundle["input_size"],
        hidden_size=config["HIDDEN_SIZE"],
        num_layers=config["NUM_LAYERS"],
        dropout=config["DROPOUT"],
        output_size=config["LOOK_FORWARD"],
    )
    return run_training_loop(model, "LSTM", data_bundle, config)


def train_cnn_model(data_bundle, config):
    model = CNNModel(
        input_size=data_bundle["input_size"],
        out_channels=config["CNN_OUT_CHANNELS"],
        kernel_size=config["CNN_KERNEL_SIZE"],
        pool_size=config["POOL_SIZE"],
        output_size=config["LOOK_FORWARD"],
    )
    return run_training_loop(model, "CNN", data_bundle, config)


def train_cnn_lstm_model(data_bundle, config):
    model = CNNLSTMModel(
        input_size=data_bundle["input_size"],
        out_channels_1=config["CNN_OUT_CHANNELS_1"],
        out_channels_2=config["CNN_OUT_CHANNELS_2"],
        kernel_size_1=config["CNN_KERNEL_SIZE_1"],
        kernel_size_2=config["CNN_KERNEL_SIZE_2"],
        pool_size=config["POOL_SIZE"],
        hidden_size=config["HIDDEN_SIZE"],
        num_layers=config["NUM_LAYERS"],
        dropout=config["DROPOUT"],
        output_size=config["LOOK_FORWARD"],
    )
    return run_training_loop(model, "CNN_LSTM", data_bundle, config)


def train_cnn_lstm_seq2seq_model(data_bundle, config):
    model = CNNLSTMSeq2SeqModel(
        input_size=data_bundle["input_size"],
        out_channels=config["CNN_OUT_CHANNELS"],
        kernel_size=config["CNN_KERNEL_SIZE"],
        pool_size=config["POOL_SIZE"],
        hidden_size=config["HIDDEN_SIZE"],
        num_layers=config["NUM_LAYERS"],
        dropout=config["DROPOUT"],
        output_size=config["LOOK_FORWARD"],
    )
    return run_seq2seq_training_loop(model, "CNN_LSTM_SEQ2SEQ", data_bundle, config)
