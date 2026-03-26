from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import build_title_suffix, ensure_dir


def inverse_transform_targets(y_scaled, target_scaler):
    """将标准化后的目标值恢复到原始尺度。"""
    original_shape = y_scaled.shape
    y_inverse = target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(original_shape)
    return y_inverse


def compute_metrics(y_true, y_pred):
    """计算总体展开后的 MSE、MAE、R2 指标。"""
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    r2 = r2_score(y_true_flat, y_pred_flat)
    return {"mse": mse, "mae": mae, "r2": r2}


def compute_metrics_by_horizon(y_true, y_pred):
    """多步预测时，按每个预测步单独计算指标。"""
    metrics_list = []
    horizon_count = y_true.shape[1]
    for step_idx in range(horizon_count):
        step_true = y_true[:, step_idx]
        step_pred = y_pred[:, step_idx]
        metrics_list.append(
            {
                "horizon_step": step_idx + 1,
                "mse": mean_squared_error(step_true, step_pred),
                "mae": mean_absolute_error(step_true, step_pred),
                "r2": r2_score(step_true, step_pred),
            }
        )
    return pd.DataFrame(metrics_list)


def save_metrics_files(model_dir, overall_metrics, metrics_by_horizon=None):
    """保存总体指标和逐步指标。"""
    overall_df = pd.DataFrame([overall_metrics])
    overall_df.to_csv(model_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    if metrics_by_horizon is not None:
        metrics_by_horizon.to_csv(model_dir / "metrics_by_horizon.csv", index=False, encoding="utf-8-sig")


def save_prediction_file(model_dir, y_times, y_true, y_pred):
    """保存测试集预测值与真实值。"""
    result_dict = {"forecast_start_time": pd.to_datetime(y_times[:, 0]).astype(str)}
    horizon_count = y_true.shape[1]

    for step_idx in range(horizon_count):
        result_dict[f"true_t+{step_idx + 1}"] = y_true[:, step_idx]
        result_dict[f"pred_t+{step_idx + 1}"] = y_pred[:, step_idx]

    pred_df = pd.DataFrame(result_dict)
    pred_df.to_csv(model_dir / "predictions.csv", index=False, encoding="utf-8-sig")


def plot_loss_curve(train_losses, val_losses, model_dir, display_site_name, model_name, config):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="训练损失")
    plt.plot(val_losses, label="验证损失")
    plt.title(f"{display_site_name}\n{model_name} 训练损失曲线\n{build_title_suffix(config, model_name)}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(model_dir / f"loss_curve_lb{config['LOOKBACK']}_lf{config['LOOK_FORWARD']}.png")
    plt.close()


def plot_test_results(model_dir, display_site_name, model_name, y_true, y_pred, config):
    """绘制测试集整体曲线、局部放大图、误差图和散点图。"""
    title_text = f"{display_site_name}\n{build_title_suffix(config, model_name)}"
    flat_true = y_true.reshape(-1)
    flat_pred = y_pred.reshape(-1)
    flat_error = flat_pred - flat_true

    plt.figure(figsize=(14, 5))
    plt.plot(flat_true, label="真实值", linewidth=1)
    plt.plot(flat_pred, label="预测值", linewidth=1)
    plt.title(f"{title_text}\n测试集真实值与预测值整体曲线图")
    plt.xlabel("测试样本展开索引")
    plt.ylabel("功率")
    plt.legend()
    plt.tight_layout()
    plt.savefig(model_dir / f"test_true_vs_pred_lb{config['LOOKBACK']}_lf{config['LOOK_FORWARD']}.png")
    plt.close()

    zoom_points = min(config["ZOOM_PLOT_POINTS"], len(flat_true))
    plt.figure(figsize=(14, 5))
    plt.plot(flat_true[:zoom_points], label="真实值", linewidth=1)
    plt.plot(flat_pred[:zoom_points], label="预测值", linewidth=1)
    plt.title(f"{title_text}\n测试集局部区间真实值与预测值放大图")
    plt.xlabel("局部样本展开索引")
    plt.ylabel("功率")
    plt.legend()
    plt.tight_layout()
    plt.savefig(model_dir / f"test_zoom_lb{config['LOOKBACK']}_lf{config['LOOK_FORWARD']}.png")
    plt.close()

    plt.figure(figsize=(14, 5))
    plt.plot(flat_error, color="tab:red", linewidth=1)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title(f"{title_text}\n误差曲线图（prediction - true）")
    plt.xlabel("测试样本展开索引")
    plt.ylabel("误差")
    plt.tight_layout()
    plt.savefig(model_dir / f"test_error_curve_lb{config['LOOKBACK']}_lf{config['LOOK_FORWARD']}.png")
    plt.close()

    if len(flat_true) > config["SCATTER_SAMPLE_LIMIT"]:
        sample_index = np.linspace(0, len(flat_true) - 1, config["SCATTER_SAMPLE_LIMIT"]).astype(int)
        scatter_true = flat_true[sample_index]
        scatter_pred = flat_pred[sample_index]
    else:
        scatter_true = flat_true
        scatter_pred = flat_pred

    plt.figure(figsize=(6, 6))
    plt.scatter(scatter_true, scatter_pred, s=12, alpha=0.5, color="tab:purple")
    min_value = min(scatter_true.min(), scatter_pred.min())
    max_value = max(scatter_true.max(), scatter_pred.max())
    plt.plot([min_value, max_value], [min_value, max_value], color="black", linestyle="--")
    plt.title(f"{title_text}\n散点图（true vs pred）")
    plt.xlabel("真实值")
    plt.ylabel("预测值")
    plt.tight_layout()
    plt.savefig(model_dir / f"test_scatter_lb{config['LOOKBACK']}_lf{config['LOOK_FORWARD']}.png")
    plt.close()


def summarize_model_result(model_dir, display_site_name, model_name, y_times, y_true_scaled, y_pred_scaled, target_scaler, train_losses, val_losses, config):
    """完成反归一化、指标保存、预测保存和测试图绘制。"""
    y_true = inverse_transform_targets(y_true_scaled, target_scaler)
    y_pred = inverse_transform_targets(y_pred_scaled, target_scaler)

    overall_metrics = compute_metrics(y_true, y_pred)
    metrics_by_horizon = None
    if y_true.shape[1] > 1:
        metrics_by_horizon = compute_metrics_by_horizon(y_true, y_pred)

    save_metrics_files(model_dir, overall_metrics, metrics_by_horizon)
    save_prediction_file(model_dir, y_times, y_true, y_pred)
    plot_loss_curve(train_losses, val_losses, model_dir, display_site_name, model_name, config)
    plot_test_results(model_dir, display_site_name, model_name, y_true, y_pred, config)

    return overall_metrics, metrics_by_horizon


def save_site_summary(site_dir, site_metrics_df):
    site_metrics_df.to_csv(site_dir / "site_summary_metrics.csv", index=False, encoding="utf-8-sig")


def plot_model_comparison(site_dir, display_site_name, site_metrics_df, config):
    """绘制站点下三个模型的指标对比柱状图。"""
    if site_metrics_df.empty:
        return

    plt.figure(figsize=(10, 6))
    x = np.arange(len(site_metrics_df))
    width = 0.25

    plt.bar(x - width, site_metrics_df["mse"], width=width, label="MSE")
    plt.bar(x, site_metrics_df["mae"], width=width, label="MAE")
    plt.bar(x + width, site_metrics_df["r2"], width=width, label="R²")
    plt.xticks(x, site_metrics_df["model_name"])
    plt.title(f"{display_site_name}\n三个模型指标对比柱状图\n{build_title_suffix(config)}")
    plt.xlabel("模型")
    plt.ylabel("指标值")
    plt.legend()
    plt.tight_layout()
    plt.savefig(site_dir / f"model_comparison_lb{config['LOOKBACK']}_lf{config['LOOK_FORWARD']}.png")
    plt.close()


def save_global_summary(results_dir, summary_df):
    ensure_dir(results_dir)
    summary_df.to_csv(Path(results_dir) / "summary_metrics.csv", index=False, encoding="utf-8-sig")
