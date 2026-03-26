from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import seaborn as sns
except ImportError:
    sns = None
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from config import build_title_suffix, ensure_dir, sanitize_name, sanitize_plot_text


def scan_excel_files(base_dir="."):
    """自动扫描当前目录下的所有 .xlsx 文件。"""
    base_dir = Path(base_dir)
    excel_files = sorted(base_dir.glob("*.xlsx"))
    print(f"当前目录扫描到 {len(excel_files)} 个 Excel 文件。")
    for file_path in excel_files:
        print(" -", file_path.name)
    return excel_files


def normalize_column_name(col_name):
    """内部匹配用的列名规范化，不改变原始列名。"""
    col_name = str(col_name).strip().lower()
    col_name = col_name.replace("\n", " ")
    col_name = re.sub(r"\s+", " ", col_name)
    return col_name


def infer_time_target_cols(columns, config):
    """优先使用配置列名，若失败则自动识别。"""
    columns = list(columns)
    normalized_map = {normalize_column_name(col): col for col in columns}

    if config["TIME_COL"] in columns:
        time_col = config["TIME_COL"]
    else:
        time_col = None
        time_keywords = [
            "time(year-month-day h:m:s)",
            "time",
            "timestamp",
            "date",
            "时间",
            "日期",
        ]
        for keyword in time_keywords:
            for normalized_col, raw_col in normalized_map.items():
                if keyword in normalized_col:
                    time_col = raw_col
                    break
            if time_col is not None:
                break
        if time_col is None:
            time_col = columns[0]
            print(f"未自动识别到时间列，默认使用第一列: {time_col}")

    if config["TARGET_COL"] in columns:
        target_col = config["TARGET_COL"]
    else:
        target_col = None
        target_keywords = [
            "power (mw)",
            "power",
            "功率",
            "有功功率",
            "发电功率",
        ]
        for keyword in target_keywords:
            for normalized_col, raw_col in normalized_map.items():
                if keyword in normalized_col:
                    target_col = raw_col
                    break
            if target_col is not None:
                break
        if target_col is None:
            target_col = columns[-1]
            print(f"未自动识别到目标列，默认使用最后一列: {target_col}")

    print(f"识别到时间列: {time_col}")
    print(f"识别到目标列: {target_col}")
    return time_col, target_col


def load_site_dataframe(file_path, config):
    """读取单个风电场 Excel，并打印基础信息。"""
    print("\n" + "=" * 80)
    print(f"开始读取文件: {file_path.name}")
    df = pd.read_excel(file_path)
    print(f"数据维度: {df.shape}")
    print("列名列表:")
    print(list(df.columns))
    print("前 5 行数据:")
    print(df.head().to_string())
    time_col, target_col = infer_time_target_cols(df.columns, config)
    return df, time_col, target_col


def clean_site_dataframe(df, time_col, target_col):
    """完成时间排序、缺失值处理、重复处理和异常值裁剪。"""
    print("\n开始数据清洗...")
    df = df.copy()
    original_rows = len(df)

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    invalid_time_count = int(df[time_col].isna().sum())

    for col in df.columns:
        if col != time_col:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    duplicate_count_before = int(df.duplicated(subset=[time_col]).sum())
    missing_count_before = int(df.isna().sum().sum())

    print(f"清洗前样本数: {original_rows}")
    print(f"无法解析的时间数量: {invalid_time_count}")
    print(f"重复时间数量: {duplicate_count_before}")
    print(f"清洗前总缺失值数量: {missing_count_before}")

    df = df.dropna(subset=[time_col]).sort_values(by=time_col).reset_index(drop=True)
    df = df.drop_duplicates(subset=[time_col], keep="first").reset_index(drop=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col not in numeric_cols:
        numeric_cols.append(target_col)

    # 缺失值处理：时间序列中优先采用插值，再做前向/后向填充
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    # 异常值处理：IQR 裁剪，不激进删样本，只截断极端值
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    missing_count_after = int(df.isna().sum().sum())
    print(f"清洗后样本数: {len(df)}")
    print(f"清洗后总缺失值数量: {missing_count_after}")
    print("数据清洗完成。")
    return df


def add_time_features(df, time_col, target_col):
    """构造基础时间特征和周期性时间特征。"""
    print("\n开始特征工程...")
    df = df.copy()
    df["hour"] = df[time_col].dt.hour
    df["dayofweek"] = df[time_col].dt.dayofweek
    df["month"] = df[time_col].dt.month
    df["dayofyear"] = df[time_col].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col != target_col]

    print("最终参与建模的特征列名如下:")
    print(feature_cols)
    return df, feature_cols


def select_weather_plot_cols(df, target_col, max_cols=4):
    """为原始数据可视化挑选代表性气象特征。"""
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != target_col]
    priority_keywords = [
        "wheel hub",
        "wind speed",
        "wind direction",
        "temperature",
        "atmosphere",
        "humidity",
    ]

    selected = []
    for keyword in priority_keywords:
        for col in numeric_cols:
            if keyword in normalize_column_name(col) and col not in selected:
                selected.append(col)
                if len(selected) >= max_cols:
                    return selected

    for col in numeric_cols:
        if col not in selected:
            selected.append(col)
        if len(selected) >= max_cols:
            break
    return selected


def plot_raw_figures(df, file_path, time_col, target_col, feature_cols, config):
    """绘制并保存原始数据可视化图。"""
    site_name = sanitize_name(file_path.stem)
    figure_dir = ensure_dir(config["RESULTS_DIR"] / "figures" / site_name)
    title_suffix = build_title_suffix(config)
    safe_target_col = sanitize_plot_text(target_col)

    plt.figure(figsize=(14, 5))
    plt.plot(df[time_col], df[target_col], color="tab:blue", linewidth=1)
    plt.title(f"{file_path.stem}\n目标功率时间序列图\n{title_suffix}")
    plt.xlabel("时间")
    plt.ylabel(safe_target_col)
    plt.tight_layout()
    plt.savefig(figure_dir / "01_power_series.png")
    plt.close()

    weather_cols = select_weather_plot_cols(df, target_col, max_cols=4)
    safe_weather_cols = [sanitize_plot_text(col) for col in weather_cols]
    plt.figure(figsize=(14, 6))
    for col, safe_col in zip(weather_cols, safe_weather_cols):
        plt.plot(df[time_col], df[col], linewidth=1, label=safe_col)
    plt.title(f"{file_path.stem}\n主要气象特征时间序列图\n{title_suffix}")
    plt.xlabel("时间")
    plt.ylabel("特征值")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "02_weather_series.png")
    plt.close()

    corr_cols = weather_cols + [target_col]
    corr_matrix = df[corr_cols].corr()
    plt.figure(figsize=(8, 6))
    if sns is not None:
        corr_matrix_display = corr_matrix.copy()
        corr_matrix_display.index = [sanitize_plot_text(col) for col in corr_matrix_display.index]
        corr_matrix_display.columns = [sanitize_plot_text(col) for col in corr_matrix_display.columns]
        sns.heatmap(corr_matrix_display, annot=True, cmap="coolwarm", fmt=".2f")
    else:
        plt.imshow(corr_matrix, cmap="coolwarm", aspect="auto")
        plt.colorbar()
        safe_corr_cols = [sanitize_plot_text(col) for col in corr_cols]
        plt.xticks(range(len(corr_cols)), safe_corr_cols, rotation=45, ha="right")
        plt.yticks(range(len(corr_cols)), safe_corr_cols)
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.title(f"{file_path.stem}\n相关系数热力图\n{title_suffix}")
    plt.tight_layout()
    plt.savefig(figure_dir / "03_corr_heatmap.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(df[target_col], bins=40, color="tab:green", edgecolor="black")
    plt.title(f"{file_path.stem}\n功率分布直方图\n{title_suffix}")
    plt.xlabel(safe_target_col)
    plt.ylabel("频数")
    plt.tight_layout()
    plt.savefig(figure_dir / "04_power_hist.png")
    plt.close()

    wind_speed_col = None
    for col in feature_cols:
        if "wind speed" in normalize_column_name(col):
            wind_speed_col = col
            break
    if wind_speed_col is None:
        wind_speed_col = feature_cols[0]

    plt.figure(figsize=(8, 5))
    plt.scatter(df[wind_speed_col], df[target_col], s=10, alpha=0.4, color="tab:red")
    plt.title(f"{file_path.stem}\n功率与关键风速特征散点图\n{title_suffix}")
    plt.xlabel(sanitize_plot_text(wind_speed_col))
    plt.ylabel(safe_target_col)
    plt.tight_layout()
    plt.savefig(figure_dir / "05_power_vs_wind.png")
    plt.close()


def split_and_scale_data(df, time_col, target_col, feature_cols, config):
    """按时间顺序切分数据，并且仅在训练集上拟合归一化器。"""
    print("\n开始按时间顺序划分训练集、验证集、测试集...")
    total_size = len(df)
    train_end = int(total_size * config["TRAIN_RATIO"])
    val_end = int(total_size * (config["TRAIN_RATIO"] + config["VAL_RATIO"]))

    if train_end <= config["LOOKBACK"]:
        raise ValueError("训练集太短，无法构造 LOOKBACK 窗口，请减小 LOOKBACK 或检查数据量。")
    if val_end >= total_size:
        raise ValueError("TRAIN_RATIO 和 VAL_RATIO 设置不合理，请重新检查。")

    print(f"总样本数: {total_size}")
    print(f"训练集区间: [0, {train_end})")
    print(f"验证集区间: [{train_end}, {val_end})")
    print(f"测试集区间: [{val_end}, {total_size})")

    train_df = df.iloc[:train_end].copy()

    # 归一化器只能在训练集上 fit，避免验证集和测试集的信息泄露到训练阶段
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    feature_scaler.fit(train_df[feature_cols].values)
    target_scaler.fit(train_df[[target_col]].values)

    full_features_scaled = feature_scaler.transform(df[feature_cols].values)
    full_target_scaled = target_scaler.transform(df[[target_col]].values)

    return {
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "full_features_scaled": full_features_scaled,
        "full_target_scaled": full_target_scaled,
        "time_values": df[time_col].values,
        "train_end": train_end,
        "val_end": val_end,
        "total_size": total_size,
    }


def create_sequences(features, targets, time_values, lookback, look_forward, start_idx, end_idx, use_history_target):
    """将时间序列构造成监督学习样本。"""
    x_list = []
    y_list = []
    y_time_list = []

    for target_start_idx in range(start_idx, end_idx - look_forward + 1):
        input_start_idx = target_start_idx - lookback
        input_end_idx = target_start_idx
        target_end_idx = target_start_idx + look_forward

        if input_start_idx < 0:
            continue

        x_feat_seq = features[input_start_idx:input_end_idx]
        if use_history_target:
            x_target_hist_seq = targets[input_start_idx:input_end_idx].reshape(-1, 1)
            x_seq = np.concatenate([x_feat_seq, x_target_hist_seq], axis=1)
        else:
            x_seq = x_feat_seq
        y_seq = targets[target_start_idx:target_end_idx].reshape(-1)
        y_time_seq = time_values[target_start_idx:target_end_idx]

        x_list.append(x_seq)
        y_list.append(y_seq)
        y_time_list.append(y_time_seq)

    x_array = np.array(x_list, dtype=np.float32)
    y_array = np.array(y_list, dtype=np.float32)
    y_time_array = np.array(y_time_list)
    return x_array, y_array, y_time_array


def build_dataloaders_from_sequences(x_train, y_train, x_val, y_val, x_test, y_test, config):
    """将数组转换为 Tensor，并构建 train/val/test DataLoader。"""
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["BATCH_SIZE"], shuffle=False)

    print("\nDataLoader 构建完成。")
    print(f"是否启用历史功率输入: {config['USE_HISTORY_TARGET']}")
    print(f"X_train shape: {x_train_tensor.shape}, y_train shape: {y_train_tensor.shape}")
    print(f"X_val shape: {x_val_tensor.shape}, y_val shape: {y_val_tensor.shape}")
    print(f"X_test shape: {x_test_tensor.shape}, y_test shape: {y_test_tensor.shape}")
    if config["USE_HISTORY_TARGET"]:
        print("说明: 输入最后 1 维包含过去 LOOKBACK 步的历史功率。")

    sample_x, sample_y = next(iter(train_loader))
    print(f"一个 batch 的输入维度: {sample_x.shape}")
    print(f"一个 batch 的输出维度: {sample_y.shape}")

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
    }


def prepare_site_data(file_path, config):
    """完成单个站点从数据读取到 DataLoader 构建的完整流程。"""
    file_path = Path(file_path)
    raw_df, time_col, target_col = load_site_dataframe(file_path, config)
    clean_df = clean_site_dataframe(raw_df, time_col, target_col)
    feature_df, feature_cols = add_time_features(clean_df, time_col, target_col)
    plot_raw_figures(feature_df, file_path, time_col, target_col, feature_cols, config)

    split_info = split_and_scale_data(feature_df, time_col, target_col, feature_cols, config)
    features_scaled = split_info["full_features_scaled"]
    targets_scaled = split_info["full_target_scaled"]
    time_values = split_info["time_values"]
    train_end = split_info["train_end"]
    val_end = split_info["val_end"]
    total_size = split_info["total_size"]

    original_input_size = len(feature_cols)
    x_train, y_train, y_time_train = create_sequences(
        features_scaled,
        targets_scaled,
        time_values,
        config["LOOKBACK"],
        config["LOOK_FORWARD"],
        start_idx=config["LOOKBACK"],
        end_idx=train_end,
        use_history_target=config["USE_HISTORY_TARGET"],
    )
    x_val, y_val, y_time_val = create_sequences(
        features_scaled,
        targets_scaled,
        time_values,
        config["LOOKBACK"],
        config["LOOK_FORWARD"],
        start_idx=train_end,
        end_idx=val_end,
        use_history_target=config["USE_HISTORY_TARGET"],
    )
    x_test, y_test, y_time_test = create_sequences(
        features_scaled,
        targets_scaled,
        time_values,
        config["LOOKBACK"],
        config["LOOK_FORWARD"],
        start_idx=val_end,
        end_idx=total_size,
        use_history_target=config["USE_HISTORY_TARGET"],
    )

    print("\n滑动窗口构造完成。")
    print(f"是否启用历史功率输入: {config['USE_HISTORY_TARGET']}")
    print(f"原始特征数: {original_input_size}")
    print(f"最终输入特征数: {x_train.shape[-1]}")
    print(f"X_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {x_val.shape}, y_val shape: {y_val.shape}")
    print(f"X_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

    loaders = build_dataloaders_from_sequences(x_train, y_train, x_val, y_val, x_test, y_test, config)

    site_name = sanitize_name(file_path.stem)
    site_dir = ensure_dir(config["RESULTS_DIR"] / site_name)

    return {
        "file_path": file_path,
        "site_name": site_name,
        "display_site_name": file_path.stem,
        "time_col": time_col,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "input_size": x_train.shape[-1],
        "original_input_size": original_input_size,
        "feature_df": feature_df,
        "feature_scaler": split_info["feature_scaler"],
        "target_scaler": split_info["target_scaler"],
        "site_dir": site_dir,
        "y_time_train": y_time_train,
        "y_time_val": y_time_val,
        "y_time_test": y_time_test,
        **loaders,
    }
