import pandas as pd

from config import configure_environment, get_config, set_plot_style, set_random_seed
from data_utils import prepare_site_data, scan_excel_files
from evaluate import plot_model_comparison, save_global_summary, save_site_summary
from train import train_cnn_lstm_model, train_cnn_model, train_lstm_model
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

def train_site(file_name, model_name="ALL", **overrides):
    """
    单站点训练入口。
    示例:
    train_site("Wind farm site 1 (Nominal capaciaty-99MW).xlsx", model_name="LSTM", EPOCHS=10)
    """
    configure_environment()
    config = get_config(**overrides)
    set_random_seed(config["RANDOM_SEED"])
    set_plot_style(config["CHINESE_FONT_PATH"])
    print(f"是否启用历史功率输入: {config['USE_HISTORY_TARGET']}")

    file_path = config["BASE_DIR"] / file_name
    data_bundle = prepare_site_data(file_path, config)

    available_trainers = {
        "LSTM": train_lstm_model,
        "CNN": train_cnn_model,
        "CNN_LSTM": train_cnn_lstm_model,
    }

    if model_name == "ALL":
        selected_models = config["SELECTED_MODELS"]
    else:
        selected_models = [model_name]

    result_rows = []
    for current_model_name in selected_models:
        if current_model_name not in available_trainers:
            raise ValueError(f"不支持的模型名称: {current_model_name}")

        trainer = available_trainers[current_model_name]
        result_row, _ = trainer(data_bundle, config)
        result_rows.append(result_row)

    site_metrics_df = pd.DataFrame(result_rows)
    save_site_summary(data_bundle["site_dir"], site_metrics_df)
    plot_model_comparison(data_bundle["site_dir"], data_bundle["display_site_name"], site_metrics_df, config)
    return site_metrics_df


def run_all_sites(**overrides):
    """
    批量训练入口。
    示例:
    run_all_sites(EPOCHS=5, LOOKBACK=48, LOOK_FORWARD=6)
    """
    configure_environment()
    config = get_config(**overrides)
    set_random_seed(config["RANDOM_SEED"])
    set_plot_style(config["CHINESE_FONT_PATH"])

    all_excel_files = scan_excel_files(config["BASE_DIR"])
    if not all_excel_files:
        raise FileNotFoundError("当前目录未扫描到 .xlsx 文件，请确认数据已放在代码同目录下。")

    if config["SELECTED_SITE_FILES"] is None:
        selected_files = [file_path.name for file_path in all_excel_files]
    else:
        selected_files = config["SELECTED_SITE_FILES"]

    all_result_dfs = []
    for file_name in selected_files:
        site_result_df = train_site(file_name=file_name, model_name="ALL", **overrides)
        all_result_dfs.append(site_result_df)

    summary_df = pd.concat(all_result_dfs, axis=0, ignore_index=True)
    save_global_summary(config["RESULTS_DIR"], summary_df)
    print("\n所有站点训练完成，汇总结果已保存到 results/summary_metrics.csv")
    return summary_df


if __name__ == "__main__":
    """
    直接运行 main.py 时，会按照 config.py 顶部参数执行。

    常见用法:
    1. 跑全部站点、全部模型:
       python main.py

    2. 在其他脚本中只跑一个站点:
       from main import train_site
       train_site("Wind farm site 1 (Nominal capacity-99MW).xlsx", model_name="LSTM")

    3. 修改 LOOKBACK 和 LOOK_FORWARD 做对比实验:
       train_site("Wind farm site 1 (Nominal capacity-99MW).xlsx", model_name="CNN", LOOKBACK=48, LOOK_FORWARD=6)

    4. 启用历史功率输入:
       train_site("Wind farm site 1 (Nominal capacity-99MW).xlsx", model_name="LSTM", USE_HISTORY_TARGET=True)

    5. 修改超参数:
       train_site(
           "Wind farm site 1 (Nominal capacity-99MW).xlsx",
           model_name="CNN_LSTM",
           BATCH_SIZE=128,
           LEARNING_RATE=0.0005,
           HIDDEN_SIZE=128,
           NUM_LAYERS=2,
           DROPOUT=0.3,
           EPOCHS=20
       )
    """
    run_all_sites()
