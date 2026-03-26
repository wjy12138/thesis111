import csv
import re
import shutil
from collections import defaultdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
THESIS_DIR = ROOT_DIR.parent
RESULTS_DIR = ROOT_DIR / "results"
FIGURE_DIR = THESIS_DIR / "figure" / "case_analysis"
CHAPTER_PATH = THESIS_DIR / "chapters" / "chap06.tex"

TARGET_LOOKBACK = 96
TARGET_LOOK_FORWARD = 12
MODEL_ORDER = ["CNN", "LSTM", "CNN_LSTM"]
MODEL_DISPLAY = {
    "CNN": "CNN",
    "LSTM": "LSTM",
    "CNN_LSTM": "CNN-LSTM",
}
SITE_DIR_PATTERN = re.compile(r"Wind_farm_site_(\d+)_Nominal_capacity-(\d+)MW")
SITE_NAME_PATTERN = re.compile(r"Wind farm site (\d+) \(Nominal capacity-(\d+)MW\)")


def read_csv_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def format_metric(value, digits=3):
    return f"{value:.{digits}f}"


def parse_site_name(site_name):
    match = SITE_NAME_PATTERN.fullmatch(site_name)
    if not match:
        raise ValueError(f"Unexpected site name: {site_name}")
    site_id = int(match.group(1))
    capacity = int(match.group(2))
    return {
        "site_id": site_id,
        "capacity": capacity,
        "site_label": f"风场{site_id}",
        "site_label_with_capacity": f"风场{site_id}（装机容量{capacity} MW）",
    }


def load_site_dir_map():
    site_dir_map = {}
    for path in RESULTS_DIR.iterdir():
        if not path.is_dir() or path.name.startswith(".") or path.name == "figures":
            continue
        match = SITE_DIR_PATTERN.fullmatch(path.name)
        if not match:
            continue
        site_id = int(match.group(1))
        site_dir_map[site_id] = path
    return site_dir_map


def load_summary_rows():
    rows = read_csv_rows(RESULTS_DIR / "summary_metrics.csv")
    filtered_rows = []
    for row in rows:
        if int(row["lookback"]) != TARGET_LOOKBACK or int(row["look_forward"]) != TARGET_LOOK_FORWARD:
            continue
        row["lookback"] = int(row["lookback"])
        row["look_forward"] = int(row["look_forward"])
        row["batch_size"] = int(row["batch_size"])
        row["hidden_size"] = int(row["hidden_size"])
        row["num_layers"] = int(row["num_layers"])
        row["epochs"] = int(row["epochs"])
        row["learning_rate"] = float(row["learning_rate"])
        row["dropout"] = float(row["dropout"])
        row["mse"] = float(row["mse"])
        row["mae"] = float(row["mae"])
        row["r2"] = float(row["r2"])
        row["train_time"] = float(row["train_time"])
        row.update(parse_site_name(row["site_name"]))
        filtered_rows.append(row)

    if not filtered_rows:
        raise FileNotFoundError("No summary rows match LOOKBACK=96 and LOOK_FORWARD=12.")

    return filtered_rows


def validate_site_summaries(summary_rows, site_dir_map):
    grouped = defaultdict(dict)
    for row in summary_rows:
        grouped[row["site_id"]][row["model_name"]] = row

    for site_id, model_rows in grouped.items():
        site_dir = site_dir_map[site_id]
        site_summary_path = site_dir / "site_summary_metrics.csv"
        site_rows = read_csv_rows(site_summary_path)
        for site_row in site_rows:
            if int(site_row["lookback"]) != TARGET_LOOKBACK or int(site_row["look_forward"]) != TARGET_LOOK_FORWARD:
                continue
            model_name = site_row["model_name"]
            if model_name not in model_rows:
                continue
            for metric_key in ("mse", "mae", "r2"):
                site_value = float(site_row[metric_key])
                summary_value = model_rows[model_name][metric_key]
                if abs(site_value - summary_value) > 1e-6:
                    raise ValueError(
                        f"Metric mismatch for site {site_id}, model {model_name}, metric {metric_key}: "
                        f"{site_value} != {summary_value}"
                    )


def build_metric_maps(summary_rows):
    by_site = defaultdict(dict)
    by_model = defaultdict(list)
    for row in summary_rows:
        by_site[row["site_id"]][row["model_name"]] = row
        by_model[row["model_name"]].append(row)
    return by_site, by_model


def compute_model_averages(by_model):
    averages = {}
    for model_name, rows in by_model.items():
        averages[model_name] = {
            "mse": sum(row["mse"] for row in rows) / len(rows),
            "mae": sum(row["mae"] for row in rows) / len(rows),
            "r2": sum(row["r2"] for row in rows) / len(rows),
            "wins": 0,
        }
    return averages


def compute_site_winners(by_site, averages):
    winners = {}
    for site_id, model_rows in by_site.items():
        best_row = min(model_rows.values(), key=lambda row: row["mse"])
        winners[site_id] = best_row["model_name"]
        averages[best_row["model_name"]]["wins"] += 1
    return winners


def select_representative_sites(by_site):
    hybrid_candidate = max(
        (
            (site_id, min(row["mse"] for model, row in site_rows.items() if model != "CNN_LSTM") - site_rows["CNN_LSTM"]["mse"])
            for site_id, site_rows in by_site.items()
            if "CNN_LSTM" in site_rows and site_rows["CNN_LSTM"]["mse"] < min(row["mse"] for model, row in site_rows.items() if model != "CNN_LSTM")
        ),
        key=lambda item: item[1],
    )[0]

    lstm_close_candidates = [
        (site_id, abs(site_rows["LSTM"]["mse"] - site_rows["CNN_LSTM"]["mse"]))
        for site_id, site_rows in by_site.items()
        if site_rows["LSTM"]["mse"] <= site_rows["CNN_LSTM"]["mse"]
    ]
    close_site = min(lstm_close_candidates, key=lambda item: item[1])[0]

    cnn_weak_site = min(by_site, key=lambda site_id: by_site[site_id]["CNN"]["r2"])

    lstm_best_candidates = [
        site_id
        for site_id, site_rows in by_site.items()
        if site_rows["LSTM"]["mse"] <= min(site_rows["CNN"]["mse"], site_rows["CNN_LSTM"]["mse"]) and site_id != close_site
    ]
    lstm_general_site = max(lstm_best_candidates, key=lambda site_id: by_site[site_id]["LSTM"]["r2"])

    return {
        "hybrid_best_site": hybrid_candidate,
        "close_site": close_site,
        "cnn_weak_site": cnn_weak_site,
        "lstm_general_site": lstm_general_site,
    }


def compute_horizon_summary(site_dir_map):
    horizon_stats = defaultdict(lambda: {"step1_mae": [], "step12_mae": [], "step1_r2": [], "step12_r2": []})
    for site_id, site_dir in site_dir_map.items():
        for model_name in MODEL_ORDER:
            path = site_dir / model_name / "metrics_by_horizon.csv"
            if not path.exists():
                continue
            rows = read_csv_rows(path)
            step_map = {int(row["horizon_step"]): row for row in rows}
            if 1 in step_map and TARGET_LOOK_FORWARD in step_map:
                horizon_stats[model_name]["step1_mae"].append(float(step_map[1]["mae"]))
                horizon_stats[model_name]["step12_mae"].append(float(step_map[TARGET_LOOK_FORWARD]["mae"]))
                horizon_stats[model_name]["step1_r2"].append(float(step_map[1]["r2"]))
                horizon_stats[model_name]["step12_r2"].append(float(step_map[TARGET_LOOK_FORWARD]["r2"]))

    summary = {}
    for model_name, stats in horizon_stats.items():
        summary[model_name] = {
            key: (sum(values) / len(values) if values else None)
            for key, values in stats.items()
        }
    return summary


def collect_figure_plan(site_dir_map, representative_sites):
    hybrid_site = representative_sites["hybrid_best_site"]
    close_site = representative_sites["close_site"]
    lstm_general_site = representative_sites["lstm_general_site"]
    cnn_weak_site = representative_sites["cnn_weak_site"]

    plan = {
        f"site{hybrid_site}_model_compare.png": site_dir_map[hybrid_site] / f"model_comparison_lb{TARGET_LOOKBACK}_lf{TARGET_LOOK_FORWARD}.png",
        f"site{hybrid_site}_cnn_pred.png": site_dir_map[hybrid_site] / "CNN" / f"test_true_vs_pred_lb{TARGET_LOOKBACK}_lf{TARGET_LOOK_FORWARD}.png",
        f"site{hybrid_site}_cnn_lstm_pred.png": site_dir_map[hybrid_site] / "CNN_LSTM" / f"test_true_vs_pred_lb{TARGET_LOOKBACK}_lf{TARGET_LOOK_FORWARD}.png",
        f"site{hybrid_site}_cnn_lstm_zoom.png": site_dir_map[hybrid_site] / "CNN_LSTM" / f"test_zoom_lb{TARGET_LOOKBACK}_lf{TARGET_LOOK_FORWARD}.png",
        f"site{hybrid_site}_cnn_lstm_error.png": site_dir_map[hybrid_site] / "CNN_LSTM" / f"test_error_curve_lb{TARGET_LOOKBACK}_lf{TARGET_LOOK_FORWARD}.png",
        f"site{hybrid_site}_cnn_lstm_scatter.png": site_dir_map[hybrid_site] / "CNN_LSTM" / f"test_scatter_lb{TARGET_LOOKBACK}_lf{TARGET_LOOK_FORWARD}.png",
        f"site{close_site}_lstm_pred.png": site_dir_map[close_site] / "LSTM" / f"test_true_vs_pred_lb{TARGET_LOOKBACK}_lf{TARGET_LOOK_FORWARD}.png",
        f"site{lstm_general_site}_lstm_pred.png": site_dir_map[lstm_general_site] / "LSTM" / f"test_true_vs_pred_lb{TARGET_LOOKBACK}_lf{TARGET_LOOK_FORWARD}.png",
        f"site{cnn_weak_site}_cnn_lstm_pred.png": site_dir_map[cnn_weak_site] / "CNN_LSTM" / f"test_true_vs_pred_lb{TARGET_LOOKBACK}_lf{TARGET_LOOK_FORWARD}.png",
    }
    return plan


def copy_selected_figures(figure_plan):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for old_file in FIGURE_DIR.iterdir():
        if old_file.is_file():
            old_file.unlink()
    copied = {}
    for dest_name, src_path in figure_plan.items():
        if not src_path.exists():
            raise FileNotFoundError(f"Required figure does not exist: {src_path}")
        dest_path = FIGURE_DIR / dest_name
        shutil.copy2(src_path, dest_path)
        copied[dest_name] = dest_path
    return copied


def build_metrics_table(by_site):
    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \caption{六个风场在不同模型下的预测指标对比}",
        r"    \label{tab:case_metrics_by_site}",
        r"    \resizebox{\textwidth}{!}{%",
        r"    \begin{tabular}{lccccccccc}",
        r"        \hline",
        r"        \multirow{2}{*}{风场} & \multicolumn{3}{c}{CNN} & \multicolumn{3}{c}{LSTM} & \multicolumn{3}{c}{CNN-LSTM} \\",
        r"        & MSE & MAE & $R^2$ & MSE & MAE & $R^2$ & MSE & MAE & $R^2$ \\",
        r"        \hline",
    ]
    for site_id in sorted(by_site):
        site_rows = by_site[site_id]
        lines.append(
            "        "
            + f"风场{site_id}"
            + " & "
            + " & ".join(
                format_metric(site_rows[model_name][metric_key], 3)
                for model_name in ("CNN", "LSTM", "CNN_LSTM")
                for metric_key in ("mse", "mae", "r2")
            )
            + r" \\"
        )
    lines.extend(
        [
            r"        \hline",
            r"    \end{tabular}%",
            r"    }",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def build_average_table(averages):
    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \caption{三种模型跨风场平均指标与最优次数统计}",
        r"    \label{tab:case_metrics_avg}",
        r"    \begin{tabular}{lcccc}",
        r"        \hline",
        r"        模型 & 平均MSE & 平均MAE & 平均$R^2$ & 最优次数 \\",
        r"        \hline",
    ]
    for model_name in MODEL_ORDER:
        row = averages[model_name]
        lines.append(
            "        "
            + f"{MODEL_DISPLAY[model_name]} & {format_metric(row['mse'])} & {format_metric(row['mae'])} & "
            + f"{format_metric(row['r2'], 4)} & {row['wins']}"
            + r" \\"
        )
    lines.extend(
        [
            r"        \hline",
            r"    \end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def build_horizon_sentence(horizon_summary):
    if not horizon_summary:
        return ""
    parts = []
    for model_name in ("CNN", "LSTM", "CNN_LSTM"):
        stats = horizon_summary.get(model_name)
        if not stats:
            continue
        parts.append(
            f"{MODEL_DISPLAY[model_name]}模型的平均MAE由第1步的{format_metric(stats['step1_mae'])}"
            f"增加到第{TARGET_LOOK_FORWARD}步的{format_metric(stats['step12_mae'])}，"
            f"平均$R^2$由{format_metric(stats['step1_r2'], 4)}下降到{format_metric(stats['step12_r2'], 4)}"
        )
    if not parts:
        return ""
    return "此外，分步预测结果表明，随着预测步长增加，三种模型的误差均呈上升趋势，具体表现为：" + "；".join(parts) + "。"


def build_chapter_content(by_site, averages, representative_sites, horizon_summary):
    hybrid_site = representative_sites["hybrid_best_site"]
    close_site = representative_sites["close_site"]
    cnn_weak_site = representative_sites["cnn_weak_site"]
    lstm_general_site = representative_sites["lstm_general_site"]

    hybrid_rows = by_site[hybrid_site]
    close_rows = by_site[close_site]
    weak_rows = by_site[cnn_weak_site]
    general_rows = by_site[lstm_general_site]

    hybrid_gain_vs_lstm = (hybrid_rows["LSTM"]["mse"] - hybrid_rows["CNN_LSTM"]["mse"]) / hybrid_rows["LSTM"]["mse"] * 100
    hybrid_gain_vs_cnn = (hybrid_rows["CNN"]["mse"] - hybrid_rows["CNN_LSTM"]["mse"]) / hybrid_rows["CNN"]["mse"] * 100

    horizon_sentence = build_horizon_sentence(horizon_summary)

    metrics_table = build_metrics_table(by_site)
    averages_table = build_average_table(averages)

    return f"""% 第6章 算例分析
\\section{{算例分析}}

\\subsection{{实验方案设计}}
本章以 \\texttt{{C:\\\\Users\\\\Don\\\\Documents\\\\毕设\\\\thesis111\\\\代码\\\\results}} 中的实验结果为依据，对三种深度学习模型在六个风场上的多步预测表现进行算例分析。为避免不同实验配置混杂，正文仅保留汇总表 \\texttt{{summary\\_metrics.csv}} 中对应的一组正式实验结果，即输入窗口长度为 {TARGET_LOOKBACK}、预测步长为 {TARGET_LOOK_FORWARD} 的实验设置。参与比较的模型包括 CNN、LSTM 和 CNN-LSTM，训练批大小为 64，学习率为 0.001，隐藏层维度为 128，循环层数为 4，训练轮数为 10，Dropout 取 0.4。训练集、验证集和测试集划分比例分别为 0.70、0.15 和 0.15。

从汇总结果可以看出，本次算例共覆盖 6 个风场、3 个模型和 18 条有效结果记录，因而能够同时从同场景横向比较和跨场景纵向比较两个层面评估模型性能。为了保证图表的代表性，后续分析优先选取模型差异明显的典型风场图像，并结合总体统计表进行归纳讨论。

\\subsection{{不同模型在同一风场上的预测结果对比}}
表~\\ref{{tab:case_metrics_by_site}} 给出了六个风场在三种模型下的 MSE、MAE 和 $R^2$ 指标。整体上看，CNN 模型在所有风场中均未取得最优结果；LSTM 在风场2、风场4和风场6上取得最低 MSE；CNN-LSTM 在风场1、风场3和风场5上取得最低 MSE。这说明在当前数据集上，纯卷积结构对长时间依赖的刻画能力仍然不足，而 LSTM 与 CNN-LSTM 则表现出更强的竞争力。

{metrics_table}

在同一风场的横向比较中，风场{hybrid_site}最具有代表性。由图~\\ref{{fig:site{hybrid_site}_model_compare}} 可见，CNN-LSTM 在该风场上的综合表现明显优于其他两种模型。其中，CNN-LSTM 的 MSE 为 {format_metric(hybrid_rows["CNN_LSTM"]["mse"])}, MAE 为 {format_metric(hybrid_rows["CNN_LSTM"]["mae"])}, $R^2$ 为 {format_metric(hybrid_rows["CNN_LSTM"]["r2"], 4)}；相比 LSTM，MSE 下降了 {format_metric(hybrid_gain_vs_lstm, 2)}\\%；相比 CNN，MSE 下降了 {format_metric(hybrid_gain_vs_cnn, 2)}\\%。从图~\\ref{{fig:site{hybrid_site}_cnn_lstm_pred}} 中左右两幅子图的对比也可以看出，CNN-LSTM 对功率峰谷变化的跟踪更紧密，而 CNN 在该风场上存在明显的振幅偏差和趋势偏移，这与其在该风场上的负 $R^2$ 结果相一致。

\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.72\\textwidth]{{figure/case_analysis/site{hybrid_site}_model_compare.png}}
    \\caption{{风场{hybrid_site}在三种模型下的预测指标对比}}
    \\label{{fig:site{hybrid_site}_model_compare}}
\\end{{figure}}

\\begin{{figure}}[H]
    \\centering
    \\begin{{minipage}}{{0.48\\textwidth}}
        \\centering
        \\includegraphics[width=\\linewidth]{{figure/case_analysis/site{hybrid_site}_cnn_lstm_pred.png}}
    \\end{{minipage}}
    \\hfill
    \\begin{{minipage}}{{0.48\\textwidth}}
        \\centering
        \\includegraphics[width=\\linewidth]{{figure/case_analysis/site{hybrid_site}_cnn_pred.png}}
    \\end{{minipage}}
    \\caption{{风场{hybrid_site}中 CNN-LSTM 与 CNN 的预测结果对比。左图为 CNN-LSTM，右图为 CNN。}}
    \\label{{fig:site{hybrid_site}_cnn_lstm_pred}}
\\end{{figure}}

\\subsection{{同一模型在不同风场上的泛化性能分析}}
从跨风场结果看，LSTM 与 CNN-LSTM 的优劣具有明显的场景依赖性。风场{close_site}中，LSTM 的 MSE 为 {format_metric(close_rows["LSTM"]["mse"])}, 略优于 CNN-LSTM 的 {format_metric(close_rows["CNN_LSTM"]["mse"])}, 且二者的 $R^2$ 均处于 0.90 左右，说明在该类时间相关性较强、模式相对稳定的风场中，单纯依靠循环结构已能取得较好的预测效果。另一方面，在风场{hybrid_site}中，CNN-LSTM 的优势更加突出，说明卷积前端提取局部变化模式后再交由 LSTM 建模，有助于缓解复杂场景下的误差积累。风场{lstm_general_site}则表明，当序列规律较为清晰时，LSTM 仍具有较强的泛化能力。

\\begin{{figure}}[H]
    \\centering
    \\begin{{minipage}}{{0.32\\textwidth}}
        \\centering
        \\includegraphics[width=\\linewidth]{{figure/case_analysis/site{close_site}_lstm_pred.png}}
    \\end{{minipage}}
    \\hfill
    \\begin{{minipage}}{{0.32\\textwidth}}
        \\centering
        \\includegraphics[width=\\linewidth]{{figure/case_analysis/site{hybrid_site}_cnn_lstm_pred.png}}
    \\end{{minipage}}
    \\hfill
    \\begin{{minipage}}{{0.32\\textwidth}}
        \\centering
        \\includegraphics[width=\\linewidth]{{figure/case_analysis/site{lstm_general_site}_lstm_pred.png}}
    \\end{{minipage}}
    \\caption{{不同风场上的代表性预测结果。左图为风场{close_site}的 LSTM 结果，中图为风场{hybrid_site}的 CNN-LSTM 结果，右图为风场{lstm_general_site}的 LSTM 结果。}}
    \\label{{fig:cross_site_prediction_examples}}
\\end{{figure}}

从最优次数统计来看，LSTM 与 CNN-LSTM 各自获得 3 次单风场最优结果，而 CNN 没有取得单场景最优。这说明混合模型并非在所有风场上都绝对占优，但其稳定性和适应复杂工况的能力更强；同时，LSTM 在若干风场上的突出表现也表明，时间依赖关系仍然是影响风电功率预测精度的核心因素之一。

\\subsection{{预测结果可视化分析}}
为了进一步观察模型的拟合细节，本节选取风场{hybrid_site}中 CNN-LSTM 的整体预测曲线、局部放大图、误差曲线和散点图进行可视化分析，如图~\\ref{{fig:site{hybrid_site}_visuals}} 所示。整体曲线显示，模型能够较好地跟踪功率变化趋势；在局部放大图中，预测曲线对波峰波谷具有较好的同步能力，但在快速上升或快速下降区间仍存在一定的平滑现象。误差曲线围绕零值上下波动，未出现长期单侧偏移，说明模型总体不存在明显系统性偏差。散点图中样本点大多分布在对角线附近，进一步说明预测值与真实值之间具有较强的一致性。

\\begin{{figure}}[H]
    \\centering
    \\begin{{minipage}}{{0.48\\textwidth}}
        \\centering
        \\includegraphics[width=\\linewidth]{{figure/case_analysis/site{hybrid_site}_cnn_lstm_pred.png}}
    \\end{{minipage}}
    \\hfill
    \\begin{{minipage}}{{0.48\\textwidth}}
        \\centering
        \\includegraphics[width=\\linewidth]{{figure/case_analysis/site{hybrid_site}_cnn_lstm_zoom.png}}
    \\end{{minipage}}\\\\[0.6em]
    \\begin{{minipage}}{{0.48\\textwidth}}
        \\centering
        \\includegraphics[width=\\linewidth]{{figure/case_analysis/site{hybrid_site}_cnn_lstm_error.png}}
    \\end{{minipage}}
    \\hfill
    \\begin{{minipage}}{{0.48\\textwidth}}
        \\centering
        \\includegraphics[width=\\linewidth]{{figure/case_analysis/site{hybrid_site}_cnn_lstm_scatter.png}}
    \\end{{minipage}}
    \\caption{{风场{hybrid_site}中 CNN-LSTM 模型的可视化结果。左上为整体预测图，右上为局部放大图，左下为误差曲线，右下为散点图。}}
    \\label{{fig:site{hybrid_site}_visuals}}
\\end{{figure}}

\\subsection{{\\texorpdfstring{{基于 MSE、MAE、R\\textsuperscript{{2}} 的综合评价}}{{基于 MSE、MAE、R2 的综合评价}}}}
为了从全局角度比较模型性能，表~\\ref{{tab:case_metrics_avg}} 给出了三种模型在六个风场上的平均指标与最优次数统计。结果表明，CNN-LSTM 的平均 MAE 最低，为 {format_metric(averages["CNN_LSTM"]["mae"])}, 平均 $R^2$ 最高，为 {format_metric(averages["CNN_LSTM"]["r2"], 4)}；LSTM 的平均 MSE 最低，为 {format_metric(averages["LSTM"]["mse"])}, 且在 3 个风场上取得单场景最优结果。相比之下，CNN 的平均 MSE 和平均 MAE 均最高，平均 $R^2$ 仅为 {format_metric(averages["CNN"]["r2"], 4)}，表现明显落后于另外两种模型。

{averages_table}

{horizon_sentence}

综合来看，若更强调平均绝对误差控制和整体拟合能力，CNN-LSTM 更具优势；若将均方误差最小化作为首要目标，LSTM 仍具有较强竞争力。因此，两类模型均可作为风电功率预测的有效方案，但 CNN-LSTM 的综合表现更加均衡。

\\subsection{{模型优缺点与误差来源分析}}
从算例结果可以总结出三类模型的特点。CNN 模型结构较为简洁，训练时间较短，但由于其主要依赖局部感受野提取时间片段特征，对长期依赖关系和复杂非线性波动的刻画能力有限，因此在多个风场中出现拟合不足现象，尤其在风场{cnn_weak_site}中表现最差。LSTM 模型能够较好地记忆时间依赖关系，在风场2、风场4和风场6上均取得较优结果，说明对于时序规律相对稳定的风场，其建模能力较强。CNN-LSTM 则兼具局部模式提取与长期依赖建模能力，在风场1、风场3和风场{hybrid_site}上取得最优结果，显示出较好的综合适应性。

误差来源主要包括以下几个方面：其一，风电功率本身受风速、风向、温度、气压和湿度等多因素共同影响，且这些变量之间存在强烈的时变耦合关系，导致输入与输出之间呈现复杂非线性映射；其二，极端天气、突发风速波动以及机组运行状态变化会带来短时异常，使模型在局部区间出现跟踪偏差；其三，不同风场的数据分布和功率曲线差异较大，使得模型在跨风场场景下难以保持完全一致的泛化效果；其四，当前实验仍采用统一的网络规模和训练轮数，不同模型尚未针对每个风场做更细致的参数优化，这也会造成部分性能差异。

\\subsection{{最优模型选择与适用场景讨论}}
从单风场最优次数看，LSTM 与 CNN-LSTM 各有 3 次最优，说明两者都具备较强的工程应用价值；从跨风场平均指标看，CNN-LSTM 在平均 MAE 和平均 $R^2$ 上更优，说明其整体拟合稳定性更强。考虑到风电功率预测任务既要求较好的趋势跟踪能力，又要求在复杂场景下保持较强鲁棒性，本文将 CNN-LSTM 作为推荐模型；同时，LSTM 可作为一个性能稳定、实现相对简洁的重要对照模型。

具体而言，当风场数据具有更明显的局部波动特征或多因素耦合效应较强时，CNN-LSTM 更适合作为优先方案；当时间序列规律较清晰、希望在较低结构复杂度下获得较好结果时，LSTM 也是可行的选择。相比之下，纯 CNN 模型更适合作为基线模型，而不宜作为当前数据集上的最终推荐方案。
"""


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    site_dir_map = load_site_dir_map()
    summary_rows = load_summary_rows()
    validate_site_summaries(summary_rows, site_dir_map)

    by_site, by_model = build_metric_maps(summary_rows)
    averages = compute_model_averages(by_model)
    compute_site_winners(by_site, averages)
    representative_sites = select_representative_sites(by_site)
    horizon_summary = compute_horizon_summary(site_dir_map)

    figure_plan = collect_figure_plan(site_dir_map, representative_sites)
    copy_selected_figures(figure_plan)

    chapter_content = build_chapter_content(by_site, averages, representative_sites, horizon_summary)
    CHAPTER_PATH.write_text(chapter_content, encoding="utf-8")

    print("Chapter 6 generated successfully.")
    print(f"Representative sites: {representative_sites}")
    print(f"Figures copied to: {FIGURE_DIR}")
    print(f"Chapter written to: {CHAPTER_PATH}")


if __name__ == "__main__":
    main()
