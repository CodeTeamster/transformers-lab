import os
import json
import re
import csv
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


from typing import List, Tuple, Optional
from matplotlib.colors import LogNorm


def plot_performance(
    perf_dir: str = "./workdir/performance",
    file_regex: str = r"performance_tome_r-(\d+)\.json",
    save_path: str = "./workdir/performance/performance.png",
):
    pattern = re.compile(file_regex)
    indices, flops, accuracy, throughput = [], [], [], []

    assert os.path.exists(perf_dir), f"Performance directory {perf_dir} does not exist."

    for fname in os.listdir(perf_dir):
        match = pattern.match(fname)
        if match:
            i = int(match.group(1))
            with open(os.path.join(perf_dir, fname), "r") as f:
                data = json.load(f)
            indices.append(i)
            # 去掉单位，只保留数字
            flops.append(float(data["flops"].replace("G", "")))
            accuracy.append(float(data["accuracy"])*100)
            throughput.append(float(data["throughput"]))

    sorted_data = sorted(zip(indices, flops, accuracy, throughput))
    indices, flops, accuracy, throughput = map(list, zip(*sorted_data))

    plt.figure(figsize=(10, 6))
    # gca() 获取当前的坐标轴对象
    ax1 = plt.gca()
    # 创建共享x轴的第二个y轴
    ax2 = ax1.twinx()
    ax1.plot(indices, flops, 'b--', label="FLOPs (g)")
    ax1.plot(indices, throughput, 'b-', label="Throughput (im/s)")
    ax2.plot(indices, accuracy, 'r-', label="Accuracy (%)")

    ax1.set_xlabel("r")
    ax1.set_ylabel("FLOPs / Throughput", color='b')
    ax2.set_ylabel("Accuracy", color='r')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    ax2.set_ylim(55.0, 90.0)
    plt.title("Token Merging Performance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.savefig(save_path)


def plot_performance2(
    perf_dirs: List[Tuple[str, str]] = [
        ('deit-tiny-patch16-224-fb-in1k', 'vit-tiny-patch16-224-augreg-in21k-ft-in1k'),
        ('deit-small-patch16-224-fb-in1k', 'vit-small-patch16-224-augreg-in1k'),
        ('deit-base-patch16-224-fb-in1k', 'vit-base-patch16-224-augreg-in1k'),
        ('deit3-large-patch16-224-fb-in22k-ft-in1k', 'vit-large-patch16-224-augreg-in21k-ft-in1k'),
    ],
    file_regex: str = r"performance_tome_r-(\d+)\.json",
    save_path: str = "./workdir/performance.png",
):
    pattern = re.compile(file_regex)
    list_len = len(perf_dirs)
    tome_r = [{'origin': [], 'augreg': []} for _ in range(list_len)]
    acc = [{'origin': [], 'augreg': []} for _ in range(list_len)]

    for i, tp in enumerate(perf_dirs):
        origin = './workdir/' + tp[0] + '.perf'
        assert os.path.exists(origin), f"Performance directory {origin} does not exist."
        for fname in os.listdir(origin):
            match = pattern.match(fname)
            if match:
                r = int(match.group(1))
                with open(os.path.join(origin, fname), "r") as f:
                    data = json.load(f)
                tome_r[i]['origin'].append(r)
                acc[i]['origin'].append(float(data["accuracy"]))
        sorted_data = sorted(zip(tome_r[i]['origin'], acc[i]['origin']))
        tome_r[i]['origin'], acc[i]['origin'] = map(list, zip(*sorted_data))

        augreg = './workdir/' + tp[1] + '.perf'
        assert os.path.exists(augreg), f"Performance directory {augreg} does not exist."
        for fname in os.listdir(augreg):
            match = pattern.match(fname)
            if match:
                r = int(match.group(1))
                with open(os.path.join(augreg, fname), "r") as f:
                    data = json.load(f)
                tome_r[i]['augreg'].append(r)
                acc[i]['augreg'].append(float(data["accuracy"]))
        sorted_data = sorted(zip(tome_r[i]['augreg'], acc[i]['augreg']))
        tome_r[i]['augreg'], acc[i]['augreg'] = map(list, zip(*sorted_data))

    color_list = ["black", "blue", "red", "orange", "purple", "cyan", "magenta", "green"]
    plt.figure(figsize=(10, 6))
    # gca() 获取当前的坐标轴对象
    ax1 = plt.gca()
    for i in range(list_len):
        ax1.plot(tome_r[i]['origin'], acc[i]['origin'], color=color_list[i], linestyle='-', label=f"{perf_dirs[i][0]}")
        ax1.plot(tome_r[i]['augreg'], acc[i]['augreg'], color=color_list[i], linestyle='-.', label=f"{perf_dirs[i][1]}")

    ax1.set_xlabel("r")
    ax1.set_ylabel("Accuracy", color='black')
    ax1.legend()
    ax1.set_ylim(30.0, 90.0)
    plt.title("ToMe Accuracy Comparision")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.savefig(save_path)


def plot_from_csv(
    csv_path: str = "./checkpoints/my-vit-small-patch16-224/20250806-181915-vit_small_patch16_224/summary.csv",
    save_path: str = "./checkpoints/my-vit-small-patch16-224/20250806-181915-vit_small_patch16_224/summary.png",
):
    epochs = []
    train_loss = []
    eval_loss = []

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_loss.append(float(row['train_loss']))
            eval_loss.append(float(row['eval_loss']))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label='Train Loss', color='blue')
    plt.plot(epochs, eval_loss, label='Eval Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs Eval Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path)


def is_line_duplicate(ax, x_data, y_data):
    for line in ax.get_lines():
        existing_x, existing_y = line.get_xdata(), line.get_ydata()
        if (existing_x == x_data).all() and (existing_y == y_data).all():
            return True
    return False


def plot_multi_performance(
    baseline_file: Optional[str] = None,
    dirs_and_files_regex: List[Tuple[str, str]] = [
        ("./workdir/vit-base-patch16-224.perf", r"tome-(\d+)_discard-([\d.]+)\.json"),
        ("./workdir/vit-base-patch16-224-sup-6-discard-0.3.perf", r"tome-(\d+)_discard-([\d.]+)\.json"),
    ],
    save_path: str = "./workdir/multi-perf.png",
    title: str = "Random Discard Performance",
    x_label: str = "Discard Rate",
    indices_range: Tuple[float, float] = (0.0, 0.9),
    accuracy_range: Tuple[float, float] = (0.0, 90.0),
    gflops_range: Tuple[float, float] = (0.0, 20.0),
    baseline_x: Optional[float] = None,
    baseline_y: Optional[float] = None,
):
    color_list = [
        "black", "blue", "red", "orange", "purple", "pink", "violet", "green",
        "gray", "yellow", "cyan", "lime", "teal", "navy", "magenta", "gold",
        "indigo", "brown", "turquoise", "darkgreen",
    ]
    plt.figure(figsize=(16, 10))
    ax1 = plt.gca()
    ax1.set_ylim(*accuracy_range)
    ax2 = ax1.twinx()
    ax2.set_ylim(*gflops_range)
    ax1.set_xlabel(x_label)
    ax1.set_xlim(*indices_range)

    ax1.set_ylabel("Accuracy (%)")
    ax2.set_ylabel("Flops (G)")
    if baseline_x is not None:
        ax1.axvline(x=baseline_x, color="gray", linestyle="--", linewidth=1, label="50% increase in speed")
    if baseline_y is not None:
        ax1.axhline(y=baseline_y, color="gray", linestyle="--", linewidth=1, label="1% decline in accuracy")

    for i, dir_and_file in enumerate(dirs_and_files_regex):
        # 1. Load data
        indices, flops, accuracy = [], [], []
        perf_dir = dir_and_file[0]
        assert os.path.exists(perf_dir), f"Directory {perf_dir} does not exist."
        file_regex = dir_and_file[1]

        if baseline_file is not None:
            with open(os.path.join(perf_dir, baseline_file), "r") as f:
                data = json.load(f)
            indices.append(0.0)
            flops.append(float(data["flops"].replace("G", "")))
            accuracy.append(float(data["accuracy"])*100)

        pattern = re.compile(file_regex)
        for fname in os.listdir(perf_dir):
            match = pattern.match(fname)
            if match:
                indice = float(match.group(1))
                with open(os.path.join(perf_dir, fname), "r") as f:
                    data = json.load(f)
                indices.append(indice)
                flops.append(float(data["flops"].replace("G", "")))
                accuracy.append(float(data["accuracy"])*100)

        # 2. Plot
        sorted_data = sorted(zip(indices, flops, accuracy))
        indices, flops, accuracy = map(list, zip(*sorted_data))
        ax1.plot(
            indices,
            accuracy,
            color=color_list[i],
            linestyle='-',
            label=f"Accuracy of {os.path.splitext(os.path.basename(perf_dir))[0]}",
        )
        if not is_line_duplicate(ax2, indices, flops):
            ax2.plot(
                indices,
                flops,
                color=color_list[i],
                linestyle='--',
                label=f"Flops of {os.path.splitext(os.path.basename(perf_dir))[0]}",
            )

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    ax1.legend(handles, labels, loc='lower left')

    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path)


def plot_acc_flops(
    baseline_file: Optional[str] = None,
    dirs_and_files_regex: List[Tuple[str, str]] = [
        ("./workdir/vit-base-patch16-224.perf", r"tome-(\d+)_discard-([\d.]+)\.json"),
        ("./workdir/vit-base-patch16-224-sup-6-discard-0.3.perf", r"tome-(\d+)_discard-([\d.]+)\.json"),
    ],
    save_path: str = "./workdir/multi-perf.png",
    title: str = "Random Discard Performance",
    accuracy_range: Tuple[float, float] = (0.0, 90.0),
    gflops_range: Tuple[float, float] = (0.0, 20.0),
):
    color_list = [
        "black", "blue", "red", "orange", "purple", "pink", "violet", "green",
        "gray", "yellow", "cyan", "lime", "teal", "navy", "magenta", "gold",
        "indigo", "brown", "turquoise", "darkgreen",
    ]
    plt.figure(figsize=(16, 10))
    ax1 = plt.gca()
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(*accuracy_range)
    ax1.set_xlabel("Flops (G)")
    ax1.set_xlim(*gflops_range)

    for i, dir_and_file in enumerate(dirs_and_files_regex):
        # 1. Load data
        flops, accuracy = [], []
        perf_dir = dir_and_file[0]
        assert os.path.exists(perf_dir), f"Directory {perf_dir} does not exist."
        file_regex = dir_and_file[1]

        if baseline_file is not None:
            with open(os.path.join(perf_dir, baseline_file), "r") as f:
                data = json.load(f)
            flops.append(float(data["flops"].replace("G", "")))
            accuracy.append(float(data["accuracy"])*100)

        pattern = re.compile(file_regex)
        for fname in os.listdir(perf_dir):
            match = pattern.match(fname)
            if match:
                with open(os.path.join(perf_dir, fname), "r") as f:
                    data = json.load(f)
                flops.append(float(data["flops"].replace("G", "")))
                accuracy.append(float(data["accuracy"])*100)

        # 2. Plot
        sorted_data = sorted(zip(flops, accuracy))
        flops, accuracy = map(list, zip(*sorted_data))
        ax1.plot(
            flops,
            accuracy,
            color=color_list[i],
            linestyle='-',
            label=f"Accuracy of {os.path.splitext(os.path.basename(perf_dir))[0]}",
        )

    handles1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(handles1, labels1, loc='lower left')

    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path)


def plot_lmms_eval_res(
    dirs_and_files_regex: List[Tuple[str, str]] = [
        ("./workdir/lmms-eval", r"llava_1.5_7b_fp16_discard-([\d.]+)_layer-0"),
        ("./workdir/lmms-eval", r"llava_1.5_7b_fp16_discard-([\d.]+)_layer-23"),
    ],
    save_path: str = "./workdir/lmms-eval/llava_1.5_7b_fp16_discard.png",
    title: str = "Random Discard Performance",
    x_label: str = "Discard Rate",
    indices_range: Tuple[float, float] = (0.00, 0.95),
    mme_cognition_range: Tuple[float, float] = (200.0, 400.0),
    mme_perception_range: Tuple[float, float] = (600.0, 1500.0),
    tflops_range: Tuple[float, float] = (0.0, 20.0),
):
    color_list = [
        "black", "blue", "red", "orange", "purple", "pink", "violet", "green",
        "gray", "yellow", "cyan", "lime", "teal", "navy", "magenta", "gold",
        "indigo", "brown", "turquoise", "darkgreen",
    ]
    plt.figure(figsize=(16, 10))
    ax1 = plt.gca()
    ax1.set_ylim(*mme_cognition_range)
    ax2 = ax1.twinx()
    ax2.set_ylim(*mme_perception_range)

    ax1.set_xlabel(x_label)
    ax1.set_xlim(*indices_range)
    xticks = np.arange(indices_range[0], indices_range[1] + 0.001, 0.05)
    ax1.set_xticks(xticks)

    ax1.set_ylabel("MME Cognition Score")
    ax2.set_ylabel("MME Perception Score")

    for i, dir_and_file in enumerate(dirs_and_files_regex):
        # 1. Load data
        indices, mme_cognition, mme_perception = [], [], []
        first_dir = dir_and_file[0]
        assert os.path.exists(first_dir), f"Directory {first_dir} does not exist."
        second_dir_regex = dir_and_file[1]

        second_dir_pattern = re.compile(second_dir_regex)
        file_pattern = re.compile(r".+_results\.json$")
        for second_dir_name in os.listdir(first_dir):
            match = second_dir_pattern.match(second_dir_name)
            if not match:
                continue
            second_dir = os.path.join(first_dir, second_dir_name, 'llava-hf__llava-1.5-7b-hf')
            if not os.path.isdir(second_dir):
                continue
            for file_name in os.listdir(second_dir):
                if file_pattern.match(file_name):
                    indice = float(match.group(1))
                    with open(os.path.join(second_dir, file_name), "r") as f:
                        data = json.load(f)
                    indices.append(indice)
                    mme_cognition.append(float(data["results"]["mme"]["mme_cognition_score,none"]))
                    mme_perception.append(float(data["results"]["mme"]["mme_perception_score,none"]))

        # 2. Plot
        sorted_data = sorted(zip(indices, mme_cognition, mme_perception))
        indices, mme_cognition, mme_perception = map(list, zip(*sorted_data))
        ax1.plot(
            indices,
            mme_cognition,
            color=color_list[i],
            linestyle='-',
            label=f"MME Cognition of {second_dir_regex}",
        )
        ax2.plot(
            indices,
            mme_perception,
            color=color_list[i],
            linestyle='--',
            label=f"MME Perception of {second_dir_regex}",
        )

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    ax1.legend(handles, labels, loc='lower left')

    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path)


def plot_lmms_eval_norm_res(
    dirs_and_files_regex: List[Tuple[str, str]] = [
        (
            "./workdir/lmms-eval",
            r"llava_1.5_7b_fp16_discard-([\d.]+)_seed-(\d+)_layer-0",
            r".+_results\.json$",
            "llava-hf__llava-1.5-7b-hf",
        ),
        (
            "./workdir/lmms-eval",
            r"qwen_2.5_vl_7b_bf16_discard-([\d.]+)_seed-(\d+)_layer-0",
            r".+_results\.json$",
            "Qwen__Qwen2.5-VL-7B-Instruct",
        ),
        (
            "./workdir/lmms-eval",
            r"intern_3.5_vl_hf_8b_bf16_discard-([\d.]+)_seed-(\d+)_layer-0",
            r".+_results\.json$",
            "OpenGVLab__InternVL3_5-8B-HF",
        ),
    ],
    save_path: str = "./workdir/lmms-eval/llava_1.5_7b_fp16_discard_norm.png",
    title: str = "Random Discard Cognition Performance",
    x_label: str = "Discard Rate",
    y_label: str = "MME Cognition Score",
    perf_keys: list[str] = ["results", "mme", "mme_cognition_score,none"],
    seed_list: List[int] = [i for i in range(1, 22, 2)],
    x_range: Tuple[float, float] = (0.00, 0.95),
    y_range: Tuple[float, float] = (200.0, 400.0),
):
    color_list = [
        "black", "blue", "red", "orange", "purple", "pink", "violet", "green",
        "gray", "yellow", "cyan", "lime", "teal", "navy", "magenta", "gold",
        "indigo", "brown", "turquoise", "darkgreen",
    ]
    plt.figure(figsize=(16, 10))
    ax1 = plt.gca()

    ax1.set_xlabel(x_label)
    ax1.set_xlim(*x_range)
    xticks = np.arange(x_range[0], x_range[1] + 0.001, 0.05)
    ax1.set_xticks(xticks)

    ax1.set_ylim(*y_range)
    ax1.set_ylabel(y_label)

    seed_set = set(seed_list)
    for i, dir_and_file in enumerate(dirs_and_files_regex):
        # 1. Load data
        y_values = {}
        first_dir = dir_and_file[0]
        assert os.path.exists(first_dir), f"Directory {first_dir} does not exist."

        second_dir_regex = dir_and_file[1]
        second_dir_pattern = re.compile(second_dir_regex)
        file_regex = dir_and_file[2]
        file_pattern = re.compile(file_regex)

        for second_dir_name in os.listdir(first_dir):
            match = second_dir_pattern.match(second_dir_name)
            if not match:
                continue
            if int(match.group(2)) not in seed_set:
                continue
            second_dir = os.path.join(first_dir, second_dir_name, dir_and_file[3])
            if not os.path.isdir(second_dir):
                continue
            for file_name in os.listdir(second_dir):
                if file_pattern.match(file_name):
                    with open(os.path.join(second_dir, file_name), "r") as f:
                        data = json.load(f)

                    indice = float(match.group(1))
                    if indice not in y_values:
                        y_values[indice] = []
                    y_values[indice].append(float(data[perf_keys[0]][perf_keys[1]][perf_keys[2]]))

        indices, mu_y, sigma_y = [], [], []
        for indice, y_list in y_values.items():
            indices.append(indice)
            mu_y.append(np.mean(y_list))
            sigma_y.append(np.std(y_list))

        # 2. Plot
        sorted_data = sorted(zip(indices, mu_y, sigma_y))
        indices, mu_y, sigma_y = map(list, zip(*sorted_data))

        ax1.plot(
            indices,
            mu_y,
            color=color_list[i],
            linestyle='-',
            label=f"{y_label} of {second_dir_regex}",
        )
        ax1.fill_between(
            indices,
            np.array(mu_y) + np.array(sigma_y),
            np.array(mu_y) - np.array(sigma_y),
            facecolor=color_list[i],
            alpha=0.4
        )

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc='lower left')

    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path)


def visualize_attention(
    multihead_attention: torch.Tensor,
    output_path: str = None,
    title: str = "Layer 5",
    pool_kernel_size: int = 10,
):
    # Assuming the input is a numpy array of shape (1, num_heads, n_tokens, n_tokens)
    # First, we average the attention scores over the multiple heads
    averaged_attention = torch.mean(
        multihead_attention,
        axis=1,
    )[0].float()# Shape: (n_tokens, n_tokens)

    # pooling the attention scores with stride pool_kernel_size
    averaged_attention = torch.nn.functional.avg_pool2d(
        averaged_attention.unsqueeze(0).unsqueeze(0),
        kernel_size=pool_kernel_size,
        stride=pool_kernel_size,
    ).squeeze(0).squeeze(0)

    if output_path is not None:
        cmap = plt.cm.get_cmap("viridis")
        plt.figure(figsize=(5, 5),dpi=400)

        # Log normalization
        log_norm = LogNorm(vmin=0.0007, vmax=averaged_attention.max())

        # set the x and y ticks to 20x of the original
        ax = sns.heatmap(
            averaged_attention,
            cmap=cmap,  # custom color map
            norm=log_norm,
            # cbar_kws={'label': 'Attention score'},
        )

        # replace the x and y ticks with string
        x_ticks = [str(i * pool_kernel_size) for i in range(0,averaged_attention.shape[0])]
        y_ticks = [str(i * pool_kernel_size) for i in range(0,averaged_attention.shape[0])]
        ax.set_xticks([i for i in range(0,averaged_attention.shape[0])])
        ax.set_yticks([i for i in range(0,averaged_attention.shape[0])])
        ax.set_xticklabels(x_ticks)
        ax.set_yticklabels(y_ticks)

        # change the x tinks font size
        plt.xticks(fontsize=3)
        plt.yticks(fontsize=3)

        # make y label vertical
        plt.yticks(rotation=0)
        plt.xticks(rotation=90)

        plt.title(title)
        # tight layout
        plt.savefig(output_path, bbox_inches='tight')

    top_five_attn = []
    for row in averaged_attention:
        # Use torch.topk to get the top 5 values and their indices
        top_values, top_indices = torch.topk(row, 5 if row.shape[-1] >=5 else row.shape[-1])
        # Convert to lists and append to the overall list
        top_five_layer_attn = torch.stack((top_indices, top_values), dim=-1)
        top_five_attn.append(top_five_layer_attn)

    return torch.stack(top_five_attn), averaged_attention


if __name__ == "__main__":
    # plot_lmms_eval_res()
    # plot_lmms_eval_norm_res(
    #     save_path="./workdir/lmms-eval/intern_llava_qwen_discard_cognition_norm.png",
    #     title="Random Discard Cognition Performance",
    #     x_label="Discard Rate",
    #     y_label="MME Cognition Score",
    #     perf_keys=["results", "mme", "mme_cognition_score,none"],
    #     seed_list=[i for i in range(1, 22, 2)],
    #     x_range=(0.00, 0.95),
    #     y_range=(200.0, 700.0),
    # )
    # plot_lmms_eval_norm_res(
    #     save_path="./workdir/lmms-eval/intern_llava_qwen_discard_perception_norm.png",
    #     title="Random Discard Perception Performance",
    #     x_label="Discard Rate",
    #     y_label="MME Perception Score",
    #     perf_keys=["results", "mme", "mme_perception_score,none"],
    #     seed_list=[i for i in range(1, 22, 2)],
    #     x_range=(0.00, 0.95),
    #     y_range=(600.0, 1800.0),
    # )
    plot_lmms_eval_norm_res(
        dirs_and_files_regex=[
            (
                "./workdir/lm-eval",
                r"mistral_v0.3_7b_bf16_discard-([\d.]+)_seed-(\d+)_layer-0",
                r"^results.*\.json$",
                "__home__jovyan__nas__yrc__model__mistralai__Mistral-7B-Instruct-v0.3",
            ),
            (
                "./workdir/lm-eval",
                r"qwen2_7b_bf16_discard-([\d.]+)_seed-(\d+)_layer-0",
                r"^results.*\.json$",
                "__home__jovyan__nas__yrc__model__Qwen__Qwen2.5-7B",
            ),
            (
                "./workdir/lm-eval",
                r"qwen3_8b_bf16_discard-([\d.]+)_seed-(\d+)_layer-0",
                r"^results.*\.json$",
                "__home__jovyan__nas__yrc__model__Qwen__Qwen3-8B",
            ),
        ],
        save_path="./workdir/lm-eval/mistral_qwen2_qwen3_discard_mmlu_norm.png",
        title="Random Discard MMLU Performance",
        x_label="Discard Rate",
        y_label="Massive Multitask Language Understanding Accuracy",
        perf_keys=["results", "mmlu", "acc,none"],
        seed_list=[i for i in range(1, 22, 2)],
        x_range=(0.00, 0.95),
        y_range=(0.00, 0.95),
    )