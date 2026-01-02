# from src.transformers_qwen2_5_vl_eval import run_eval as qwen_eval, defaultargs as qwen_eval_defaultargs
# from src.transformers_qwen2_5_vl_clip_metric import run_eval as qwen_clip_eval, defaultargs as qwen_clip_eval_defaultargs
# from src.transformers_vit_eval import run_eval as tf_eval, defaultargs as tf_eval_defaultargs
# from src.timm_vit_train import run_train as timm_train, defaultargs as timm_train_defaultargs
# from src.timm_vit_eval import run_eval as timm_eval, defaultargs as timm_eval_defaultargs
# from transformers import ViTImageProcessor, ViTRDForImageClassification, ViTRDConfig
from utils import plot_multi_performance, plot_acc_flops


model = {
    'vit-base-patch16-224': 'google',
    'deit_base_patch16_224.fb_in1k': 'transformers',
    'vit_base_patch16_224.augreg_in1k': 'transformers',
    'vit_base_patch16_224.augreg_in21k_ft_in1k': 'transformers',
    'vit_base_patch16_224.augreg2_in21k_ft_in1k': 'transformers',   # SOTA for ViT
    'deit_base_distilled_patch16_224.fb_in1k': 'timm',
    'deit3_base_patch16_224.fb_in1k': 'timm',
    'deit3_base_patch16_224.fb_in22k_ft_in1k': 'timm',              # SOTA for DeiT
}


discard_regex = r"discard-([\d.]+)\.json"
divprune_regex = r"divprune-([\d.]+)\.json"
evit_regex = r"evit-([\d.]+)\.json"
tome_regex = r"tome-(\d+)\.json"
rate_comparison = [
    ("./workdir/augreg2.sup-0-discard-0.6-layer-40.perf",),
    ("./workdir/augreg2.sup-0-discard-0.6-layer-4.perf",),
]
sup_comparison = [
    ("./workdir/augreg2.sup-0-discard-0.6-layer-40.perf",),
    ("./workdir/augreg2.sup-0-discard-0.6-layer-4.perf",),
]
strategy_comparison = [
    ("./workdir/augreg2.sup-0-discard-0.6-layer-40.perf",),
    ("./workdir/augreg2.sup-0-discard-0.6-layer-4.perf",),
]
custom_comparison = [
    ("./workdir/augreg2.sup-0-discard-0.6-layer-40.perf",),
    ("./workdir/augreg2.sup-0-discard-0.6-layer-4.perf",),
]


if __name__ == "__main__":
    # model = ViTRDConfig()
    # model.save_pretrained("./config.json")
    # processor = ViTImageProcessor()
    # processor.save_pretrained("./preprocessor_config.json")

    plot_parameters = [
        (discard_regex, 'discard', 'Random Discard Performance', 'Discard Rate'),
        (divprune_regex, 'divprune', 'Diversity Pruning Performance', 'DivPrune Rate'),
        (evit_regex, 'evit', 'EViT Pruning Performance', 'EViT Rate'),
    ]
    for parameter in plot_parameters:
        rate_files_regex = [item + (parameter[0],) for item in rate_comparison]
        sup_files_regex = [item + (parameter[0],) for item in sup_comparison]
        strategy_files_regex = [item + (parameter[0],) for item in strategy_comparison]
        custom_files_regex = [item + (parameter[0],) for item in custom_comparison]
        plot_multi_performance(
            baseline_file='normal.json',
            dirs_and_files_regex=rate_files_regex,
            save_path=f'./workdir/{parameter[1]}-augreg2-sup-discard*-layer.png',
            title=parameter[2],
            x_label=parameter[3],
            indices_range=(0.1, 0.9),
            accuracy_range=(20, 90),
            gflops_range=(0, 20),
            # baseline_x=0.336,
            # baseline_y=80.5,
        )
        plot_multi_performance(
            baseline_file='normal.json',
            dirs_and_files_regex=sup_files_regex,
            save_path=f'./workdir/{parameter[1]}-augreg2-sup*-discard-layer.png',
            title=parameter[2],
            x_label=parameter[3],
            indices_range=(0, 0.9),
            accuracy_range=(20, 90),
            gflops_range=(0, 25),
        )
        plot_multi_performance(
            baseline_file='normal.json',
            dirs_and_files_regex=strategy_files_regex,
            save_path=f'./workdir/{parameter[1]}-augreg2-sup-discard-layer*.png',
            title=parameter[2],
            x_label=parameter[3],
            indices_range=(0, 0.9),
            accuracy_range=(20, 90),
            gflops_range=(0, 20),
        )
        plot_multi_performance(
            baseline_file='normal.json',
            dirs_and_files_regex=custom_files_regex,
            save_path=f'./workdir/{parameter[1]}-augreg2-custom.png',
            title=parameter[2],
            x_label=parameter[3],
            indices_range=(0, 0.9),
            accuracy_range=(0, 90),
            gflops_range=(0, 25),
        )

        plot_acc_flops(
            baseline_file='normal.json',
            dirs_and_files_regex=rate_files_regex,
            save_path=f'./workdir/{parameter[1]}-acc-flops-augreg2-sup-discard*-layer.png',
            title=parameter[2],
            accuracy_range=(0, 90),
            gflops_range=(0, 20),
        )
        plot_acc_flops(
            baseline_file='normal.json',
            dirs_and_files_regex=sup_files_regex,
            save_path=f'./workdir/{parameter[1]}-acc-flops-augreg2-sup*-discard-layer.png',
            title=parameter[2],
            accuracy_range=(0, 90),
            gflops_range=(0, 25),
        )
        plot_acc_flops(
            baseline_file='normal.json',
            dirs_and_files_regex=strategy_files_regex,
            save_path=f'./workdir/{parameter[1]}-acc-flops-augreg2-sup-discard-layer*.png',
            title=parameter[2],
            accuracy_range=(0, 90),
            gflops_range=(0, 20),
        )
        plot_acc_flops(
            baseline_file='normal.json',
            dirs_and_files_regex=custom_files_regex,
            save_path=f'./workdir/{parameter[1]}-acc-flops-augreg2-custom.png',
            title=parameter[2],
            accuracy_range=(0, 90),
            gflops_range=(0, 25),
        )