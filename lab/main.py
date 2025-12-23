# from src.transformers_qwen2_5_vl_eval import run_eval as qwen_eval, defaultargs as qwen_eval_defaultargs
# from src.transformers_qwen2_5_vl_clip_metric import run_eval as qwen_clip_eval, defaultargs as qwen_clip_eval_defaultargs
# from src.transformers_vit_eval import run_eval as tf_eval, defaultargs as tf_eval_defaultargs
# from src.timm_vit_train import run_train as timm_train, defaultargs as timm_train_defaultargs
# from src.timm_vit_eval import run_eval as timm_eval, defaultargs as timm_eval_defaultargs
# from transformers import ViTImageProcessor, ViTRDForImageClassification, ViTRDConfig
from utils import plot_multi_performance


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
    ("./workdir/augreg2-sup-layerwise1-discard-0.4-layer-0.perf",),
    ("./workdir/augreg2-sup-layerwise2-discard-0.4-layer-0.perf",),
]
sup_comparison = [
    ("./workdir/augreg2-sup-layerwise1-discard-0.4-layer-0.perf",),
    ("./workdir/augreg2-sup-layerwise2-discard-0.4-layer-0.perf",),
]
strategy_comparison = [
    ("./workdir/augreg2-sup-layerwise1-discard-0.4-layer-0.perf",),
    ("./workdir/augreg2-sup-layerwise2-discard-0.4-layer-0.perf",),
]


if __name__ == "__main__":
    # model = ViTRDConfig()
    # model.save_pretrained("./config.json")
    # processor = ViTImageProcessor()
    # processor.save_pretrained("./preprocessor_config.json")

    rate_comparison_discard = [item + (discard_regex,) for item in rate_comparison]
    sup_comparison_discard = [item + (discard_regex,) for item in sup_comparison]
    strategy_comparison_discard = [item + (discard_regex,) for item in strategy_comparison]
    plot_multi_performance(
        baseline_file='normal.json',
        dirs_and_files_regex=rate_comparison_discard,
        save_path='./workdir/discard-augreg2-sup-discard*-layer.png',
        title='Random Discard Performance',
        x_label='Discard Rate',
        indices_range=(0.1, 0.9),
        accuracy_range=(20, 90),
        gflops_range=(0, 20),
    )
    plot_multi_performance(
        baseline_file='normal.json',
        dirs_and_files_regex=sup_comparison_discard,
        save_path='./workdir/discard-augreg2-sup*-discard-layer.png',
        title='Random Discard Performance',
        x_label='Discard Rate',
        indices_range=(0, 0.9),
        accuracy_range=(20, 90),
        gflops_range=(0, 25),
    )
    plot_multi_performance(
        baseline_file='normal.json',
        dirs_and_files_regex=strategy_comparison_discard,
        save_path='./workdir/discard-augreg2-sup-discard-layer*.png',
        title='Random Discard Performance',
        x_label='Discard Rate',
        indices_range=(0, 0.9),
        accuracy_range=(20, 90),
        gflops_range=(0, 20),
    )

    rate_comparison_divprune = [item + (divprune_regex,) for item in rate_comparison]
    sup_comparison_divprune = [item + (divprune_regex,) for item in sup_comparison]
    strategy_comparison_divprune = [item + (divprune_regex,) for item in strategy_comparison]
    plot_multi_performance(
        baseline_file='normal.json',
        dirs_and_files_regex=rate_comparison_divprune,
        save_path='./workdir/divprune-augreg2-sup-discard*-layer.png',
        title='Diversity Pruning Performance',
        x_label='DivPrune Rate',
        indices_range=(0, 0.9),
        accuracy_range=(20, 90),
        gflops_range=(0, 20),
    )
    plot_multi_performance(
        baseline_file='normal.json',
        dirs_and_files_regex=sup_comparison_divprune,
        save_path='./workdir/divprune-augreg2-sup*-discard-layer.png',
        title='Diversity Pruning Performance',
        x_label='DivPrune Rate',
        indices_range=(0, 0.9),
        accuracy_range=(20, 90),
        gflops_range=(0, 25),
    )
    plot_multi_performance(
        baseline_file='normal.json',
        dirs_and_files_regex=strategy_comparison_divprune,
        save_path='./workdir/divprune-augreg2-sup-discard-layer*.png',
        title='Diversity Pruning Performance',
        x_label='DivPrune Rate',
        indices_range=(0, 0.9),
        accuracy_range=(20, 90),
        gflops_range=(0, 20),
    )

    rate_comparison_evit = [item + (evit_regex,) for item in rate_comparison]
    sup_comparison_evit = [item + (evit_regex,) for item in sup_comparison]
    strategy_comparison_evit = [item + (evit_regex,) for item in strategy_comparison]
    plot_multi_performance(
        baseline_file='normal.json',
        dirs_and_files_regex=rate_comparison_evit,
        save_path='./workdir/evit-augreg2-sup-discard*-layer.png',
        title='EViT Pruning Performance',
        x_label='EViT Rate',
        indices_range=(0, 0.9),
        accuracy_range=(0, 90),
        gflops_range=(0, 20),
    )
    plot_multi_performance(
        baseline_file='normal.json',
        dirs_and_files_regex=sup_comparison_evit,
        save_path='./workdir/evit-augreg2-sup*-discard-layer.png',
        title='EViT Pruning Performance',
        x_label='EViT Rate',
        indices_range=(0, 0.9),
        accuracy_range=(0, 90),
        gflops_range=(0, 25),
    )
    plot_multi_performance(
        baseline_file='normal.json',
        dirs_and_files_regex=strategy_comparison_evit,
        save_path='./workdir/evit-augreg2-sup-discard-layer*.png',
        title='EViT Pruning Performance',
        x_label='EViT Rate',
        indices_range=(0, 0.9),
        accuracy_range=(0, 90),
        gflops_range=(0, 20),
    )