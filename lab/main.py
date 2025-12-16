# from src.transformers_qwen2_5_vl_eval import run_eval as qwen_eval, defaultargs as qwen_eval_defaultargs
# from src.transformers_qwen2_5_vl_clip_metric import run_eval as qwen_clip_eval, defaultargs as qwen_clip_eval_defaultargs
# from src.transformers_vit_eval import run_eval as tf_eval, defaultargs as tf_eval_defaultargs
# from src.timm_vit_train import run_train as timm_train, defaultargs as timm_train_defaultargs
# from src.timm_vit_eval import run_eval as timm_eval, defaultargs as timm_eval_defaultargs
# from utils import plot_multi_performance, plot_performance, plot_from_csv
from transformers.models.vit.convert_vit_timm_to_pytorch import convert_vit_checkpoint
from transformers.models.deit.convert_deit_timm_to_pytorch import convert_deit_checkpoint


import numpy as np


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


discard_regex = r"tome-0_discard-([\d.]+)\.json"
# discard_perf_vertical = [
#     ("./workdir/vit_base_patch16_224.augreg2_in21k_ft_in1k.perf", discard_regex),
#     ("./workdir/vit-graft-base-patch16-224-augreg2_in21k_ft_in1k-sup-6-discard-0.1.perf", discard_regex),
#     ("./workdir/vit-graft-base-patch16-224-augreg2_in21k_ft_in1k-sup-6-discard-0.2.perf", discard_regex),
#     ("./workdir/vit-graft-base-patch16-224-augreg2_in21k_ft_in1k-sup-6-discard-0.3.perf", discard_regex),
#     ("./workdir/vit-graft-base-patch16-224-augreg2_in21k_ft_in1k-sup-6-discard-0.4.perf", discard_regex),
#     ("./workdir/vit-graft-base-patch16-224-augreg2_in21k_ft_in1k-sup-6-discard-0.5.perf", discard_regex),
#     ("./workdir/vit-graft-base-patch16-224-augreg2_in21k_ft_in1k-sup-6-discard-0.51.perf", discard_regex),
#     ("./workdir/vit-graft-base-patch16-224-augreg2_in21k_ft_in1k-sup-12-discard-0.5.perf", discard_regex),
# ]
discard_perf_vertical = [
    ("./workdir/vit-base-patch16-224.perf", discard_regex),
    ("./workdir/vit-graft-base-patch16-224-sup-6-discard-0.3.perf", discard_regex),
    ("./workdir/vit-graft-base-patch16-224-sup-0-discard-0.3-layer-0.perf", discard_regex),
]
discard_perf_horizontal = [
    ("./workdir/vit-base-patch16-224.perf", discard_regex),
    ("./workdir/deit_base_patch16_224.fb_in1k.perf", discard_regex),
    ("./workdir/vit_base_patch16_224.augreg_in1k.perf", discard_regex),
    ("./workdir/vit_base_patch16_224.augreg_in21k_ft_in1k.perf", discard_regex),
    ("./workdir/vit_base_patch16_224.augreg2_in21k_ft_in1k.perf", discard_regex),
]


tome_regex = r"tome-(\d+)_discard-0.0\.json"
tome_perf = [
    ("./workdir/vit-base-patch16-224.perf", tome_regex),
    ("./workdir/vit-graft-base-patch16-224-sup-6-discard-0.3.perf", tome_regex),
    ("./workdir/vit-graft-base-patch16-224-sup-0-discard-0.3-layer-0.perf", tome_regex),
]


if __name__ == "__main__":
    # convert_deit_checkpoint(
    #     'deit_base_distilled_patch16_224',
    #     '/home/jovyan/nas/yrc/model/timm/deit_base_distilled_patch16_224.fb_in1k/model.safetensors',
    #     '/home/jovyan/nas/yrc/model/transformers/deit_base_distilled_patch16_224.fb_in1k/',
    # )
    convert_vit_checkpoint(
        'deit3_base_patch16_224',
        '/home/jovyan/nas/yrc/model/timm/deit3_base_patch16_224.fb_in1k/model.safetensors',
        '/home/jovyan/nas/yrc/model/transformers/deit3_base_patch16_224.fb_in1k/',
    )
    # model = ViTGraftConfig()
    # model.save_pretrained("./config.json")
    # processor = ViTImageProcessor()
    # processor.save_pretrained("./preprocessor_config.json")

    # args = tf_eval_defaultargs()
    # for i, (model_name, lib) in enumerate(model.items()):
    #     if lib == 'timm':
    #         continue
    #     args.model = model_name
    #     args.pretrained_model_path = f'/home/jovyan/nas/yrc/model/{lib}/{model_name}/'
    #     args.results_save_path = f'./workdir/{model_name}.perf'

    #     for r in range(0, 17):
    #         args.discard_rate = 0.0
    #         args.tome_r = r
    #         print(f"************** {model_name}, tome_r: {r} **************")
    #         tf_eval(args)

    #     for i in np.arange(0, 1.0, 0.1):
    #         args.tome_r = 0
    #         args.discard_rate = round(i, 1)
    #         print(f"************** {model_name}, discard_rate: {i} **************")
    #         tf_eval(args)

    # for i in range(len(model)):
    #     print(f"************** {model[i]} **************")
    #     plot_performance(perf_dir=f"./workdir/{model[i]}.perf",
    #                      save_path=f"./workdir/{model[i]}.perf/performance.png")

    # plot_multi_performance(
    #     dirs_and_files_regex=discard_perf_vertical,
    #     indice_type="discard",
    #     save_path="./workdir/discard-sup-0-6-multi-perf.png",
    #     baseline_x=0.336,
    #     baseline_y=80.5,
    # )
    # plot_multi_performance(
    #     dirs_and_files_regex=discard_perf_horizontal,
    #     indice_type="discard",
    #     save_path="./workdir/discard-horizontal-multi-perf.png",
    # )
    # plot_multi_performance(
    #     dirs_and_files_regex=tome_perf,
    #     indice_type="tome",
    #     save_path="./workdir/tome-sup-0-6-multi-perf.png",
    # )