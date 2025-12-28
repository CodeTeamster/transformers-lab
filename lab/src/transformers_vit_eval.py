from transformers.models import ViTRDForImageClassification, ViTImageProcessor
from torchvision import datasets, transforms
from tqdm import tqdm
from thop import profile, clever_format


import os
import sys
import time
import torch
import argparse
import json
import numpy as np


class ModelWrapper(torch.nn.Module):
    def __init__(self, model, discard_rate):
        super().__init__()
        self.model = model
        self.discard_rate = discard_rate

    def forward(self, pixel_values):
        discard_before_layers = [False] * self.model.config.num_hidden_layers
        discard_before_layers[0] = True
        return self.model(
            pixel_values=pixel_values,
            discard_rate=self.discard_rate,
            discard_before_layers=discard_before_layers,
            seed=42,
        )


@torch.no_grad()
def inference(
    model: torch.nn.Module,
    image_processor,
    data_loader: torch.utils.data.DataLoader,
    device: str,
    discard_rate: float = 0.0,
):
    model.eval()

    accu_num = torch.zeros(1).to(device)
    sample_num = 0

    tqdm_loader = tqdm(data_loader, file=sys.stdout)
    for images, labels in tqdm_loader:
        images = image_processor(images, return_tensors="pt")
        images = {k: v.to(device) for k, v in images.items()}
        labels = labels.to(device)

        discard_before_layers = [False] * model.config.num_hidden_layers
        discard_before_layers[0] = True
        preds = model(
            **images,
            discard_rate=discard_rate,
            discard_before_layers=discard_before_layers,
            seed=42,
        ).logits.argmax(dim=-1)

        accu_num += (preds == labels).sum()
        sample_num += labels.size(0)

        tqdm_loader.set_postfix(acc=f"{accu_num.item() / sample_num:.3f}")

    return accu_num.item() / sample_num


def run_eval(args: argparse.Namespace):
    if args.tome_r > 0:
        file_name = f"tome-{args.tome_r}.json"
    elif args.discard_rate > 0:
        file_name = f"discard-{args.discard_rate:.2f}.json"
    elif args.divprune > 0:
        file_name = f"divprune-{args.divprune:.2f}.json"
    elif args.evit > 0:
        file_name = f"evit-{args.evit:.2f}.json"
    else:
        file_name = f"normal.json"
    if os.path.exists(os.path.join(args.results_save_path, file_name)):
        print(f"Results file already exists: {os.path.join(args.results_save_path, file_name)}. Skipping...")
        return

    os.environ.pop("DIVPRUNE", None)
    os.environ.pop("EViT", None)
    os.environ.pop("TOME_R", None)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.divprune:
        os.environ["DIVPRUNE"] = str(1 - args.divprune)
    if args.evit:
        os.environ["EViT"] = str(1 - args.evit)
    if args.tome_r:
        os.environ["TOME_R"] = str(args.tome_r)

    # generate dataloader
    data_transform = transforms.Compose([
        transforms.Resize(args.vit_input_size),
        transforms.CenterCrop(args.vit_input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    test_dataset = datasets.ImageNet(root=args.dataset_path, split='val', transform=data_transform)
    total_images = len(test_dataset)

    batch_size = min(args.batch_size, total_images)
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 32])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        collate_fn=None,
    )

    # create model
    model = ViTRDForImageClassification.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_path,
        use_safetensors=args.use_safetensors,
    ).to(device)
    image_processor = ViTImageProcessor.from_pretrained(
        pretrained_model_name_or_path=args.image_processor_path,
    )

    # warm-up GPU
    dummy_input = torch.randn(1, 3, args.vit_input_size, args.vit_input_size).to(device)
    # calculate FLOPs and Params
    dummy_model = ModelWrapper(model, args.discard_rate)
    flops, params = profile(dummy_model, inputs=(dummy_input,), verbose=False)
    flops, params = clever_format([flops, params], "%.2f")
    with torch.no_grad():
        for _ in range(9):
            _ = dummy_model(dummy_input)
    torch.cuda.synchronize()

    with torch.no_grad():
        start_time = time.time()
        test_acc = inference(
            model=model,
            image_processor=image_processor,
            data_loader=data_loader,
            device=device,
            discard_rate=args.discard_rate,
        )
    torch.cuda.synchronize()
    os.environ.pop("DIVPRUNE", None)
    os.environ.pop("EViT", None)
    os.environ.pop("TOME_R", None)

    inference_time = time.time() - start_time
    throughput = total_images / inference_time if inference_time > 0 else 0

    if args.tome_r > 0:
        print(f"\n--- ToMe r={args.tome_r} Metrics ---")
    elif args.discard_rate > 0:
        print(f"\n--- Discard Rate={args.discard_rate} Metrics ---")
    elif args.divprune > 0:
        print(f"\n--- DivPrune Rate={args.divprune} Metrics ---")
    elif args.evit > 0:
        print(f"\n--- EViT Rate={args.evit} Metrics ---")
    else:
        print(f"\n--- Normal Metrics ---")
    print(f"FLOPs: {flops}, Params: {params}")
    print(f"Total images processed: {total_images}")
    print(f"Accuracy: {test_acc*100:.2f}%")
    print(f"Total inference time: {inference_time:.4f} seconds")
    print(f"Throughput: {throughput:.2f} im/s")

    if args.results_save_path:
        if not os.path.exists(args.results_save_path):
            os.mkdir(args.results_save_path)

        results = {
            "device": str(device),
            "flops": flops,
            "params": params,
            "total_images": total_images,
            "accuracy": test_acc,
            "total_inference_time": inference_time,
            "throughput": throughput
        }

        if args.tome_r > 0:
            file_name = f"tome-{args.tome_r}.json"
        elif args.discard_rate > 0:
            file_name = f"discard-{args.discard_rate:.2f}.json"
        elif args.divprune > 0:
            file_name = f"divprune-{args.divprune:.2f}.json"
        elif args.evit > 0:
            file_name = f"evit-{args.evit:.2f}.json"
        else:
            file_name = f"normal.json"
        with open(os.path.join(args.results_save_path, file_name), "w") as f:
            json.dump(results, f, indent=4)

def defaultargs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to use for inference',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--pretrained-model-path',
        type=str,
        default='/starry-nas/yrc/model/yeruichen/vit-rd-base-patch16-224.augreg2_in21k_ft_in1k.sup-layerwise1/',
        help='Path to the pretrained model',
    )
    parser.add_argument(
        '--image-processor-path',
        type=str,
        default='/home/jovyan/nas/yrc/model/transformers/vit_base_patch16_224.augreg2_in21k_ft_in1k/',
        help='Path to the image processor',
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='/home/jovyan/nas/yrc/dataset/imagenet-1k/',
        help='Path to the dataset',
    )
    parser.add_argument(
        '--vit-input-size',
        type=int,
        default=224,
        help='Input size of the ViT, [224 | 384]',
    )
    parser.add_argument(
        '--results-save-path',
        type=str,
        default='./workdir/vit-rd-base-patch16-224.augreg2_in21k_ft_in1k.sup-layerwise1-discard-0.5.perf',
        help='Save results to JSON file',
    )
    parser.add_argument(
        '--discard-rate',
        type=float,
        default=0.,
        help='Token discarding rate',
    )
    parser.add_argument(
        '--divprune',
        type=float,
        default=0.,
        help='Token pruning rate by divprune method',
    )
    parser.add_argument(
        '--evit',
        type=float,
        default=0.,
        help='Token pruning rate by EViT method',
    )
    parser.add_argument(
        '--tome-r',
        type=int,
        default=0,
        help='Token merging\'s hyper-parameter',
    )
    parser.add_argument(
        '--use-safetensors',
        default=True,
        action='store_true',
        help='Whether to use safetensors',
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = defaultargs()

    model_path_prefix = './ckpts/'
    save_path_prefix = './workdir/'
    model_paths = [
        'augreg2.sup-layerwise1-discard-0.4-layer-0/checkpoint-460460/',
        'augreg2.sup-layerwise2-discard-0.4-layer-0/checkpoint-480480/',
        'augreg2.sup-0-discard-0.6-layer-4/checkpoint-420420/'
    ]
    for model_path in model_paths:
        args.pretrained_model_path = model_path_prefix + model_path
        args.results_save_path = save_path_prefix + model_path.split('/', 1)[0] + '.perf'

        print(f'model: {args.pretrained_model_path}')
        print(f'save: {args.results_save_path}')
        for i in np.arange(0, 1.0, 0.1):
            args.discard_rate = round(i, 2)
            args.divprune = 0
            args.evit = 0
            args.tome_r = 0
            run_eval(args)

        for i in np.arange(0.1, 1.0, 0.1):
            args.discard_rate = 0
            args.divprune = round(i, 2)
            args.evit = 0
            args.tome_r = 0
            run_eval(args)

        for i in np.arange(0.1, 1.0, 0.1):
            args.discard_rate = 0
            args.divprune = 0
            args.evit = round(i, 2)
            args.tome_r = 0
            run_eval(args)

        for i in range(1, 17):
            args.discard_rate = 0
            args.divprune = 0
            args.evit = 0
            args.tome_r = i
            run_eval(args)

