from transformers import Qwen2_5_VLModel, Qwen2VLImageProcessor, Qwen2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.functional import normalize, cosine_similarity
from accelerate import Accelerator


import argparse
import os
import sys
import torch
import evaluate


def run_eval(args: argparse.Namespace):
    model = Qwen2_5_VLModel.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_path,
        torch_dtype="auto",
    )
    processor = Qwen2VLImageProcessor.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_path,
        use_fast=False,
    )
    tokenizer = Qwen2Tokenizer.from_pretrained(pretrained_model_name_or_path=args.pretrained_model_path)

    val_datasets = load_dataset(args.val_dataset_path, split="test")
    total_samples = len(val_datasets)
    batch_size = min(args.batch_size, total_samples)
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 32])
    data_loader = DataLoader(
        val_datasets,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        collate_fn=lambda batch: ([item["image"] for item in batch], [item["caption"][0] for item in batch]),
    )

    # Distributed inference
    accelerator = Accelerator()
    model, data_loader = accelerator.prepare(model, data_loader)

    model.eval()
    tqdm_loader = tqdm(data_loader, file=sys.stdout, disable=not accelerator.is_local_main_process)
    metric = evaluate.load("./src/evaluate-metric/normal/mean.py")
    metric_value = 0
    with torch.no_grad():
        for images, captions in tqdm_loader:
            texts = tokenizer(captions, return_tensors="pt", padding=True)
            texts = {k: v.to(accelerator.device) for k, v in texts.items()}
            texts_embeds = model.module.get_input_embeddings()(texts['input_ids'])
            text_features = texts_embeds.mean(dim=1)

            images = processor(images, return_tensors="pt")
            images = {k: v.to(accelerator.device) for k, v in images.items()}
            image_embeds = model.module.get_image_features(images['pixel_values'], images['image_grid_thw'])
            image_features = torch.stack([img.mean(dim=0) for img in image_embeds], dim=0)  # [batch_size, text_features]

            image_features = normalize(image_features, dim=-1)
            text_features = normalize(text_features, dim=-1)
            similarity = cosine_similarity(text_features, image_features, dim=-1)

            similarity = accelerator.gather_for_metrics(similarity)
            if accelerator.is_local_main_process:
                metric.add_batch(value=similarity)
                metric_value = metric.compute()
                tqdm_loader.set_postfix(similarity=f"{metric_value['mean']:.3f}")

    return metric_value


def defaultargs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pretrained-model-path',
        type=str,
        default="/home/jovyan/nas/yrc/model/Qwen/Qwen2.5-VL-7B-Instruct/",
        help='Path to the pretrained model',
    )
    parser.add_argument(
        '--val-dataset-path',
        type=str,
        default="lmms-lab/flickr30k",
        help='Local path or hf\'s name to the image for evaluation',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=2,
        help='Batch size for inference',
    )

    return parser.parse_args()


if __name__ == "__main__":
    metric = run_eval(defaultargs())
    print(metric)