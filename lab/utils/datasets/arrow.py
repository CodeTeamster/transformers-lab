from datasets import load_dataset, load_from_disk
from typing import Callable
from transformers import ViTImageProcessor
from PIL import Image


import io


def create_arrow_datasets(
    parquet_train_path: str=None,
    parquet_validation_path: str=None,
    image_folder_path: str=None,
    arrow_save_path: str=None,
    process: Callable=None,
    batch_size: int=1,
    num_proc: int=8
):
    if parquet_train_path is not None and parquet_validation_path is not None:
        print(f"1. Loading parquet dataset from: {parquet_train_path} and {parquet_validation_path}")
        datasets = load_dataset(
            "parquet",
            data_files={
                "train": parquet_train_path,
                "validation": parquet_validation_path
            },
            num_proc=num_proc,
        )
    elif image_folder_path is not None:
        print(f"1. Loading image folder dataset from: {image_folder_path}")
        datasets = load_dataset(
            "imagefolder",
            data_dir=image_folder_path,
            num_proc=num_proc,
        )

    print(f"2. Processing dataset: batch_size = {batch_size}, num_proc = {num_proc}")
    datasets = datasets.map(
        process,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=["image"]
    )
    print(f"3. Saving dataset to disk: {arrow_save_path}")
    datasets.save_to_disk(arrow_save_path)


if __name__ == "__main__":
    image_processor = ViTImageProcessor.from_pretrained(
        '/home/jovyan/nas/yrc/model/google/vit-base-patch16-224/',
        use_fast=False
    )
    def process(examples):
        if isinstance(examples["image"][0], dict):
            images = []
            for img in examples["image"]:
                img_opj = Image.open(io.BytesIO(img['bytes'])).convert("RGB")
                images.append(img_opj)
        else:
            images = examples["image"]

        processed_inputs = image_processor(
            images=images,
            return_tensors="pt",
        )
        return processed_inputs

    create_arrow_datasets(
        # parquet_train_path="/home/jovyan/nas/yrc/dataset/tiny-imagenet/train/train-00000-of-00001-1359597a978bc4fa.parquet",
        # parquet_validation_path="/home/jovyan/nas/yrc/dataset/tiny-imagenet/val/valid-00000-of-00001-70d52db3c749a935.parquet",
        image_folder_path="/home/jovyan/nas/yrc/dataset/imagenet-1k",
        arrow_save_path="/home/jovyan/nas/yrc/dataset/imagenet-1k/arrow-batch256-nw16/",
        process=process,
        batch_size=256,
        num_proc=16
    )
    dataset = load_from_disk("/home/jovyan/nas/yrc/dataset/imagenet-1k/arrow-batch256-nw16/")
    print(dataset['train'][0].keys())
