from transformers import AutoImageProcessor
from datasets import load_dataset, load_from_disk
from enum import Enum
from typing import Optional
from PIL import Image


import os
import io


class DatasetFormat(Enum):
    PARQUET = "parquet"
    ARROW = "arrow"
    TAR = "tar"
    FOLDER = "folder"
    CSV = "csv"


class ImageNetDatasetUtils:
    def __init__(
        self,
        dataset_mode: DatasetFormat=DatasetFormat.ARROW,
        dataset_path: str=None,
        batch_size: int=32,
        num_proc: int=8,
        image_processor_path: Optional[str]=None
    ):
        self.dataset_mode = dataset_mode
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_proc = num_proc
        self.image_processor_path = image_processor_path

    def load(self):
        if self.image_processor_path:
            if not os.path.exists(self.image_processor_path):
                raise FileNotFoundError(f"Image processor not found: {self.image_processor_path}")
            image_processor = AutoImageProcessor.from_pretrained(
                self.image_processor_path,
                use_fast=False
            )
            if not os.path.exists(self.dataset_path):
                raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
            # TODO: unify variety dataset loading
            datasets = load_dataset(
                "parquet",
                data_files={
                    "train": self.dataset_path+"train/train-00000-of-00001-1359597a978bc4fa.parquet",
                    "validation": self.dataset_path+"val/valid-00000-of-00001-70d52db3c749a935.parquet"
                },
                num_proc=self.num_proc,
            )

            train_len = len(datasets["train"])
            val_len = len(datasets["validation"])
            batch_size = min(self.batch_size, train_len)

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

            datasets["train"] = datasets["train"].map(
                process,
                batched=True,
                batch_size=batch_size,
                num_proc=self.num_proc,
                remove_columns=["image"]
            )
            datasets["validation"] = datasets["validation"].map(
                process,
                batched=True,
                batch_size=batch_size,
                num_proc=self.num_proc,
                remove_columns=["image"]
            )
        else:
            datasets = load_from_disk(self.dataset_path)
            train_len = len(datasets["train"])
            val_len = len(datasets["validation"])
            batch_size = min(self.batch_size, train_len)

        return datasets, train_len, val_len, batch_size