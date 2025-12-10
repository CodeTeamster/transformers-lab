from distil import GraftDistilTrainer, UnfreezeMode
from transformers import ViTGraftForImageClassification, ViTGraftConfig
from transformers import TrainingArguments, DefaultDataCollator, EarlyStoppingCallback
from transformers import logging
from utils import ImageNetDatasetUtils, DatasetFormat


import os
import json
import timm
import torch
import argparse


def run_train(args: argparse.Namespace):
    if args.log_level:
        logging.set_verbosity(args.log_level.upper())

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    imagenet = ImageNetDatasetUtils(
        dataset_mode=DatasetFormat.ARROW,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_proc=8,
        image_processor_path=args.image_processor_path
    )
    datasets, train_len, val_len, batch_size = imagenet.load()
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"[LOAD_DATASET]: train({train_len}) validation({val_len}) batch_size({batch_size})")

    for path, lib in args.teacher_model_path.items():
        if lib == "transformers":
            teacher = ViTGraftForImageClassification.from_pretrained(
                path,
                num_labels=args.num_labels,
                ignore_mismatched_sizes=True
            )
        elif lib == "timm":
            model_name = os.path.basename(os.path.dirname(path))
            teacher = timm.create_model(
                model_name,
                checkpoint_path=path
            )

    if args.student_config_path:
        # training vit from scratch
        student_config = ViTGraftConfig.from_pretrained(args.student_config_path)
        student = ViTGraftForImageClassification(student_config)
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[INITIAL_MODEL]: Create a model from the config: {args.student_config_path}")
    else:
        assert args.student_model_path, "Either args.student_config_path or args.student_model_path must be provided."
        student = ViTGraftForImageClassification.from_pretrained(
            args.student_model_path,
            num_labels=args.num_labels,
            ignore_mismatched_sizes=True
        )
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"[INITIAL_MODEL]: Transfer a model from: {args.student_model_path}")

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"[TRAIN_START]: device{[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")

    output_dir = args.ckpt_save_path
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        os.makedirs(output_dir, exist_ok=True)
        print(f"[CREATE_SAFE_PATH]: Save checkpoints in: {output_dir}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_epochs,
        fp16=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 32]),
        dataloader_pin_memory=True,
        log_level=args.log_level,
        logging_dir="./logs",
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
        save_total_limit=args.ckpt_history_amount,
        remove_unused_columns=True,
    )
    trainer = GraftDistilTrainer(
        teacher=teacher,
        student=student,
        unfreeze_mode=UnfreezeMode.ALL,
        layer_num=args.layer_num,
        train_step=23,
        discard_rate=args.discard_rate,
        discard_before_layers=args.discard_before_layers,
        temperature=args.temperature,
        alpha_param=args.alpha_param,
        beta_param=args.beta_param,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=DefaultDataCollator(),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.earlystop_patience)
        ],
    )
    trainer.train(resume_from_checkpoint=args.resume_from_ckpt)
    res = trainer.evaluate()

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"[TRAIN_END]: best model in {trainer.state.best_model_checkpoint}")
        print(res)


def defaultargs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--num-labels',
        type=int,
        default=1000,
        help='Number of labels for classification'
    )
    parser.add_argument(
        '--teacher-model-path',
        type=json.loads,
        default='{"/home/jovyan/nas/yrc/model/google/vit-base-patch16-224/": "transformers"}',
        help='Path to the teachers model in Json format'
    )
    parser.add_argument(
        '--student-config-path',
        type=str,
        default='',
        help='Path to the student config, if this argument is given, args.student_model_path will be ignored'
    )
    parser.add_argument(
        '--student-model-path',
        type=str,
        default='/home/jovyan/nas/yrc/model/yeruichen/vit-graft-base-patch16-224/',
        help='Path to the student model, either this or args.student_config_path must be selected'
    )
    parser.add_argument(
        '--resume-from-ckpt',
        action='store_true',
        help='Whether to resume from checkpoint'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='/home/jovyan/nas/yrc/dataset/imagenet-1k/arrow/',
        help='Path to the dataset'
    )
    parser.add_argument(
        '--image-processor-path',
        type=str,
        default='/home/jovyan/nas/yrc/model/google/vit-base-patch16-224/',
        help='Path to the image processor, if this argument is null, the preprocessing will be skipped'
    )
    parser.add_argument(
        '--ckpt-save-path',
        type=str,
        default='/home/jovyan/nas/yrc/workspace/hugging-face-experiments/experiments/checkpoints/vit-base-patch16-224-mt2-distill',
        help='Save checkpoints to file'
    )
    parser.add_argument('--ckpt-history-amount', type=int, default=10)
    parser.add_argument(
        '--log-level',
        type=str,
        default='info',
        help='Logging level, ["debug" | "info" | "warning" | "error" | "critical"]'
    )
    parser.add_argument('--layer-num', type=int, default=12)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--warmup-ratio', type=float, default=0.1)
    parser.add_argument('--num-epochs', type=int, default=300)
    parser.add_argument('--earlystop-patience', type=int, default=10)
    parser.add_argument('--discard-rate', type=float, default=0.3)
    parser.add_argument('--discard-before-layers', type=int, nargs='+', default=[])
    parser.add_argument('--temperature', type=float, default=5.)
    parser.add_argument('--alpha-param', type=float, default=0.3)
    parser.add_argument('--beta-param', type=float, default=0.7)

    return parser.parse_args()


if __name__ == "__main__":
    run_train(defaultargs())