from transformers import Trainer, PreTrainedModel
from accelerate.test_utils.testing import get_backend
from typing import List, Union, Optional
from enum import Flag, auto


import torch
import torch.nn as nn
import torch.nn.functional as F
import evaluate
import numpy as np


class UnfreezeMode(Flag):
    HEAD = auto()
    SUP_POSITION = auto()
    VIT_LAYER = auto()
    TAIL = auto()
    ALL = HEAD | SUP_POSITION | VIT_LAYER | TAIL
    NONE = 0


def compute_metrics(eval_pred):
    accuracy = evaluate.load("./src/evaluate-metric/accuracy")
    predictions, labels = eval_pred
    acc = accuracy.compute(references=labels, predictions=np.argmax(predictions, axis=1))
    return {"accuracy": acc["accuracy"]}


class ViTRDDistilTrainer(Trainer):
    def __init__(
        self,
        teacher: Union[PreTrainedModel, nn.Module]=None,
        student: PreTrainedModel=None,
        unfreeze_mode: UnfreezeMode=UnfreezeMode.ALL,
        layer_num: int=None,
        discard_rate: float=None,
        discard_before_layers: Optional[List[int]]=None,
        temperature: float=None,
        alpha_param: float=None,
        beta_param: float=None,
        compute_metrics=compute_metrics,
        *args,
        **kwargs
    ):
        # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
        device, _, _ = get_backend()
        self._student = student.to(device)
        self._teacher = teacher.to(device)
        self._teacher.eval()
        self._layer_num = layer_num
        self._discard_rate = discard_rate
        self._discard_before_layers = [False] * self._layer_num
        for layer_id in discard_before_layers or []:
            self._discard_before_layers[layer_id] = True
        self._temperature = temperature
        self._alpha_param = alpha_param
        self._beta_param = beta_param
        self._distill_loss_function = nn.KLDivLoss(reduction="batchmean")

        self._graft_and_freeze(
            teacher=self._teacher,
            student=self._student,
            unfreeze_mode=unfreeze_mode,
            vit_layers=[i for i in range(12)],
            train_mode=True
        )
        super().__init__(
            model=self._student,
            compute_metrics=compute_metrics,
            *args,
            **kwargs,
        )

    def _is_same_parameter(self, model_a_elem: PreTrainedModel, model_b_elem: PreTrainedModel):
        for param_a, param_b in zip(model_a_elem.parameters(), model_b_elem.parameters()):
            if not torch.equal(param_a, param_b):
                return False
        return True

    def _graft_and_freeze(
        self,
        teacher: Union[PreTrainedModel, nn.Module],
        student: PreTrainedModel,
        unfreeze_mode: UnfreezeMode,
        vit_layers: List[int] = None,
        train_mode: bool = True,
    ):
        assert vit_layers is not None, "Please provide the vit_layers to be grafted."
        with torch.no_grad():
            teacher_dict = {name: param for name, param in teacher.named_parameters()}
            vit_layers_str = [
                f"vit.encoder.layer.{layer_id}." for layer_id in vit_layers
            ]
            for name, param in student.named_parameters():
                # 1. Head
                if (
                    name == 'vit.embeddings.cls_token'
                    or name == 'vit.embeddings.patch_embeddings.projection.weight'
                    or name == 'vit.embeddings.patch_embeddings.projection.bias'
                ):
                    param.data.copy_(teacher_dict[name].data.clone())
                    param.requires_grad = False
                    if bool(unfreeze_mode & UnfreezeMode.HEAD):
                        param.requires_grad = True
                    continue

                # 2. Supplement and Position
                if (
                    name == 'vit.embeddings.sup_token'
                    or name == 'vit.embeddings.position_embeddings'
                ):
                    param.requires_grad = False
                    if bool(unfreeze_mode & UnfreezeMode.SUP_POSITION):
                        param.requires_grad = True
                    continue

                # 3. Tail
                if (
                    name == 'vit.layernorm.weight'
                    or name == 'vit.layernorm.bias'
                    or name == 'classifier.weight'
                    or name == 'classifier.bias'
                ):
                    param.data.copy_(teacher_dict[name].data.clone())
                    param.requires_grad = False
                    if bool(unfreeze_mode & UnfreezeMode.TAIL):
                        param.requires_grad = True
                    continue

                # 4. Vit Layers
                if any(nd in name for nd in vit_layers_str):
                    param.data.copy_(teacher_dict[name].data.clone())
                    param.requires_grad = False
                    if bool(unfreeze_mode & UnfreezeMode.VIT_LAYER):
                        param.requires_grad = True
        student.eval()
        if train_mode:
            student.train()

        return

    def compute_loss(self, student_model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            student_output = student_model(
                **inputs,
                discard_rate=self._discard_rate,
                discard_before_layers=self._discard_before_layers,
                seed=int(self._discard_rate * 100),
            )
            return (student_output.loss, student_output)

        with torch.no_grad():
            if isinstance(self._teacher, PreTrainedModel):
                teacher_output = self._teacher(**inputs)
                teacher_logits = teacher_output.logits
                teacher_soft = F.softmax(teacher_logits / self._temperature, dim=-1)

        student_output = student_model(
            **inputs,
            discard_rate=self._discard_rate,
            discard_before_layers=self._discard_before_layers
        )
        student_logits = student_output.logits
        student_loss = student_output.loss
        student_soft = F.log_softmax(student_logits / self._temperature, dim=-1)

        # Compute the distillation loss
        distill_loss = self._distill_loss_function(student_soft, teacher_soft) * (self._temperature ** 2)
        # Compute the final loss
        loss = self._alpha_param * student_loss + self._beta_param * distill_loss

        return loss


if __name__ == "__main__":
    predictions = np.array([
        [2.0, 0.5, 0.3],
        [0.1, 1.5, 2.2],
        [1.2, 2.1, 0.3],
        [2.2, 0.1, 3.0],
        [2.5, 0.2, 0.1]
    ])
    labels = np.array([0, 2, 1, 2, 0])

    eval_pred = (predictions, labels)
    print(compute_metrics(eval_pred))