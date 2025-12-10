from transformers import Trainer
from accelerate.test_utils.testing import get_backend


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import evaluate
import numpy as np
import transformers


def compute_metrics(eval_pred):
    accuracy = evaluate.load("./src/evaluate-metric/accuracy")
    predictions, labels = eval_pred
    acc = accuracy.compute(references=labels, predictions=np.argmax(predictions, axis=1))
    return {"accuracy": acc["accuracy"]}


class MultiTeacherDistilTrainer(Trainer):
    def __init__(
        self,
        teachers: list=None,
        student=None,
        temperature: float=None,
        alpha_param: float=None,
        beta_param: float=None,
        layer_param: list=None,
        tome_r: int=0,
        compute_metrics=compute_metrics,
        *args,
        **kwargs
    ):
        super().__init__(model=student, compute_metrics=compute_metrics, *args, **kwargs)
        self.teachers = teachers
        self.tome_r = tome_r
        self.loss_function = nn.KLDivLoss(reduction="batchmean")
        self.teacher_select_loss_function = nn.CrossEntropyLoss(reduction="mean")
        device, _, _ = get_backend() # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
        for teacher in self.teachers:
            teacher.to(device)
            teacher.eval()
        self.temperature = temperature
        assert alpha_param + beta_param < 1.0, "alpha_param + beta_param should be less than 1.0."
        self.alpha_param = alpha_param
        self.beta_param = beta_param
        if layer_param is not None:
            assert abs(sum(layer_param) - 1.0) < 1e-6, "The sum of all elements in layer_param must be 1."
        self.layer_param = layer_param

    def _select_teacher(self, inputs):
        os.environ["TOME_R"] = str(self.tome_r)

        loss = float('inf')
        # teacher 需要提供 ToMe 下每层 merge_fn，origin 下每层输出和最终的 logits
        for teacher in self.teachers:
            if isinstance(teacher, transformers.PreTrainedModel):
                teacher_output = teacher(**inputs, output_hidden_states=True)
                teacher_logits = teacher_output.logits
                teacher_hidden_states = teacher_output.hidden_states
                teacher_loss = teacher_output.loss
                teacher_merge_fn = teacher.vit.merge_fn
            elif isinstance(teacher, torch.nn.Module):
                teacher_logits, teacher_hidden_states = teacher(inputs['pixel_values'], output_hidden_states=True)
                teacher_loss = self.teacher_select_loss_function(teacher_logits, inputs['labels'])
                teacher_merge_fn = teacher.merge_fn

            if teacher_loss < loss:
                loss = teacher_loss
                best_teacher = teacher
                merge_fn = teacher_merge_fn
                logits = teacher_logits
                hidden_states = teacher_hidden_states

        os.environ.pop("TOME_R", None)

        return best_teacher, merge_fn, logits, hidden_states

    def compute_loss(self, student, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            student_output = student(**inputs)
            return (student_output.loss, student_output)

        with torch.no_grad():
            teacher, merge_fn, teacher_logits, teacher_hidden_states = self._select_teacher(inputs)

        student_output = student(**inputs, merge_fn=merge_fn, output_hidden_states=True, output_align_hidden_states=False)
        student_target_loss = student_output.loss
        student_hidden_states = student_output.hidden_states

        # Compute soft targets for teacher and student
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_output.logits / self.temperature, dim=-1)

        # Compute the cls token loss
        cls_loss = self.loss_function(soft_student, soft_teacher) * (self.temperature ** 2)
        # Compute the hidden states loss
        hidden_loss = 0.
        if self.layer_param is None:
            for s_hid, t_hid in zip(student_hidden_states, teacher_hidden_states):
                hidden_loss += F.mse_loss(s_hid, t_hid)
            hidden_loss = hidden_loss / len(student_hidden_states)  # optional, take the average
        else:
            for i, (s_hid, t_hid) in enumerate(zip(student_hidden_states, teacher_hidden_states)):
                hidden_loss += F.mse_loss(s_hid, t_hid) * self.layer_param[i]

        # Calculate final loss
        loss = (1. - self.beta_param - self.alpha_param) * student_target_loss + \
            self.alpha_param * cls_loss + self.beta_param * hidden_loss

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