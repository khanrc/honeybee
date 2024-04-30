import torch
from transformers import TrainerCallback


class AbstractorTypeConvertCallback(TrainerCallback):
    "A callback that coverts type of abstractor module"

    def on_step_begin(self, args, state, control, **kwargs):
        # conver model to fp32
        if kwargs["model"].proj_type == "d-abs":
            # Cast the type of ddetr abstractor to float32 because ddetr does not support fp16 or bf16.
            kwargs["model"].abstractor.to(torch.float)

    def on_prediction_step(self, args, state, control, **kwargs):
        # conver model to fp32
        if kwargs["model"].proj_type == "d-abs":
            # Cast the type of ddetr abstractor to float32 because ddetr does not support fp16 or bf16.
            kwargs["model"].abstractor.to(torch.float)
