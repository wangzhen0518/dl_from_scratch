import torch
from torch import nn
from dataclasses import dataclass
from typing import Union, List, Optional
from torch import Tensor


@dataclass
class LoraConfig:
    rank: int
    inject_module_name: Optional[Union[str, List[str]]] = None
    inject_module_type: type = nn.Linear
    dropout_p: float = 0.0


class LoraLayer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class LoraLinear(LoraLayer):
    def __init__(self, base_layer: nn.Linear, lora_config: LoraConfig, **kwargs) -> None:
        super().__init__()

        assert isinstance(base_layer, nn.Linear), f"base_layer is not nn.Linear but {type(base_layer)}"

        self.base_layer = base_layer
        self.factory_kwargs = {"device": base_layer.weight.device, "dtype": base_layer.weight.dtype}
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.lora_config = lora_config
        self.merged = False

        self.lora_A = nn.Linear(self.in_features, self.lora_config.rank, bias=False, **self.factory_kwargs)
        self.lora_B = nn.Linear(self.lora_config.rank, self.out_features, bias=False, **self.factory_kwargs)
        self.lora_dropout_layer = nn.Dropout(p=self.lora_config.dropout_p) if self.lora_config.dropout_p > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        output = self.base_layer(x)
        if not self.merged:
            lora_output = self.lora_B(self.lora_A(self.lora_dropout_layer(x)))
            output = output + lora_output
        return output


def get_parent_child(model: nn.Module, name: str):
    """返回 name 对应的 parent layer 和 parent layer 与 name 相关的属性名称"""
    cnt = name.count(".")
    if cnt > 0:
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
    else:
        parent = model
        child_name = name
    return parent, child_name


def replace_layers(model: nn.Module):
    cnt = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parent, child_name = get_parent_child(model, name)
            if not isinstance(parent, LoraLayer):
                target_layer = LoraLinear(module, LoraConfig(8, name, nn.Linear))
                setattr(parent, child_name, target_layer)
                print(f"{cnt}: {child_name} ({name})\n{parent}\n")
                cnt += 1


def statistic_parameter(model: nn.Module):
    trainable_params = 0
    freeze_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            freeze_params += param.numel()
    return {"trainable_params": trainable_params, "freeze_params": freeze_params}
