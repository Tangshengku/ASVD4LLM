import os

import torch
import torch.nn as nn

from modules.svd_linear import SVDLinear


ASVD_MODELING_CODE = r'''import torch
import torch.nn as nn

try:
    from transformers import MistralForCausalLM
except ImportError:
    try:
        from transformers.models.mistral.modeling_mistral import MistralForCausalLM
    except ImportError:
        MistralForCausalLM = None

try:
    from transformers import Qwen3ForCausalLM
except ImportError:
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    except ImportError:
        try:
            from transformers import Qwen2ForCausalLM as Qwen3ForCausalLM
        except ImportError:
            try:
                from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM as Qwen3ForCausalLM
            except ImportError:
                Qwen3ForCausalLM = None


class SVDLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()
        self.BLinear = nn.Linear(in_features, rank, bias=False)
        self.ALinear = nn.Linear(rank, out_features, bias=bias)
        self.truncation_rank = rank

    def forward(self, inp):
        return self.ALinear(self.BLinear(inp))


def _set_module(root, name, module):
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def _replace_asvd_linears(model, config):
    for name, info in getattr(config, "asvd_linear_info", {}).items():
        _set_module(
            model,
            name,
            SVDLinear(
                info["in_features"],
                info["out_features"],
                info["rank"],
                bias=info["bias"],
            ),
        )


if MistralForCausalLM is not None:
    class ASVDMistralForCausalLM(MistralForCausalLM):
        def __init__(self, config):
            super().__init__(config)
            _replace_asvd_linears(self, config)
else:
    class ASVDMistralForCausalLM(nn.Module):
        def __init__(self, config):
            raise ImportError("MistralForCausalLM is unavailable in this transformers installation.")


if Qwen3ForCausalLM is not None:
    class ASVDQwen3ForCausalLM(Qwen3ForCausalLM):
        def __init__(self, config):
            super().__init__(config)
            _replace_asvd_linears(self, config)
else:
    class ASVDQwen3ForCausalLM(nn.Module):
        def __init__(self, config):
            raise ImportError("Qwen3ForCausalLM/Qwen2ForCausalLM is unavailable in this transformers installation.")
'''


def _asvd_auto_class_name(model):
    model_type = getattr(model.config, "model_type", "").lower()
    model_name = getattr(model.config, "_name_or_path", "").lower()
    if model_type == "mistral" or "mistral" in model_name:
        return "ASVDMistralForCausalLM"
    if model_type in {"qwen3", "qwen2"} or "qwen" in model_name:
        return "ASVDQwen3ForCausalLM"
    raise NotImplementedError(
        f"HF-style ASVD export currently supports Mistral and Qwen. Got model_type={model_type!r}."
    )


def _collect_asvd_linear_info(model):
    info = {}
    for name, module in model.named_modules():
        if isinstance(module, SVDLinear):
            info[name] = {
                "in_features": module.BLinear.weight.size(1),
                "out_features": module.ALinear.weight.size(0),
                "rank": module.truncation_rank,
                "bias": module.ALinear.bias is not None,
            }
    return info


def _set_module(root, name, module):
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def _svd_linear_to_dense(module):
    dense_weight = module.ALinear.weight @ module.BLinear.weight
    dense = nn.Linear(
        dense_weight.size(1),
        dense_weight.size(0),
        bias=module.ALinear.bias is not None,
    )
    dense = dense.to(device=dense_weight.device, dtype=dense_weight.dtype)
    with torch.no_grad():
        dense.weight.copy_(dense_weight)
        if module.ALinear.bias is not None:
            dense.bias.copy_(module.ALinear.bias)
    return dense


def densify_svd_linears(model):
    svd_names = [name for name, module in model.named_modules() if isinstance(module, SVDLinear)]
    for name in svd_names:
        module = model.get_submodule(name)
        _set_module(model, name, _svd_linear_to_dense(module))
    return len(svd_names)


def _save_dense_hf(model, tokenizer, output_dir):
    n_densified = densify_svd_linears(model)

    for attr in ("asvd_linear_info", "auto_map"):
        if hasattr(model.config, attr):
            delattr(model.config, attr)

    tokenizer.save_pretrained(output_dir)
    try:
        model.save_pretrained(output_dir, safe_serialization=True)
    except TypeError:
        model.save_pretrained(output_dir)

    print(f"Saved dense HF model to {output_dir} (densified {n_densified} SVDLinear modules)")


def _save_asvd_custom_hf(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    asvd_linear_info = _collect_asvd_linear_info(model)
    if not asvd_linear_info:
        print("WARNING: no SVDLinear modules found; saving a standard HF model.")

    class_name = _asvd_auto_class_name(model)
    model.config.asvd_linear_info = asvd_linear_info
    model.config.architectures = [class_name]
    model.config.auto_map = {"AutoModelForCausalLM": f"modeling_asvd.{class_name}"}

    with open(os.path.join(output_dir, "modeling_asvd.py"), "w") as f:
        f.write(ASVD_MODELING_CODE)

    tokenizer.save_pretrained(output_dir)
    try:
        model.save_pretrained(output_dir, safe_serialization=True)
    except TypeError:
        model.save_pretrained(output_dir)

    print(f"Saved HF-style ASVD model to {output_dir}")


def save_asvd_hf(model, tokenizer, output_dir, dense=True):
    os.makedirs(output_dir, exist_ok=True)
    if dense:
        _save_dense_hf(model, tokenizer, output_dir)
    else:
        _save_asvd_custom_hf(model, tokenizer, output_dir)
