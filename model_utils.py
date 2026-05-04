def get_model_type(model):
    return getattr(model.config, "model_type", "").lower()


def get_model_name(model):
    return getattr(model.config, "_name_or_path", "").lower()


def get_decoder_layers(model):
    model_type = get_model_type(model)
    model_name = get_model_name(model)

    if model_type == "opt" or "opt" in model_name:
        return model.model.decoder.layers

    if model_type in {"llama", "mistral", "qwen2", "qwen3", "gemma", "gemma2"}:
        return model.model.layers

    if any(name in model_name for name in ("llama", "mistral", "qwen", "gemma")):
        return model.model.layers

    raise NotImplementedError(
        f"Unsupported model architecture for sequential layer access: "
        f"model_type={model_type!r}, name={model_name!r}"
    )
