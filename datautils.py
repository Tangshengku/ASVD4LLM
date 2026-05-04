import os
import numpy as np
import torch
from datasets import load_dataset
import random
import io
import json

"""
doc https://huggingface.co/docs/datasets/loading
doc https://huggingface.co/docs/datasets/process
doc https://huggingface.co/blog/llama2#how-to-prompt-llama-2
"""


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def sample_train_loaders(name, tokenizer, nsamples=128, seed=0, seqlen=2048):
    set_seed(seed)
    if "wikitext2" in name:
        traindata = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="train",
        )
        traindata = "\n\n".join(traindata["text"])
    elif "c4" in name:
        traindata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
        traindata = "\n\n".join(traindata["text"])
    else:
        raise NotImplementedError

    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, len(traindata) - seqlen * 2 - 1)
        j = i + seqlen * 2
        # breakpoint()
        trainenc = tokenizer(traindata[i:j], return_tensors="pt")
        inp = trainenc.input_ids[:, :seqlen]
        trainloader.append(inp)
    return trainloader


def get_redpajama_train(tokenizer, percent=10, seed=3, batch_size=128, max_length=2048):
    def tokenization(example):
        return tokenizer(example["text"], truncation=True, max_length=max_length)

    if percent != 100:
        split = f"train[:{int(850000*percent/100)}]"
    else:
        split = "train"
    dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split=split)

    processed_dataset = dataset.map(tokenization, batched=True, batch_size=batch_size, num_proc=os.cpu_count())
    return processed_dataset


def get_english_quote(dataset_name, tokenizer):
    data = load_dataset(dataset_name)
    data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
    return data["train"]


def get_qat_dataset(name, tokenizer, data_percent):
    if name == "red_pajama":
        data = get_redpajama_train(tokenizer, data_percent)

    elif name == "Abirate/english_quotes":
        data = get_english_quote(name, tokenizer)
    else:
        raise NotImplementedError
    data = data.shuffle()
    return data


llama_chat_format = """<s>[INST] <<SYS>>
"Below is an instruction that describes a task. Write a response that appropriately completes the request."
<</SYS>>

{{ instruction }} [/INST] {{ response }} </s>
"""


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


MIXTURE_CALIB_NAMES = {"mixture", "wikitext2_evol-codealpaca_tulu-math"}


def _format_messages(messages):
    turns = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if content:
            turns.append(f"{role}: {content}")
    return "\n".join(turns)


def _format_evol_codealpaca(example):
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    parts = [f"Instruction:\n{instruction}"]
    if input_text:
        parts.append(f"Input:\n{input_text}")
    if output:
        parts.append(f"Response:\n{output}")
    return "\n\n".join(parts)


def _format_tulu_math(example):
    if "messages" in example:
        return _format_messages(example["messages"])
    if "prompt" in example:
        return example["prompt"]
    return ""


def _tokenize_calib_text(tokenizer, text, seqlen, use_bos):
    if use_bos and tokenizer.bos_token is not None:
        text = tokenizer.bos_token + text
    trainenc = tokenizer(text, return_tensors="pt")
    inp = trainenc.input_ids[:, :seqlen]
    attention_mask = torch.ones_like(inp)
    return {"input_ids": inp, "attention_mask": attention_mask}


def _sample_from_text_corpus(tot_text, tokenizer, seqlen, use_bos):
    i = random.randint(0, len(tot_text) - seqlen - 1)
    j = i + seqlen * 10
    txt = tot_text[i:j]
    ind = txt.find(".")
    txt = txt[ind + 1 :].strip()
    return _tokenize_calib_text(tokenizer, txt, seqlen, use_bos)


def _sample_from_instruction_dataset(dataset, formatter, tokenizer, seqlen, use_bos):
    texts = []
    token_count = 0
    attempts = 0
    while token_count < seqlen and attempts < 64:
        example = dataset[random.randint(0, len(dataset) - 1)]
        text = formatter(example).strip()
        attempts += 1
        if not text:
            continue
        texts.append(text)
        token_count += len(tokenizer.encode(text, add_special_tokens=False))
    return _tokenize_calib_text(tokenizer, "\n\n".join(texts), seqlen, use_bos)


def get_mixture_calib_data(tokenizer, nsamples, seqlen=2048, seed=3, use_bos=False):
    random.seed(seed)
    set_seed(seed)

    wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    wikitext = "\n\n".join(wikitext["text"])
    evol_codealpaca = load_dataset("theblackcat102/evol-codealpaca-v1", split="train")
    tulu_math = load_dataset("allenai/tulu-3-sft-personas-math", split="train")

    sources = [
        lambda: _sample_from_text_corpus(wikitext, tokenizer, seqlen, use_bos),
        lambda: _sample_from_instruction_dataset(evol_codealpaca, _format_evol_codealpaca, tokenizer, seqlen, use_bos),
        lambda: _sample_from_instruction_dataset(tulu_math, _format_tulu_math, tokenizer, seqlen, use_bos),
    ]

    traindataset = []
    for i in range(nsamples):
        traindataset.append(sources[i % len(sources)]())
    random.shuffle(traindataset)
    return traindataset


def get_calib_data(name, tokenizer, model_id, nsamples, seqlen=2048, seed=3, use_bos=False):
    print(f" get_ptq_calib_data {name}, nsamples={nsamples}, seqlen={seqlen}, {seed}")
    cache_file = f"cache/{name}_{model_id.replace('/','_')}_{nsamples}_{seqlen}_{seed}_bos{use_bos}.pt"
    print(f"cache_file={cache_file}")
    if not os.path.exists("cache"):
        os.makedirs("cache")
    if os.path.exists(cache_file):
        traindataset = torch.load(cache_file)
        return traindataset
    random.seed(seed)
    set_seed(seed)
    if name in MIXTURE_CALIB_NAMES:
        traindataset = get_mixture_calib_data(tokenizer, nsamples, seqlen=seqlen, seed=seed, use_bos=use_bos)
        torch.save(traindataset, cache_file)
        return traindataset
    if name == "c4":
        traindata = load_dataset(
            "allenai/c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"}, split="train"
        )
        tot_text = "\n\n".join(traindata["text"])
    elif name == "wikitext2":
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        tot_text = "\n\n".join(traindata["text"])
    elif name == "ptb":
        traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
        tot_text = "\n\n".join(traindata["sentence"])
    elif name == "alpaca":
        # this is for chat models
        data_path = "data/alpaca_data.json"
        list_data_dict = jload(data_path)
        traindataset = []
        selected_data_dict = random.sample(list_data_dict, nsamples)
        for example in selected_data_dict:
            if example.get("input", "") == "":
                s = llama_chat_format.format(instruction=example["instruction"], response=example["output"])
                trainenc = tokenizer(s, return_tensors="pt")
                inp = trainenc.input_ids[:, :seqlen]
                attention_mask = torch.ones_like(inp)
                traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
        return traindataset
    elif name == "selfgen":
        raise NotImplementedError

    else:
        raise NotImplementedError
    print(f"tot_text={len(tot_text)}")
    traindataset = []
    for _ in range(nsamples):
        traindataset.append(_sample_from_text_corpus(tot_text, tokenizer, seqlen, use_bos))
    torch.save(traindataset, cache_file)
    return traindataset


def get_eval_loaders(name, tokenizer):
    if "wikitext2" in name:
        testdata = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test",
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    if "ptb" in name:
        valdata = load_dataset(
            "ptb_text_only",
            "penn_treebank",
            split="validation",
        )
        testenc = tokenizer("\n\n".join(valdata["sentence"]), return_tensors="pt")
        return testenc
    if "c4" in name:
        testdata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    raise NotImplementedError
