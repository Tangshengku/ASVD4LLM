import torch
import torch.nn as nn
from tqdm import tqdm
import os

from datautils import get_eval_loaders


class EvalLM:
    def __init__(
        self,
        model,
        tokenizer,
        # device="cuda:0",
        batch_size=1,
    ):
        super().__init__()

        # assert isinstance(device, str)
        assert isinstance(batch_size, int)

        # self._device = torch.device(device)
        self._device = model.device

        # self.model = model.to(self.device)
        self.model = model
        self.model.eval()

        self.tokenizer = tokenizer

        self.vocab_size = self.tokenizer.vocab_size

        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size

        self.seqlen = 2048

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.model.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model(inps)[0]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False)


def _load_lm_eval():
    try:
        from lm_eval.base import BaseLM
        from lm_eval import evaluator
    except ImportError:
        from lm_eval.api.model import LM as BaseLM
        from lm_eval import evaluator
    return BaseLM, evaluator


def _make_lm_eval_wrapper():
    BaseLM, _ = _load_lm_eval()
    if issubclass(EvalLM, BaseLM):
        return EvalLM
    return type("HarnessEvalLM", (EvalLM, BaseLM), {})


@torch.no_grad()
def evaluate_perplexity(model, dataset, limit, batch_size=1):
    """
    dataset: input ids tensor of shape [batch, sequence length]
    """
    nsamples, seqlen = dataset.size()
    batch_size = max(1, batch_size)
    if limit > 0:
        nsamples = min(nsamples, limit)

    total_nll = torch.zeros((), device=model.device, dtype=torch.float32)
    total_tokens = 0

    for i in range(0, nsamples, batch_size):
        batch = dataset[i : min(i + batch_size, nsamples)]
        input_ids = batch[:, :-1].to(model.device)
        labels = batch[:, 1:].contiguous().to(model.device)
        logits = model(input_ids=input_ids)[0]
        loss_fct = nn.CrossEntropyLoss(reduction="sum")
        neg_log_likelihood = loss_fct(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )
        total_nll += neg_log_likelihood.float()
        total_tokens += labels.numel()
    ppl = torch.exp(total_nll / total_tokens)
    return ppl.item()


@torch.no_grad()
def evaluate_model(
    model,
    tokenizer,
    model_name,
    tasks,
    eval_ppl="",
    num_fewshot=0,
    limit=-1,
    batch_size=1,
    use_bos=False,
    eval_ppl_fraction=1.0,
):
    """
    model: model name
    limit: number of test samples for debug, set to -1 is no limit
    tasks: str tasks are split by ,
    num_fewshot: Number of examples in few-shot context
    eval_ppl: str datasets are split by , such as 'wikitext2,ptb,c4'
    """
    lm = EvalLM(model, tokenizer, batch_size=batch_size)
    results = {}
    if eval_ppl:
        for dataset in eval_ppl.split(","):
            dataset = dataset.strip()
            if not dataset:
                continue
            cache_testloader = f"/tmp/{dataset}_testloader_{model_name.replace('/', '_')}_all.cache"
            if os.path.exists(cache_testloader):
                testloader = torch.load(cache_testloader)
                # print(f"load calibration from {cache_testloader}")
            else:
                testloader = get_eval_loaders(dataset, tokenizer)
                torch.save(testloader, cache_testloader)
            # print(dataset)
            testenc = testloader.input_ids
            eval_seqlen = lm.seqlen
            if use_bos:
                eval_seqlen -= 1
            nsamples = testenc.numel() // eval_seqlen
            if 0 < eval_ppl_fraction < 1:
                nsamples = max(1, int(nsamples * eval_ppl_fraction))
            if limit > 0:
                nsamples = min(nsamples, limit)
            use_cache = lm.model.config.use_cache
            lm.model.config.use_cache = False
            lm.model.eval()
            nlls = []

            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * eval_seqlen) : ((i + 1) * eval_seqlen)].to(lm.device)
                if use_bos:
                    bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * batch.size(dim=0)).to(lm.device)
                    batch = torch.cat([bos_tokens_tensor, batch], dim=1)
                outputs = lm.model.model(batch)
                hidden_states = outputs[0]  # .to(lm.model.lm_head.weight.device)
                if use_bos:
                    hidden_states = hidden_states[:, 1:, :]
                logits = lm.model.lm_head(hidden_states)  # .contiguous()
                shift_logits = logits[:, :-1, :]  # .contiguous()
                shift_labels = testenc[:, (i * eval_seqlen) : ((i + 1) * eval_seqlen)][:, 1:].to(lm.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * eval_seqlen
                nlls.append(neg_log_likelihood)
                # if i == 1:
                #     print(
                #         "memory_allocated",
                #         i,
                #         torch.cuda.memory_allocated() / 1024 / 1024,
                #         "max memory_allocated",
                #         torch.cuda.max_memory_allocated() / 1024**2,
                #     )

            ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * eval_seqlen))
            print(dataset, ppl.item())
            lm.model.config.use_cache = use_cache
            results[dataset] = ppl.item()
    if tasks == "longbench":
        from tools.eval_longbench import eval_longbench, full_longeval_datasets, small_longeval_datasets

        longbench_results = eval_longbench(model, tokenizer, model_name, datasets=full_longeval_datasets)
        results.update(longbench_results)
        tasks = ""
    elif tasks == "small_longbench":
        from tools.eval_longbench import eval_longbench, full_longeval_datasets, small_longeval_datasets

        longbench_results = eval_longbench(model, tokenizer, model_name, datasets=small_longeval_datasets)
        results.update(longbench_results)
        tasks = ""
    elif tasks == "mmlu":
        tasks = "hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions"
    elif tasks == "llmqat":
        # tasks = "boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa"
        tasks = "lambada_openai,openbookqa"
    if tasks != "":
        HarnessEvalLM = _make_lm_eval_wrapper()
        _, evaluator = _load_lm_eval()
        lm = HarnessEvalLM(model, tokenizer, batch_size=batch_size)
        t_results = evaluator.simple_evaluate(
            lm,
            tasks=tasks.split(","),
            batch_size=batch_size,
            num_fewshot=num_fewshot,
            limit=None if limit == -1 else limit,
            no_cache=True,
        )
        t_results = t_results["results"]
        acc_list = [t_results[key]["acc"] for key in t_results.keys() if "acc" in t_results[key]]
        t_results["mean"] = sum(acc_list) / len(acc_list)
        results.update(t_results)
        print(results)
        # print mean
        print(f"\n\n===== mean acc: {sum(acc_list)/len(acc_list)} =====\n\n")

    return results
