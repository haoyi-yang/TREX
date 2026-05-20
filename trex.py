#!/usr/bin/env python3

import argparse
import random
import sys
import time
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def find_layers(module, layers=[nn.Linear], name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child,
                layers=layers,
                name=name + "." + name1 if name != "" else name1,
            )
        )
    return res


class NormAccumulator:
    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]
        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0
        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples


class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    trainenc = tokenizer(
        " ".join(traindata["text"]), return_tensors="pt", truncation=False, add_special_tokens=False
    )
    testenc = tokenizer(
        "\n\n".join(testdata["text"]), return_tensors="pt", truncation=False, add_special_tokens=False
    )
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    valdata = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    valenc = tokenizer(
        " ".join(valdata[:1100]["text"]), return_tensors="pt", truncation=False, add_special_tokens=False
    )
    valenc = valenc.input_ids[:, : (256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if tokenizer is None:
        raise ValueError("tokenizer is required for get_loaders")
    if "wikitext2" in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    raise ValueError(f"Unknown dataset name for get_loaders: {name}")


@torch.no_grad()
def prepare_calibration_input(model, dataloader, device, nsamples=128):
    use_cache_exists = hasattr(model.config, "use_cache")
    if use_cache_exists:
        old_use_cache = model.config.use_cache
        model.config.use_cache = False
    if hasattr(model, "hf_device_map") and "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]
    layers = model.model.layers
    seqlen = model.seqlen
    hidden = model.config.hidden_size
    dtype = next(iter(model.parameters())).dtype
    inps = torch.empty((nsamples, seqlen, hidden), dtype=dtype, device=device)
    outs = torch.empty_like(inps)
    filled = 0
    attention_mask_ref = [None]
    position_ids_ref = [None]

    def pre_hook(module, inputs):
        nonlocal filled
        x = inputs[0]
        bsz = x.size(0)
        take = min(bsz, nsamples - filled)
        inps[filled : filled + take].copy_(x[:take])
        filled += take
        return None

    handle = layers[0].register_forward_pre_hook(pre_hook)
    for batch in dataloader:
        if filled >= nsamples:
            break
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device) if len(batch) > 1 and batch[1] is not None else None
        position_ids = batch[2].to(device) if len(batch) > 2 and batch[2] is not None else None
        if attention_mask is not None and attention_mask.dtype == torch.long:
            attention_mask = attention_mask.bool()
        if attention_mask_ref[0] is None:
            attention_mask_ref[0] = attention_mask
        if position_ids_ref[0] is None:
            position_ids_ref[0] = position_ids
        _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
    handle.remove()
    inps = inps[:filled]
    outs = outs[:filled].zero_()
    attn = attention_mask_ref[0]
    pos = position_ids_ref[0]
    position_embeddings = None
    if hasattr(model.model, "rotary_emb") and pos is not None:
        position_embeddings = model.model.rotary_emb(inps[:1], position_ids=pos)
    if use_cache_exists:
        model.config.use_cache = old_use_cache
    return inps, outs, attn, pos, position_embeddings


def load_model_and_tokenizer(model_name, device, trust_remote_code=False, dtype=torch.float32):
    print(f"Loading model: {model_name}")
    if isinstance(dtype, str):
        if dtype == "auto":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                model_dtype = torch.bfloat16
            else:
                model_dtype = torch.float32
        elif dtype == "float32":
            model_dtype = torch.float32
        elif dtype == "float16":
            model_dtype = torch.float16
        elif dtype == "bfloat16":
            model_dtype = torch.bfloat16
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
    else:
        model_dtype = dtype
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=model_dtype,
        trust_remote_code=trust_remote_code,
    )
    model = model.to(device)
    print(f"Model loaded on {device}")
    return model, tokenizer, config


def get_batch(data, batch_size, block_size, device):
    if len(data) < block_size + 1:
        raise ValueError(f"Dataset too short: {len(data)} < {block_size + 1}")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


def evaluate_model(model, tokenizer, token_ids, batch_size, block_size, device, eval_iters=None):
    vocab_size = tokenizer.vocab_size
    if token_ids.max().item() >= vocab_size:
        valid_mask = token_ids < vocab_size
        token_ids = token_ids[valid_mask]
    model.eval()
    if eval_iters is None:
        max_iters = (len(token_ids) - block_size) // (batch_size * block_size)
        eval_iters = max(1, max_iters)
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i in tqdm(range(eval_iters), desc="Evaluation", file=sys.stdout):
            try:
                x, y = get_batch(token_ids, batch_size, block_size, device)
                outputs = model(input_ids=x)
                logits = outputs.logits
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = y[:, :-1].contiguous()
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-1,
                )
                batch_tokens = shift_labels.numel()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens
                del x, y, outputs, logits, shift_logits, shift_labels, loss
            except Exception as e:
                print(f"Warning: Error in batch {i}: {e}")
                if device == "cuda":
                    torch.cuda.empty_cache()
                continue
    if total_tokens == 0:
        raise ValueError("No tokens were evaluated")
    if device == "cuda":
        torch.cuda.empty_cache()
    val_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(val_loss)).item()
    print(f"Validation loss: {val_loss:.4f}, perplexity: {perplexity:.4f}")
    return val_loss, perplexity


def _plan_for_one_row(
    r: int,
    perm: List[int],
    n_p: int,
    n_select: int,
    row_blocks: Optional[torch.Tensor] = None,
) -> Tuple[int, List[Tuple[int, int]], float]:
    t0 = time.perf_counter()
    taken_n: set = set()
    used_as_source: set = set()
    plan: List[Tuple[int, int]] = []
    nc = n_p - 1
    for idx in perm:
        if n_select > 0 and len(plan) >= n_select:
            break
        if len(taken_n) + len(used_as_source) >= nc:
            break
        n, j = idx // n_p, idx % n_p
        if n == 0 or n in taken_n:
            continue
        if n == j:
            continue
        if j in taken_n or n in used_as_source:
            continue
        taken_n.add(n)
        if j != 0:
            used_as_source.add(j)
        plan.append((n, j))
        if row_blocks is not None:
            bn = n - 1
            if j == 0:
                row_blocks[bn, :, :].zero_()
            else:
                bj = j - 1
                row_blocks[bn, :, :].copy_(row_blocks[bj, :, :])
    t_loop = time.perf_counter() - t0
    return (r, plan, t_loop)


def build_merge_prune_plan(
    score: torch.Tensor,
    nr: int,
    nc: int,
    merge_prune_ratio: float,
    W: Optional[torch.Tensor] = None,
    block_size: Optional[int] = None,
    merge_fully: bool = False,
    row_group_idx: Optional[int] = None,
    col_group_idx: Optional[int] = None,
    log_file: Optional[Any] = None,
    w_batch_group: Optional[torch.Tensor] = None,
) -> List[List[Tuple[int, int]]]:
    n_p = nc + 1
    if merge_fully:
        n_select = -1
    else:
        n_select = max(0, int(nc * merge_prune_ratio))
    if not merge_fully and n_select == 0:
        return [[] for _ in range(nr)]
    score = score.float()
    for r in range(nr):
        score[r].diagonal().fill_(float("inf"))
    score_flat = score.view(nr, -1)
    perm_all = torch.argsort(score_flat, dim=1, stable=True).cpu().tolist()
    num_workers = min(nr, 64)
    row_plans: List[Optional[List[Tuple[int, int]]]] = [None] * nr
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for r in range(nr):
            row_blocks = w_batch_group[r] if w_batch_group is not None else None
            futures[executor.submit(_plan_for_one_row, r, perm_all[r], n_p, n_select, row_blocks)] = r
        for fut in as_completed(futures):
            r, plan, _ = fut.result()
            row_plans[r] = plan
    return list(row_plans)


def apply_merge_prune_plan(
    W: torch.Tensor,
    row_plans: List[List[Tuple[int, int]]],
    block_size: int,
) -> None:
    nr = W.shape[0] // block_size
    bs = block_size
    device = W.device
    W_cpu = W.cpu().clone()
    results = []
    for r in range(nr):
        ro = r * bs
        row = W_cpu[ro : ro + bs, :].clone()
        for n, j in row_plans[r]:
            co_n = (n - 1) * bs
            if j == 0:
                row[:, co_n : co_n + bs].zero_()
            else:
                co_j = (j - 1) * bs
                row[:, co_n : co_n + bs].copy_(row[:, co_j : co_j + bs])
        results.append(row)
    out = torch.cat(results, dim=0)
    W.copy_(out.to(device))
    if device.type == "cuda":
        torch.cuda.synchronize()


def block_merge(args, model, tokenizer, device):
    use_cache_exists = hasattr(model.config, "use_cache")
    if use_cache_exists:
        old_use_cache = model.config.use_cache
        model.config.use_cache = False
    block_size = getattr(args, "block_size", 64)
    cols_group_size = getattr(args, "cols_group_size", 1024)
    has_device_map = hasattr(model, "hf_device_map") and model.hf_device_map is not None
    layers = model.model.layers
    num_layers = len(layers)
    all_plans: List[Dict[str, Any]] = []
    timings = {"data_load": 0.0, "prepare_calibration": 0.0, "forward": 0.0, "score_and_plan": 0.0}
    merge_log_path = getattr(args, "merge_log_path", "merge.log")
    Path(merge_log_path).parent.mkdir(parents=True, exist_ok=True)

    def tee(msg: str, file=None) -> None:
        print(msg)
        if file is not None:
            file.write(msg + "\n")
            file.flush()

    log_file = open(merge_log_path, "w", encoding="utf-8")
    tee("Loading data...", log_file)
    t0 = time.perf_counter()
    trainloader, _ = get_loaders(
        "c4", nsamples=args.num_samples, seed=args.seed, seqlen=args.seq_len, tokenizer=tokenizer
    )
    timings["data_load"] = time.perf_counter() - t0
    tee("Data loaded.", log_file)
    dataloader_for_cal = []
    for inp, _ in trainloader:
        mask = torch.ones_like(inp, dtype=torch.bool, device=inp.device)
        pos = torch.arange(inp.size(1), device=inp.device, dtype=torch.long).unsqueeze(0).expand(inp.size(0), -1)
        dataloader_for_cal.append((inp, mask, pos))
    t0 = time.perf_counter()
    inps, outs, attn, pos, position_embeddings = prepare_calibration_input(
        model, dataloader_for_cal, device, nsamples=args.num_samples
    )
    timings["prepare_calibration"] = time.perf_counter() - t0
    target_layer = getattr(args, "target_layer", None)
    target_layers = getattr(args, "target_layers", None)
    if target_layers is not None and not isinstance(target_layers, (list, tuple, set)):
        target_layers = None
    if target_layers is not None and len(target_layers) == 0:
        target_layers = None
    if target_layer is not None and target_layers is not None:
        raise ValueError("use either target_layer or target_layers, not both")
    target_name = getattr(args, "target_name", None)
    if target_layer is not None or target_layers is not None or target_name is not None:
        if target_layers is not None:
            tee(f"Target only: layers={sorted(target_layers)} name={target_name}", log_file)
        else:
            tee(f"Target only: layer={target_layer} name={target_name}", log_file)
    tee(
        f"\nMerge block: {num_layers} layers, block_size={block_size}, cols_group_size={cols_group_size}",
        log_file,
    )
    tee(f"Activation collection: {args.num_samples} samples, seqlen {inps.shape[1]}", log_file)

    for i, layer in enumerate(tqdm(layers, desc="Merge block")):
        if has_device_map and f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps = inps.to(dev)
            outs = outs.to(dev)
            attn = attn.to(dev)
            pos = pos.to(dev)
            if position_embeddings is not None:
                if isinstance(position_embeddings, tuple):
                    position_embeddings = tuple(t.to(dev) for t in position_embeddings)
                else:
                    position_embeddings = position_embeddings.to(dev)
        linear_layers = find_layers(layer)
        norm_accumulators = {}
        for name in linear_layers:
            norm_accumulators[name] = NormAccumulator(
                linear_layers[name], layer_id=i, layer_name=name
            )

        def make_add_batch(name):
            def hook(_, inp, out):
                norm_accumulators[name].add_batch(inp[0].data, out.data)

            return hook

        handles = [
            linear_layers[name].register_forward_hook(make_add_batch(name))
            for name in linear_layers
        ]
        t0 = time.perf_counter()
        for j in range(args.num_samples):
            with torch.no_grad():
                out = layer(
                    inps[j : j + 1],
                    attention_mask=attn,
                    position_ids=pos,
                    position_embeddings=position_embeddings,
                )[0]
                outs[j] = out[0]
        if device == "cuda":
            torch.cuda.synchronize()
        timings["forward"] += time.perf_counter() - t0
        for handle in handles:
            handle.remove()

        if target_layers is not None and i not in target_layers:
            inps, outs = outs, inps
            continue
        if target_layer is not None and i != target_layer:
            inps, outs = outs, inps
            continue

        apply_plan = getattr(args, "apply_plan", False)
        t0 = time.perf_counter()
        for name_idx, name in enumerate(linear_layers):
            if target_name is not None and name != target_name:
                continue
            linear = linear_layers[name]
            W = linear.weight.data
            out_f, in_f = W.shape[0], W.shape[1]
            if out_f % block_size != 0 or in_f % block_size != 0:
                continue
            nr, nc = out_f // block_size, in_f // block_size
            acc = norm_accumulators[name]
            scaler = acc.scaler_row.to(W.device).float().clamp(min=1e-8)
            s_blocks_flat = scaler.view(nc, block_size)
            norm_s = torch.sqrt(s_blocks_flat).norm(p=2, dim=1)
            del s_blocks_flat
            w_blocks = W.float().view(nr, block_size, nc, block_size).permute(0, 2, 1, 3)
            batch_size_rows = 512
            zero_block = torch.zeros(
                1,
                block_size,
                block_size,
                device=W.device,
                dtype=W.dtype,
            )
            merge_prune_ratio = getattr(args, "merge_prune_ratio", 0.2)
            row_plans: List[List[Tuple[int, int]]] = [[] for _ in range(nr)]
            num_batches = (nr + batch_size_rows - 1) // batch_size_rows
            merge_fully = getattr(args, "merge_fully", False)
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size_rows
                end_idx = min(start_idx + batch_size_rows, nr)
                batch_rows = end_idx - start_idx
                if batch_rows <= 0:
                    continue
                w_batch = w_blocks[start_idx:end_idx]
                zero_batch = zero_block.unsqueeze(0).expand(batch_rows, -1, -1, -1)
                num_col_groups = (nc + cols_group_size - 1) // cols_group_size
                for col_group_idx in range(num_col_groups):
                    col_start = col_group_idx * cols_group_size
                    col_end = min(col_start + cols_group_size, nc)
                    curr_nc = col_end - col_start
                    if curr_nc <= 0:
                        continue
                    w_batch_group = w_batch[:, col_start:col_end]
                    w_batch_with_zero = torch.cat([zero_batch, w_batch_group], dim=1)
                    w_flat_group = w_batch_with_zero.view(batch_rows, curr_nc + 1, -1)
                    norm_s_group = norm_s[col_start:col_end]
                    norm_s_ext_group = torch.cat(
                        [
                            torch.ones(1, device=W.device, dtype=W.dtype),
                            norm_s_group,
                        ]
                    )
                    dist_batch_group = torch.cdist(w_flat_group, w_flat_group, p=2)
                    score_batch_group = (dist_batch_group * norm_s_ext_group.unsqueeze(0).unsqueeze(2)).cpu()
                    del w_batch_with_zero, w_flat_group, dist_batch_group, norm_s_group, norm_s_ext_group
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    row_plans_batch_group = build_merge_prune_plan(
                        score_batch_group,
                        batch_rows,
                        curr_nc,
                        merge_prune_ratio,
                        W=None,
                        block_size=None,
                        merge_fully=merge_fully,
                        row_group_idx=batch_idx,
                        col_group_idx=col_group_idx,
                        log_file=log_file,
                        w_batch_group=w_batch_group if apply_plan else None,
                    )
                    del score_batch_group, w_batch_group
                    for local_r, plan_local in enumerate(row_plans_batch_group):
                        global_r = start_idx + local_r
                        if global_r >= nr:
                            continue
                        if not plan_local:
                            continue
                        mapped_plan: List[Tuple[int, int]] = []
                        for n_local, j_local in plan_local:
                            n_global = col_start + n_local
                            if j_local == 0:
                                j_global = 0
                            else:
                                j_global = col_start + j_local
                            mapped_plan.append((n_global, j_global))
                        row_plans[global_r].extend(mapped_plan)
                del w_batch, zero_batch
                if device == "cuda":
                    torch.cuda.empty_cache()
            if apply_plan:
                W.copy_(
                    w_blocks.permute(0, 2, 1, 3).reshape(out_f, in_f).to(W.dtype)
                )
            del w_blocks, zero_block
            all_plans.append(
                {
                    "layer": i,
                    "name_idx": name_idx,
                    "row_plans": row_plans,
                    "nr": nr,
                    "nc": nc,
                    "block_size": block_size,
                }
            )

        if device == "cuda":
            torch.cuda.synchronize()
        timings["score_and_plan"] += time.perf_counter() - t0

        if apply_plan:
            with torch.no_grad():
                for j in range(args.num_samples):
                    out = layer(
                        inps[j : j + 1],
                        attention_mask=attn,
                        position_ids=pos,
                        position_embeddings=position_embeddings,
                    )[0]
                    outs[j] = out[0]
        inps, outs = outs, inps

    if use_cache_exists:
        model.config.use_cache = old_use_cache
    total = timings["data_load"] + timings["prepare_calibration"] + timings["forward"] + timings["score_and_plan"]
    tee(
        f"block_merge total: {total:.2f}s block_size={block_size} cols_group_size={cols_group_size} (log: {merge_log_path})",
        log_file,
    )
    log_file.close()
    return all_plans


def _encode_plans_to_tensors(all_plans: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    e = len(all_plans)
    layers = torch.zeros(e, dtype=torch.long)
    name_idx = torch.zeros(e, dtype=torch.long)
    nr = torch.zeros(e, dtype=torch.long)
    nc = torch.zeros(e, dtype=torch.long)
    block_size = torch.zeros(e, dtype=torch.long)
    pairs_list: List[List[int]] = []
    for entry_idx, entry in enumerate(all_plans):
        layers[entry_idx] = entry["layer"]
        name_idx[entry_idx] = entry["name_idx"]
        nr[entry_idx] = entry["nr"]
        nc[entry_idx] = entry["nc"]
        block_size[entry_idx] = entry["block_size"]
        for row, row_plan in enumerate(entry["row_plans"]):
            for n, j in row_plan:
                pairs_list.append([entry_idx, row, n, j])
    if pairs_list:
        pairs = torch.tensor(pairs_list, dtype=torch.long)
    else:
        pairs = torch.zeros(0, 4, dtype=torch.long)
    return {
        "layers": layers,
        "name_idx": name_idx,
        "nr": nr,
        "nc": nc,
        "block_size": block_size,
        "pairs": pairs,
    }


def save_merge_plans(path: str, all_plans: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(_encode_plans_to_tensors(all_plans), path)


def _parse_layer_indices(tokens: list[str]) -> list[int]:
    out: list[int] = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a_s, b_s = tok.split("-", 1)
            a, b = int(a_s), int(b_s)
            if b < a:
                raise ValueError(f"invalid layer range token '{tok}'")
            out.extend(list(range(a, b + 1)))
        else:
            out.append(int(tok))
    seen = set()
    uniq = []
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        uniq.append(x)
    return uniq


def parse_args():
    p = argparse.ArgumentParser(description="Run block merge with configurable args")
    p.add_argument("--model", type=str, required=True, help="Model name or path")
    p.add_argument("--device", type=str, default=None, help="Device (default: cuda if available)")
    p.add_argument("--seq-len", type=int, default=2048, help="Sequence length for calibration and eval")
    p.add_argument("--num-samples", type=int, default=128, help="Number of calibration samples")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--block-size", type=int, default=2, help="Block size for merge/prune")
    p.add_argument("--cols-group-size", type=int, default=1024, help="Column group size for block_merge")
    p.add_argument(
        "--merge-prune-ratio",
        type=float,
        default=0.6,
        help="Merge/prune ratio (fraction of blocks to merge or prune)",
    )
    p.add_argument("--apply-plan", action="store_true", help="Apply plan during planning (in-place)")
    p.add_argument("--merge-fully", action="store_true", help="Merge until no more valid pairs (ignore ratio)")
    p.add_argument("--target-layer", type=int, default=None, help="Only process this layer index (default: all)")
    p.add_argument(
        "--target-layers",
        type=int,
        nargs="+",
        default=None,
        help="Only process these layer indices (space-separated). Mutually exclusive with --target-layer.",
    )
    p.add_argument(
        "--layer-range",
        type=str,
        default="",
        help='Inclusive layer range like "10-17". Equivalent to --target-layers 10 11 ... 17.',
    )
    p.add_argument("--target-name", type=str, default=None, help="Only process this weight name (e.g. mlp.down_proj)")
    p.add_argument("--merge-log-path", type=str, default="merge.log", help="Path to merge log file")
    p.add_argument(
        "--save-plans-path",
        type=str,
        default=None,
        help="Path to save plans tensor (.pt) (default: do not save)",
    )
    p.add_argument("--eval-iters", type=int, default=20, help="Evaluation iterations")
    p.add_argument("--eval-batch-size", type=int, default=4, help="Evaluation batch size")
    p.add_argument(
        "--eval-dataset",
        type=str,
        choices=["wikitext2", "c4"],
        default="wikitext2",
        help="Corpus for before/after PPL eval (default: wikitext2).",
    )
    p.add_argument("--no-eval", action="store_true", help="Skip eval before/after (only run block_merge)")
    return p.parse_args()


def main():
    cli = parse_args()
    device = cli.device or ("cuda" if torch.cuda.is_available() else "cpu")
    seqlen = cli.seq_len

    if cli.target_layer is not None and (cli.target_layers is not None or cli.layer_range.strip()):
        raise SystemExit("use either --target-layer OR (--target-layers / --layer-range), not both")
    if cli.target_layers is not None and cli.layer_range.strip():
        raise SystemExit("use either --target-layers OR --layer-range, not both")

    target_layers = None
    if cli.target_layers is not None:
        target_layers = sorted(set(cli.target_layers))
    elif cli.layer_range.strip():
        target_layers = _parse_layer_indices([cli.layer_range])

    args = Namespace(
        num_samples=cli.num_samples,
        seed=cli.seed,
        seq_len=seqlen,
        sparsity_ratio=0.2,
        block_size=cli.block_size,
        cols_group_size=cli.cols_group_size,
        merge_prune_ratio=cli.merge_prune_ratio,
        apply_plan=cli.apply_plan,
        merge_fully=cli.merge_fully,
        target_layer=cli.target_layer,
        target_layers=target_layers,
        target_name=cli.target_name,
        merge_log_path=cli.merge_log_path,
        save_plans_path=cli.save_plans_path,
    )

    print("Loading model and tokenizer...")
    t0 = time.perf_counter()
    model, tokenizer, _ = load_model_and_tokenizer(
        model_name=cli.model,
        device=device,
        trust_remote_code=True,
        dtype="float16" if device == "cuda" else "float32",
    )
    if not hasattr(model, "seqlen"):
        model.seqlen = seqlen
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"  done in {time.perf_counter() - t0:.2f}s")

    num_layers = len(model.model.layers)
    if args.target_layers is not None:
        bad = [i for i in args.target_layers if i < 0 or i >= num_layers]
        if bad:
            raise SystemExit(f"invalid layer indices {bad} for model with {num_layers} layers")

    token_ids = None
    loss_before = ppl_before = loss_after = ppl_after = None

    if not cli.no_eval:
        print(f"\n--- Eval (before, dataset={cli.eval_dataset}) ---")
        _, valenc = get_loaders(
            cli.eval_dataset,
            nsamples=args.num_samples,
            seed=args.seed,
            seqlen=seqlen,
            tokenizer=tokenizer,
        )
        token_ids = valenc.input_ids.squeeze(0).to(device)
        loss_before, ppl_before = evaluate_model(
            model,
            tokenizer,
            token_ids,
            batch_size=cli.eval_batch_size,
            block_size=seqlen,
            device=device,
            eval_iters=cli.eval_iters,
        )
        print(f"  val_loss={loss_before:.4f}, ppl={ppl_before:.4f}")

    print(
        f"\n--- block_merge (prune_ratio={args.merge_prune_ratio}, block_size={args.block_size}, apply_plan={args.apply_plan}) ---"
    )
    t0 = time.perf_counter()
    all_plans = block_merge(args, model, tokenizer, device)
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"  block_merge total (main): {time.perf_counter() - t0:.2f}s")
    print(f"  plans: {len(all_plans)} layer/weight entries")

    if not cli.no_eval:
        print(f"\n--- Eval (after, dataset={cli.eval_dataset}) ---")
        if token_ids is None:
            _, valenc = get_loaders(
                cli.eval_dataset,
                nsamples=args.num_samples,
                seed=args.seed,
                seqlen=seqlen,
                tokenizer=tokenizer,
            )
            token_ids = valenc.input_ids.squeeze(0).to(device)
        loss_after, ppl_after = evaluate_model(
            model,
            tokenizer,
            token_ids,
            batch_size=cli.eval_batch_size,
            block_size=seqlen,
            device=device,
            eval_iters=cli.eval_iters,
        )
        print(f"  val_loss={loss_after:.4f}, ppl={ppl_after:.4f}")
        print("\n--- Comparison ---")
        print(f"  val_loss: {loss_before:.4f} -> {loss_after:.4f}")
        print(f"  ppl:      {ppl_before:.4f} -> {ppl_after:.4f}")

    if args.merge_log_path:
        with open(args.merge_log_path, "a", encoding="utf-8") as f:
            f.write(f"\n--- Evaluation (dataset={cli.eval_dataset}) ---\n")
            if loss_before is not None:
                f.write(f"  eval (before): val_loss={loss_before:.4f} ppl={ppl_before:.4f}\n")
            if loss_after is not None:
                f.write(f"  eval (after):  val_loss={loss_after:.4f} ppl={ppl_after:.4f}\n")
            if loss_before is not None and loss_after is not None:
                f.write(
                    f"  comparison:    val_loss {loss_before:.4f} -> {loss_after:.4f}  ppl {ppl_before:.4f} -> {ppl_after:.4f}\n"
                )
            f.flush()

    if args.save_plans_path:
        save_merge_plans(args.save_plans_path, all_plans)
        print(f"\n  plans saved to {args.save_plans_path}")


if __name__ == "__main__":
    main()
