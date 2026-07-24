"""
Train the v2 Recurrent Refiner: a frozen 8B-class base model (4-bit) plus a
small trainable recurrent reasoning block with ACT-style adaptive halting.

Designed to be run in short bursts across whatever free GPU platform is
available (Kaggle, Lightning AI Studio, Colab, ...) - pass --hub_repo_id to
checkpoint through a private HuggingFace Hub repo so any platform can resume
where the last one left off.

python -m recurrent_refiner.train --base_model Qwen/Qwen2.5-Coder-7B-Instruct
"""
import argparse
import math
import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from .config import RefinerConfig
from .model import CodeRecurrentModel


# Small, diverse Python corpus that always works offline (no dataset download,
# no network dependency). Only used as a fallback when --dataset isn't given
# or fails to load.
INLINE_CODE_SAMPLES = [
    "def hello():\n    print('hello world')\n\ndef main():\n    hello()",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    p = arr[0]\n    l = [x for x in arr[1:] if x <= p]\n    r = [x for x in arr[1:] if x > p]\n    return quicksort(l) + [p] + quicksort(r)",
    "def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]",
    "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n-1)",
    "def binary_search(arr, x):\n    l, r = 0, len(arr)-1\n    while l <= r:\n        m = (l+r)//2\n        if arr[m] == x: return m\n        if arr[m] < x: l = m+1\n        else: r = m-1\n    return -1",
    "class Stack:\n    def __init__(self):\n        self.items = []\n    def push(self, x):\n        self.items.append(x)\n    def pop(self):\n        return self.items.pop()\n    def is_empty(self):\n        return len(self.items) == 0",
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    m = len(arr)//2\n    l = merge_sort(arr[:m])\n    r = merge_sort(arr[m:])\n    return merge(l, r)\n\ndef merge(l, r):\n    res = []\n    i = j = 0\n    while i < len(l) and j < len(r):\n        if l[i] < r[j]:\n            res.append(l[i]); i += 1\n        else:\n            res.append(r[j]); j += 1\n    return res + l[i:] + r[j:]",
]


def load_code_samples(dataset_name):
    """Load training text. If `dataset_name` is given, try the HF `datasets`
    library and fall back to the built-in inline corpus on any failure
    (missing package, no internet, dataset renamed/removed, unknown schema)
    so training never hard-fails just because of dataset flakiness."""
    if not dataset_name:
        return list(INLINE_CODE_SAMPLES)

    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split="train")
        texts = []
        for ex in ds:
            for key in ("code", "content", "text", "func_code_string"):
                if isinstance(ex, dict) and ex.get(key):
                    texts.append(ex[key])
                    break
        if not texts:
            raise ValueError("no known text/code field (tried code/content/text/func_code_string)")
        print(f"Loaded {len(texts)} examples from dataset '{dataset_name}'")
        return texts
    except Exception as e:
        print(
            f"[WARN] failed to load dataset '{dataset_name}' ({e}); "
            f"falling back to the {len(INLINE_CODE_SAMPLES)} built-in samples"
        )
        return list(INLINE_CODE_SAMPLES)


class CodeCompletionCollator:
    def __init__(self, tokenizer, max_len: int = 2048):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        texts = [ex["content"] if isinstance(ex, dict) else ex for ex in batch]
        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return enc.input_ids, enc.attention_mask


class CodeDataset:
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx % len(self.samples)]


def train_step(model, input_ids, attention_mask, optimizer, scheduler,
               moe_aux_weight: float, act_loop_penalty: float):
    model.refiner.train()
    if not model._use_base_head:
        model.lm_head.train()

    logits, moe_aux_loss, act_ponder_loss = model(input_ids, attention_mask=attention_mask)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()
    labels = shift_labels.masked_fill(shift_mask == 0, -100)

    ce_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )
    loss = ce_loss + moe_aux_weight * moe_aux_loss + act_loop_penalty * act_ponder_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        (p for p in model.parameters() if p.requires_grad), 1.0
    )
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    n_tokens = shift_mask.sum().item()
    return ce_loss.item(), moe_aux_loss.item(), act_ponder_loss.item(), n_tokens


@torch.no_grad()
def evaluate(model, val_loader):
    if val_loader is None:
        return None
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for input_ids, attention_mask in val_loader:
        input_ids = input_ids.to(next(model.base.parameters()).device)
        attention_mask = attention_mask.to(input_ids.device)
        logits, _, _ = model(input_ids, attention_mask=attention_mask)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()
        labels = shift_labels.masked_fill(shift_mask == 0, -100)

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += shift_mask.sum().item()

    model.refiner.train()
    if not model._use_base_head:
        model.lm_head.train()
    return total_loss / max(total_tokens, 1)


def pull_checkpoint_from_hub(hub_repo_id, local_dir):
    """Download the most recent checkpoint from a private HF Hub repo, if any
    exists yet. Returns the local path or None."""
    if not hub_repo_id:
        return None
    try:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        files = [f for f in api.list_repo_files(hub_repo_id) if f.endswith(".pt")]
        if not files:
            print(f"No checkpoints found yet in hub repo '{hub_repo_id}' - starting fresh")
            return None
        # best.pt if present, else the highest step_N.pt
        import re
        def step_of(f):
            m = re.search(r"step_(\d+)\.pt$", f)
            return int(m.group(1)) if m else -1
        best = [f for f in files if os.path.basename(f) == "best.pt"]
        target = best[0] if best else max(files, key=step_of)
        path = hf_hub_download(hub_repo_id, target, local_dir=local_dir)
        print(f"Resumed checkpoint '{target}' from hub repo '{hub_repo_id}'")
        return path
    except Exception as e:
        print(f"[WARN] could not pull checkpoint from hub repo '{hub_repo_id}' ({e}); starting fresh")
        return None


def push_checkpoint_to_hub(hub_repo_id, local_path):
    if not hub_repo_id:
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(hub_repo_id, private=True, exist_ok=True)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=os.path.basename(local_path),
            repo_id=hub_repo_id,
        )
        print(f"Pushed {os.path.basename(local_path)} to hub repo '{hub_repo_id}'")
    except Exception as e:
        print(f"[WARN] could not push checkpoint to hub repo '{hub_repo_id}' ({e})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--no_4bit", action="store_true",
                         help="Disable 4-bit quantization (needs a lot more VRAM).")
    parser.add_argument("--dataset", default="google-research-datasets/mbpp",
                         help="HF dataset name for training text (374 examples). Falls "
                              "back to the built-in offline samples if it can't be loaded.")
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--moe_aux_weight", type=float, default=0.01)
    parser.add_argument("--act_loop_penalty", type=float, default=0.01)
    parser.add_argument("--max_loop_iters", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--output_dir", default="./refiner_checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                         help="Local checkpoint path to resume from (overrides --hub_repo_id pull).")
    parser.add_argument("--hub_repo_id", type=str, default=None,
                         help="Private HF Hub repo (e.g. 'username/recurrent-refiner-ckpts') "
                              "used as a checkpoint bus across training platforms.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = args.max_seq_len

    print("Loading model...")
    cfg = RefinerConfig(
        base_model_id=args.base_model,
        load_in_4bit=not args.no_4bit,
        max_seq_len=args.max_seq_len,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_loop_iters=args.max_loop_iters,
        moe_aux_loss_weight=args.moe_aux_weight,
        act_loop_penalty=args.act_loop_penalty,
    )
    model = CodeRecurrentModel(cfg)
    trainable = model.get_trainable_params()
    print(f"Trainable parameters: {trainable:,}")

    os.makedirs(args.output_dir, exist_ok=True)

    resume_path = args.resume or pull_checkpoint_from_hub(args.hub_repo_id, args.output_dir)
    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu")
        model.refiner.load_state_dict(ckpt["refiner"])
        model.lm_head.load_state_dict(ckpt["lm_head"])
        print(f"Resumed from {resume_path}")

    samples = load_code_samples(args.dataset)

    rng = random.Random(0)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    n_val = int(len(indices) * args.val_frac) if len(indices) > 1 else 0
    val_indices = indices[:n_val]
    train_indices = indices[n_val:] if n_val > 0 else indices

    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]
    print(f"Dataset: {len(train_samples)} train / {len(val_samples)} val samples")

    collator = CodeCompletionCollator(tokenizer, args.max_seq_len)
    train_ds = CodeDataset(train_samples)
    loader = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=collator, shuffle=True)

    val_loader = None
    if val_samples:
        val_ds = CodeDataset(val_samples)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collator, shuffle=False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.1)

    total_steps = args.max_steps
    warmup = int(0.05 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    step = 0
    ema_loss = None
    best_loss = float("inf")
    last_val_loss = None
    data_iter = iter(loader)

    torch.cuda.empty_cache()

    print(f"Starting training for {total_steps} steps...")
    while step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            continue

        input_ids, attention_mask = batch
        device = next(model.base.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        ce_loss, moe_aux_loss, act_ponder_loss, n_tokens = train_step(
            model, input_ids, attention_mask, optimizer, scheduler,
            args.moe_aux_weight, args.act_loop_penalty,
        )
        ema_loss = ce_loss if ema_loss is None else 0.98 * ema_loss + 0.02 * ce_loss
        ppl = math.exp(ce_loss)
        sr = model.refiner.get_spectral_radius()

        print(
            f"Step {step:>4d}/{total_steps} | "
            f"Loss {ce_loss:.4f} (EMA {ema_loss:.4f}) | MoE-aux {moe_aux_loss:.4f} | "
            f"Depth {act_ponder_loss:.2f}/{args.max_loop_iters} | PPL {ppl:.2f} | "
            f"rho(A) {sr:.4f} | Tok {n_tokens}"
        )

        if step % args.eval_every == 0 and step > 0 and val_loader is not None:
            last_val_loss = evaluate(model, val_loader)
            print(f"  [eval] step {step} | val_loss {last_val_loss:.4f} | val_ppl {math.exp(last_val_loss):.2f}")

        if step % args.save_every == 0 and step > 0:
            ckpt = {
                "refiner": model.refiner.state_dict(),
                "lm_head": model.lm_head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "loss": ce_loss,
            }
            path = f"{args.output_dir}/step_{step}.pt"
            torch.save(ckpt, path)
            print(f"Saved {path}")
            push_checkpoint_to_hub(args.hub_repo_id, path)

            score = last_val_loss if last_val_loss is not None else ema_loss
            if score < best_loss:
                best_loss = score
                best_path = f"{args.output_dir}/best.pt"
                torch.save(ckpt, best_path)
                push_checkpoint_to_hub(args.hub_repo_id, best_path)

        step += 1

    final_path = f"{args.output_dir}/final.pt"
    torch.save(
        {"refiner": model.refiner.state_dict(), "lm_head": model.lm_head.state_dict(), "config": cfg},
        final_path,
    )
    push_checkpoint_to_hub(args.hub_repo_id, final_path)
    print("Done!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"[TRAIN ERROR] {e}")
        traceback.print_exc()
        raise
