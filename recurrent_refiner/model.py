import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from .config import RefinerConfig
from .stable_recurrence import StableRecurrence, RecurrentTransformerBlock


class LoopIndexPE(nn.Module):
    def __init__(self, dim: int, max_loops: int = 8):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_loops = max_loops
        self._build_cache()

    def _build_cache(self):
        idx = torch.arange(self.max_loops, device=self.inv_freq.device).float().unsqueeze(-1)
        freqs = idx * self.inv_freq.unsqueeze(0)
        pe = torch.cat([freqs.sin(), freqs.cos()], dim=-1)
        self.register_buffer("cache", pe, persistent=False)

    def forward(self, loop_idx: int):
        if loop_idx < self.max_loops:
            return self.cache[loop_idx]
        # loop_idx exceeds what was precomputed - compute on the fly instead
        # of crashing on an out-of-range index.
        idx = torch.tensor([float(loop_idx)], device=self.inv_freq.device)
        freqs = idx.unsqueeze(-1) * self.inv_freq.unsqueeze(0)
        pe = torch.cat([freqs.sin(), freqs.cos()], dim=-1)
        return pe[0]


class RecurrentRefiner(nn.Module):
    """Recurrent reasoning block with ACT-style adaptive halting.

    Instead of a fixed `n_loops` chosen at generation time (v1's approach,
    which broke whenever it didn't match what was used in training), each
    token predicts its own per-loop halting probability. The final
    representation is a probability-weighted blend of the hidden state at
    each loop depth, so there's no single depth hyperparameter to desync
    between training and inference - only `max_loop_iters`, a safety ceiling
    that's fine to keep fixed.
    """

    def __init__(self, config: RefinerConfig):
        super().__init__()
        self.config = config
        self.max_loop_iters = config.max_loop_iters
        self.hidden_size = config.refiner_hidden

        self.input_proj = nn.Linear(config.hidden_size, config.refiner_hidden, bias=False)
        self.output_proj = nn.Linear(config.refiner_hidden, config.hidden_size, bias=False)

        self.stable_rec = StableRecurrence(config.refiner_hidden, config.dt_init)

        self.layers = nn.ModuleList([
            RecurrentTransformerBlock(
                hidden=config.refiner_hidden,
                n_heads=config.refiner_heads,
                n_routed=config.n_routed_experts,
                n_shared=config.n_shared_experts,
                n_per_tok=config.n_experts_per_tok,
                expert_hidden=config.expert_hidden,
                lora_rank=config.lora_rank,
            )
            for _ in range(config.refiner_layers)
        ])

        self.loop_pe = LoopIndexPE(config.refiner_hidden, max_loops=config.max_loop_iters)
        self.halt_proj = nn.Linear(config.refiner_hidden, 1)
        self.act_epsilon = config.act_epsilon
        self.ln_f = nn.LayerNorm(config.refiner_hidden)

    def _loop_step(self, h: torch.Tensor, e: torch.Tensor, t: int):
        loop_pe = self.loop_pe(t)
        h = h + loop_pe.view(1, 1, -1)
        h = self.stable_rec(h, e)
        moe_aux_loss = h.new_zeros(())
        for layer in self.layers:
            h, layer_aux = layer(h, e)
            moe_aux_loss = moe_aux_loss + layer_aux
        return h, moe_aux_loss

    def forward(self, base_hidden: torch.Tensor, n_loops: int = None):
        max_loops = n_loops or self.max_loop_iters

        e = self.input_proj(base_hidden)
        h = e.clone()
        B, T, _ = h.shape

        halted = h.new_zeros(B, T)
        still_running = h.new_ones(B, T)
        ponder_cost = h.new_zeros(B, T)
        weighted_h = torch.zeros_like(h)
        total_moe_aux = base_hidden.new_zeros(())

        use_checkpoint = self.training and self.config.gradient_checkpointing

        for t in range(max_loops):
            if use_checkpoint:
                h, moe_aux = torch.utils.checkpoint.checkpoint(
                    self._loop_step, h, e, t, use_reentrant=False
                )
            else:
                h, moe_aux = self._loop_step(h, e, t)
            total_moe_aux = total_moe_aux + moe_aux

            p = torch.sigmoid(self.halt_proj(h)).squeeze(-1)
            is_last_loop = (t == max_loops - 1)

            would_cross = (halted + p * still_running) >= (1.0 - self.act_epsilon)
            halts_now = (would_cross | is_last_loop) & (still_running > 0)

            remainder = (1.0 - halted).clamp(min=0.0)
            weight = torch.where(halts_now, remainder, p) * still_running

            weighted_h = weighted_h + weight.unsqueeze(-1) * h
            ponder_cost = ponder_cost + still_running
            halted = halted + weight
            still_running = still_running * (~halts_now).float()

        moe_aux_loss = total_moe_aux / max_loops
        act_ponder_loss = ponder_cost.mean()

        out = self.ln_f(weighted_h)
        return self.output_proj(out), moe_aux_loss, act_ponder_loss

    @torch.no_grad()
    def get_spectral_radius(self) -> float:
        return self.stable_rec.get_spectral_radius()


class CodeRecurrentModel(nn.Module):
    def __init__(self, config: RefinerConfig, device_index: int = 0):
        super().__init__()
        self.config = config
        print(f"Loading base model: {config.base_model_id} (4-bit={config.load_in_4bit})")

        load_kwargs = dict(trust_remote_code=True)
        if config.load_in_4bit:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "load_in_4bit=True requires a CUDA GPU (bitsandbytes 4-bit has no "
                    "CPU kernel). Set load_in_4bit=False for CPU-only runs."
                )
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            # device_map="auto" tends to be overly conservative on a single
            # small GPU and offloads part of the model to CPU/disk, which
            # bitsandbytes 4-bit doesn't support without extra flags. Pinning
            # everything to one GPU avoids that (standard pattern for bnb
            # 4-bit). device_index lets each DDP rank pin its own copy of the
            # frozen base to its own GPU (rank N -> cuda:N).
            load_kwargs["device_map"] = {"": device_index}
        else:
            load_kwargs["dtype"] = torch.bfloat16

        hf_base = AutoModelForCausalLM.from_pretrained(config.base_model_id, **load_kwargs)

        detected_hidden = hf_base.config.hidden_size
        if config.hidden_size != detected_hidden:
            print(f"Auto-detected hidden_size={detected_hidden}, overriding config")
            object.__setattr__(config, "hidden_size", detected_hidden)
        if config.vocab_size != hf_base.config.vocab_size:
            print(f"Auto-detected vocab_size={hf_base.config.vocab_size}, overriding config")
            object.__setattr__(config, "vocab_size", hf_base.config.vocab_size)

        if hasattr(hf_base, "lm_head") and hf_base.lm_head is not None:
            self.lm_head = hf_base.lm_head
            for p in self.lm_head.parameters():
                p.requires_grad = False
            self._use_base_head = True
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            nn.init.normal_(self.lm_head.weight, std=0.02)
            self._use_base_head = False

        self.base = hf_base
        for p in self.base.parameters():
            p.requires_grad = False
        self.base.eval()

        self.refiner = RecurrentRefiner(config)

        # With load_in_4bit + device_map="auto", the base places itself on
        # its target device automatically (accelerate/bitsandbytes handle
        # this at load time - calling .cuda() on the whole model afterward,
        # like v1 did, does NOT work with 4-bit modules). Follow suit here
        # so refiner/lm_head end up on the same device without the caller
        # needing to do anything.
        base_device = next(self.base.parameters()).device
        if base_device.type == "cuda":
            self.refiner = self.refiner.to(base_device)
            if not self._use_base_head:
                self.lm_head = self.lm_head.to(base_device)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None,
                n_loops: int = None):
        with torch.no_grad():
            base_out = self.base(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

        base_hidden = base_out.hidden_states[-1].to(self.refiner.input_proj.weight.dtype)
        refined, moe_aux_loss, act_ponder_loss = self.refiner(base_hidden, n_loops=n_loops)
        logits = self.lm_head(refined.to(self.lm_head.weight.dtype))
        return logits, moe_aux_loss, act_ponder_loss

    @torch.no_grad()
    def _generate_with_cache(self, input_ids: torch.Tensor, n_loops: int = None,
                              max_new: int = 128, temperature: float = 0.7,
                              eos_token_id: int = None) -> torch.Tensor:
        if eos_token_id is None:
            eos_token_id = getattr(self.base.config, "eos_token_id", 0)

        dtype = self.refiner.input_proj.weight.dtype
        past_length = input_ids.shape[1]

        base_out = self.base(
            input_ids, output_hidden_states=True, return_dict=True, use_cache=True
        )
        base_hidden = base_out.hidden_states[-1].to(dtype)
        past_key_values = base_out.past_key_values

        for _ in range(max_new):
            refined, _, _ = self.refiner(base_hidden, n_loops=n_loops)
            logits = self.lm_head(refined.to(self.lm_head.weight.dtype))
            next_logit = logits[:, -1, :] / temperature
            next_id = torch.multinomial(F.softmax(next_logit, dim=-1), 1)
            input_ids = torch.cat([input_ids, next_id], dim=1)

            if next_id.item() == eos_token_id:
                break

            next_out = self.base(
                next_id, past_key_values=past_key_values,
                output_hidden_states=True, return_dict=True, use_cache=True,
            )
            past_key_values = next_out.past_key_values
            next_h = next_out.hidden_states[-1].to(dtype)
            base_hidden = torch.cat([base_hidden, next_h[:, -1:, :]], dim=1)

        return input_ids[:, past_length:]

    def generate(self, tokenizer, prompt: str, max_new: int = 128,
                 n_loops: int = None, temperature: float = 0.7):
        self.eval()
        inputs = tokenizer(prompt, return_tensors="pt").to(next(self.base.parameters()).device)
        out = self._generate_with_cache(
            inputs.input_ids, n_loops=n_loops,
            max_new=max_new, temperature=temperature,
        )
        return tokenizer.decode(out[0], skip_special_tokens=True)

    def get_trainable_params(self):
        return sum(p.numel() for p in self.refiner.parameters() if p.requires_grad) + \
               (0 if self._use_base_head else sum(p.numel() for p in self.lm_head.parameters() if p.requires_grad))
