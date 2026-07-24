import torch
import torch.nn as nn
import torch.nn.functional as F


class StableRecurrence(nn.Module):
    def __init__(self, dim: int, dt_init: float = 0.1):
        super().__init__()
        self.dim = dim
        log_A = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(log_A, mean=-1.0, std=0.1)
        self.log_A = log_A

        log_dt = nn.Parameter(torch.tensor(dt_init).log())
        self.register_parameter("log_dt", log_dt)

        self.B_proj = nn.Linear(dim, dim, bias=False)
        nn.init.orthogonal_(self.B_proj.weight, gain=0.9)

    @property
    def A_diag(self):
        dt = self.log_dt.exp()
        A_cont = -self.log_A.exp()
        return (A_cont * dt).exp()

    def get_spectral_radius(self) -> float:
        return self.A_diag.abs().max().item()

    def forward(self, h: torch.Tensor, proj_e: torch.Tensor) -> torch.Tensor:
        return h * self.A_diag.unsqueeze(0).unsqueeze(0) + self.B_proj(proj_e)


class RecurrentMoEFFN(nn.Module):
    def __init__(self, hidden: int, expert_hidden: int,
                 n_routed: int, n_shared: int, n_per_tok: int):
        super().__init__()
        self.n_routed = n_routed
        self.n_shared = n_shared
        self.n_per_tok = n_per_tok

        self.shared = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, expert_hidden, bias=False),
                nn.GELU(),
                nn.Linear(expert_hidden, hidden, bias=False),
            )
            for _ in range(n_shared)
        ])

        self.routed_gate = nn.Linear(hidden, n_routed, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, expert_hidden, bias=False),
                nn.GELU(),
                nn.Linear(expert_hidden, hidden, bias=False),
            )
            for _ in range(n_routed)
        ])

    def forward(self, x: torch.Tensor):
        shared_out = sum(shared(x) for shared in self.shared)

        logits = self.routed_gate(x)
        probs = F.softmax(logits, dim=-1)
        weights, indices = torch.topk(probs, self.n_per_tok, dim=-1)

        flat_x = x.view(-1, x.size(-1))
        flat_indices = indices.view(-1, self.n_per_tok)
        flat_weights = weights.view(-1, self.n_per_tok)
        flat_probs = probs.view(-1, self.n_routed)

        routed_out = torch.zeros_like(flat_x)
        for e_idx in range(self.n_routed):
            for k in range(self.n_per_tok):
                mask = (flat_indices[:, k] == e_idx)
                if mask.any():
                    routed_out[mask] = routed_out[mask] + flat_weights[mask, k:k+1] * self.experts[e_idx](flat_x[mask])

        # Switch-Transformer-style load-balancing loss: penalizes routing that
        # collapses onto a few experts. 1.0 == perfectly uniform routing.
        n_tokens = flat_x.size(0)
        dispatch = F.one_hot(flat_indices, num_classes=self.n_routed).float()
        frac_dispatched = dispatch.sum(dim=(0, 1)) / (n_tokens * self.n_per_tok)
        mean_prob = flat_probs.mean(dim=0)
        aux_loss = self.n_routed * (frac_dispatched * mean_prob).sum()

        return shared_out + routed_out.view_as(x), aux_loss


class RecurrentTransformerBlock(nn.Module):
    def __init__(self, hidden: int, n_heads: int,
                 n_routed: int, n_shared: int, n_per_tok: int,
                 expert_hidden: int, lora_rank: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)

        head_dim = hidden // n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

        self.lora_rank = lora_rank
        if lora_rank > 0:
            self.q_lora_A = nn.Parameter(torch.zeros(hidden, lora_rank))
            self.q_lora_B = nn.Parameter(torch.zeros(lora_rank, hidden))
            self.v_lora_A = nn.Parameter(torch.zeros(hidden, lora_rank))
            self.v_lora_B = nn.Parameter(torch.zeros(lora_rank, hidden))

            for p in [self.q_lora_A, self.q_lora_B, self.v_lora_A, self.v_lora_B]:
                nn.init.normal_(p, std=0.02)

        self.ffn = RecurrentMoEFFN(hidden, expert_hidden, n_routed, n_shared, n_per_tok)

    def forward(self, x: torch.Tensor, input_e: torch.Tensor):
        residual = x
        x = self.ln1(x)
        B, T, D = x.shape

        x_inj = x + input_e

        q = self.q_proj(x_inj)
        k = self.k_proj(x_inj)
        v = self.v_proj(x_inj)

        if self.lora_rank > 0:
            lora_scale = q.new_tensor(1.0 / self.lora_rank)
            q = q + (x_inj @ self.q_lora_A @ self.q_lora_B) * lora_scale
            v = v + (x_inj @ self.v_lora_A @ self.v_lora_B) * lora_scale

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, D)
        x = residual + self.o_proj(attn)
        ffn_out, aux_loss = self.ffn(self.ln2(x))
        x = x + ffn_out
        return x, aux_loss
