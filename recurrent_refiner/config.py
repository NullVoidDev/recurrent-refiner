from dataclasses import dataclass


@dataclass
class RefinerConfig:
    base_model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    load_in_4bit: bool = True

    # Auto-detected from the base model at load time (CodeRecurrentModel.__init__
    # overrides these) - values here are just a reasonable starting guess.
    hidden_size: int = 3584
    vocab_size: int = 151936

    refiner_hidden: int = 1024
    refiner_layers: int = 2
    refiner_heads: int = 8

    # Safety cap on loop iterations. ACT halting decides the *effective* depth
    # per token dynamically; this is only a ceiling, not a fixed depth to match
    # between training and inference (see v1's n_loops-mismatch bug).
    max_loop_iters: int = 8
    act_epsilon: float = 0.01
    act_loop_penalty: float = 0.01

    n_shared_experts: int = 1
    n_routed_experts: int = 4
    n_experts_per_tok: int = 2
    expert_hidden: int = 1024
    moe_aux_loss_weight: float = 0.01

    dt_init: float = 0.1
    lora_rank: int = 16
    lora_alpha: float = 32.0

    gradient_checkpointing: bool = True
    max_seq_len: int = 2048
    learning_rate: float = 5e-5
    batch_size: int = 1
    grad_accum_steps: int = 8
