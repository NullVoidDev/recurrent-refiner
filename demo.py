import torch
from recurrent_refiner import RefinerConfig, CodeRecurrentModel
from transformers import AutoTokenizer

cfg = RefinerConfig(base_model_id="Qwen/Qwen2.5-Coder-7B-Instruct")

model = CodeRecurrentModel(cfg)
print(f"Trainable params: {model.get_trainable_params():,}")

sr = model.refiner.get_spectral_radius()
print(f"Spectral radius {sr:.4f} (must be < 1)")

tok = AutoTokenizer.from_pretrained(cfg.base_model_id, trust_remote_code=True)
out = model.generate(tok, "Write a function to compute fibonacci numbers\n", max_new=64)
print(f"\nGenerated:\n{out}")
