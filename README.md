# Recurrent Refiner v2

Treina um pequeno bloco recorrente de raciocínio ("refiner") em cima de um modelo de
código congelado — sem re-treinar o modelo base do zero. O refiner é o único módulo
treinável; o base model nunca recebe gradiente.

```
[Qwen2.5-Coder-7B-Instruct congelado, 4-bit] -> hidden_states -> [Recurrent Refiner] -> logits
```

## O que mudou da v1

- **Base model**: `Qwen2.5-Coder-1.5B` → `Qwen2.5-Coder-7B-Instruct`, carregado em 4-bit
  (`bitsandbytes`) para caber em GPUs de 16GB (Kaggle/Colab) e na GTX 1660 Ti (6GB) local.
- **Halting adaptativo (ACT)** no lugar de `n_loops` fixo: cada token aprende sua própria
  probabilidade de parar a cada loop; a saída final é uma combinação ponderada dos estados
  em cada profundidade. Elimina a classe de bug onde treino e geração usavam número de
  loops diferentes (documentado em `{Bugs}/recurrent-refiner-n-loops-mismatch-treino-inferencia.md`
  no Obsidian).
- **Checkpoint bus via HuggingFace Hub** (`--hub_repo_id`): permite treinar em rodízio
  entre plataformas gratuitas (Kaggle, Lightning AI Studio, Colab) sem lógica de
  salvar/baixar específica de cada uma.

Arquitetura interna (SSM diagonal estável, MoE shared+routed com load-balancing loss,
LoRA na atenção, gradient checkpointing) reaproveitada da v1, já validada.

## Componentes

| Componente | Arquivo | Função |
|---|---|---|
| `StableRecurrence` | `stable_recurrence.py` | Recorrência LTI estável, raio espectral < 1 garantido |
| `RecurrentTransformerBlock` | `stable_recurrence.py` | Atenção causal + MoE FFN com injeção residual |
| `RecurrentMoEFFN` | `stable_recurrence.py` | MoE com experts roteados (top-k) + compartilhados, com load-balancing loss |
| `LoopIndexPE` | `model.py` | Positional encoding por índice do loop, somado ao hidden state |
| `RecurrentRefiner` | `model.py` | Loop com halting adaptativo (ACT) + todos os componentes acima |
| `CodeRecurrentModel` | `model.py` | Wrapper completo: base 4-bit congelado + refiner + lm_head + geração com KV-cache |
| `RefinerConfig` | `config.py` | Todos os hiperparâmetros |

## Uso

```bash
pip install -e .
python -m recurrent_refiner.train --hub_repo_id seu-usuario/recurrent-refiner-ckpts
```

Sem `--hub_repo_id`, salva só localmente em `refiner_checkpoints/`. Sem `--dataset`,
usa `mbpp` por padrão (cai para um corpus inline se não conseguir baixar).

## Meta realista

Loop/recorrência aumenta profundidade de raciocínio por parâmetro, não conhecimento de
mundo. A meta é um modelo pequeno que raciocina desproporcionalmente bem pro tamanho
dele — não "8B equivalente a 70B em tudo".

## Referência

Inspirado (não copiado) em ideias de `kyegomez/OpenMythos` — um repositório especulativo
e não-oficial, sem qualquer afiliação confirmada com a Anthropic.
