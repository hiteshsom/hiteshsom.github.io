# Interpretability Blog

## Question
Which layers is responsible the most for decision making of a token in GPT-2

## Experiment
Layer patching with France/Germany prompts.

## Code
```python
"""
Attention vs MLP activation patching (GPT-2, HuggingFace).

Extends block-level patching: for each layer, compare patching only the post-attn
residual vs only the MLP branch output vs the full block output (all at last token).
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
model.eval()

n_layer = model.config.n_layer

prompt_A = "The capital of France is"
prompt_B = "The capital of Germany is"

paris_ids = tokenizer.encode(" Paris", add_special_tokens=False)
assert len(paris_ids) == 1, f"' Paris' is not a single token: {paris_ids}"
paris_id = paris_ids[0]


def prob_next_token(prompt: str, token_id: int) -> float:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, use_cache=False)
        logits = out.logits[:, -1, :]
        return torch.softmax(logits, dim=-1)[0, token_id].item()


def get_block_outputs_last_pos(prompt: str) -> list[torch.Tensor]:
    """runs the model on one prompt and returns, for each transformer block, the hidden state at the 
    last sequence position after that block."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    return [hs[:, -1, :].detach().clone() for hs in out.hidden_states[1:]]


def collect_sublayer_activations_from_prompt(prompt: str):
    """
    One forward on `prompt`; for each layer store:
      - h_mid: residual after attention add (last position)
      - mlp_raw: MLP output before second residual add (last position)
    """
    res_in = [None] * n_layer
    attn_slice = [None] * n_layer
    mlp_raw = [None] * n_layer

    def pre_block(layer_idx):
        def hook(module, inputs):
            x = inputs[0]
            res_in[layer_idx] = x[:, -1, :].detach().clone()

        return hook

    def hook_attn(layer_idx):
        def hook(module, inputs, output):
            t = output[0] if isinstance(output, tuple) else output
            attn_slice[layer_idx] = t[:, -1, :].detach().clone()

        return hook

    def hook_mlp(layer_idx):
        def hook(module, inputs, output):
            mlp_raw[layer_idx] = output[:, -1, :].detach().clone()

        return hook

    handles = []
    try:
        for l in range(n_layer):
            block = model.transformer.h[l]
            handles.append(block.register_forward_pre_hook(pre_block(l)))
            handles.append(block.attn.register_forward_hook(hook_attn(l)))
            handles.append(block.mlp.register_forward_hook(hook_mlp(l)))

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    h_mid = [res_in[l] + attn_slice[l] for l in range(n_layer)]
    return h_mid, mlp_raw


def _prob_B_with_hooks(hook_installers: list):
    handles = []
    try:
        for install in hook_installers:
            install(handles)
        inputs = tokenizer(prompt_B, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, use_cache=False)
            logits = out.logits[:, -1, :]
            return torch.softmax(logits, dim=-1)[0, paris_id].item()
    finally:
        for h in handles:
            h.remove()


def patched_prob_full_block(layer: int, h_out_A: list[torch.Tensor]) -> float:
    block = model.transformer.h[layer]
    patch = h_out_A[layer]

    def hook_fn(module, inp, out):
        if isinstance(out, tuple):
            h = out[0].clone()
            rest = out[1:]
            h[:, -1, :] = patch
            return (h, *rest)
        h = out.clone()
        h[:, -1, :] = patch
        return h

    def install(handles):
        handles.append(block.register_forward_hook(hook_fn))

    return _prob_B_with_hooks([install])


def patched_prob_attn_only(layer: int, h_mid_A: list[torch.Tensor]) -> float:
    block = model.transformer.h[layer]
    patch_mid = h_mid_A[layer]
    res_B = [None]

    def pre_hook(module, inputs):
        x = inputs[0]
        res_B[0] = x[:, -1, :].clone()

    def attn_hook(module, inputs, output):
        if isinstance(output, tuple):
            a = output[0].clone()
            a[:, -1, :] = patch_mid - res_B[0]
            return (a,) + tuple(output[1:])
        a = output.clone()
        a[:, -1, :] = patch_mid - res_B[0]
        return a

    def install(handles):
        handles.append(block.register_forward_pre_hook(pre_hook))
        handles.append(block.attn.register_forward_hook(attn_hook))

    return _prob_B_with_hooks([install])


def patched_prob_mlp_only(layer: int, mlp_raw_A: list[torch.Tensor]) -> float:
    block = model.transformer.h[layer]
    patch_mlp = mlp_raw_A[layer]

    def mlp_hook(module, inputs, output):
        o = output.clone()
        o[:, -1, :] = patch_mlp
        return o

    def install(handles):
        handles.append(block.mlp.register_forward_hook(mlp_hook))

    return _prob_B_with_hooks([install])


def main():
    pA = prob_next_token(prompt_A, paris_id)
    pB = prob_next_token(prompt_B, paris_id)
    print(f'Baseline P(" Paris" | A) = {pA:.6f}')
    print(f'Baseline P(" Paris" | B) = {pB:.6f}\n')

    h_mid_A, mlp_raw_A = collect_sublayer_activations_from_prompt(prompt_A)
    h_out_A = get_block_outputs_last_pos(prompt_A)

    print("Layer | P_full  d_full | P_attn  d_attn | P_mlp  d_mlp")
    print("-" * 72)

    best_full_delta = 0
    best_layer = -1
    rows = []
    for l in range(n_layer):
        p_full = patched_prob_full_block(l, h_out_A)
        p_attn = patched_prob_attn_only(l, h_mid_A)
        p_mlp = patched_prob_mlp_only(l, mlp_raw_A)
        d_full = p_full - pB
        d_attn = p_attn - pB
        d_mlp = p_mlp - pB
        rows.append((d_full, d_attn, d_mlp))
        if abs(d_full) > abs(best_full_delta):
            best_full_delta = d_full
            best_layer = l
        print(
            f"{l+1:5d} | {p_full:.4f} {d_full:+7.4f} | "
            f"{p_attn:.4f} {d_attn:+7.4f} | {p_mlp:.4f} {d_mlp:+7.4f}"
        )

    d_full_b, d_attn_b, d_mlp_b = rows[best_layer]
    print()
    print(
        f"Largest |d_full| at layer {best_layer + 1}: "
        f"d_full={d_full_b:+.4f}, d_attn={d_attn_b:+.4f}, d_mlp={d_mlp_b:+.4f}"
    )
    print(
        "If d_attn is closer to d_full than d_mlp, attention dominates that layer's patch effect; "
        "if d_mlp is closer, MLP dominates."
    )


if __name__ == "__main__":
    main()
```

## Result
```bash
Baseline P(" Paris" | A) = 0.032245
Baseline P(" Paris" | B) = 0.001311

Layer | P_full  d_full | P_attn  d_attn | P_mlp  d_mlp
------------------------------------------------------------------------
    1 | 0.0013 +0.0000 | 0.0013 +0.0000 | 0.0013 +0.0000
    2 | 0.0013 -0.0000 | 0.0013 -0.0000 | 0.0013 +0.0000
    3 | 0.0013 +0.0000 | 0.0013 +0.0000 | 0.0013 +0.0000
    4 | 0.0013 +0.0000 | 0.0013 +0.0000 | 0.0013 +0.0000
    5 | 0.0014 +0.0001 | 0.0014 +0.0001 | 0.0014 +0.0001
    6 | 0.0014 +0.0001 | 0.0014 +0.0001 | 0.0013 -0.0000
    7 | 0.0013 +0.0000 | 0.0013 +0.0000 | 0.0011 -0.0002
    8 | 0.0014 +0.0000 | 0.0014 +0.0000 | 0.0012 -0.0001
    9 | 0.0021 +0.0008 | 0.0021 +0.0008 | 0.0012 -0.0001
   10 | 0.0206 +0.0193 | 0.0206 +0.0193 | 0.0015 +0.0002
   11 | 0.0344 +0.0331 | 0.0344 +0.0331 | 0.0017 +0.0004
   12 | 0.0002 -0.0011 | 0.0322 +0.0309 | 0.0017 +0.0004

Largest |d_full| at layer 11: d_full=+0.0331, d_attn=+0.0331, d_mlp=+0.0004
If d_attn is closer to d_full than d_mlp, attention dominates that layer's patch effect; if d_mlp is closer, MLP dominates.
```

## Interpretation
P_attn say the highest jumps at Layer/Block 10 and Layer 11. P_mlp jumps is almost same accross all layers.
So basically the P_full increase at Layer 10 and Layer 11 is mainly caused by attention layers. Hence attention layers are mainly responsible in 
deciding the decision for next token.

