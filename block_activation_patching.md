# Interpretability Blog


## Question
   Where does the decision signal for a token emerge in GPT-2?


## Experiment
   Activation patching with France/Germany prompts.


## Code
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
model.eval()

prompt_A = "The capital of France is"
prompt_B = "The capital of Germany is"

# GPT-2 tokens are space-sensitive: usually " Paris" is a single token, "Paris" may not be.
paris_ids = tokenizer.encode(" Paris", add_special_tokens=False)
assert len(paris_ids) == 1, f"' Paris' is not a single token: {paris_ids}"
paris_id = paris_ids[0]

def prob_of_token(prompt: str, token_id: int) -> float:
    """P(token_id as next token | prompt) using the real final layer output."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, use_cache=False)
        logits = out.logits[:, -1, :]  # next-token logits
        p = torch.softmax(logits, dim=-1)[0, token_id].item()
    return p

def get_layer_outputs_at_last_pos(prompt: str):
    """
    Returns a list of tensors, one per transformer block (12 for GPT-2 small),
    each tensor is the block output hidden state at the last position: shape (1, d_model).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    # out.hidden_states: [embeds, after block0, after block1, ..., after block11]
    # We'll store after each block (skip embeddings at index 0)
    per_block_lastpos = []
    for hs in out.hidden_states[1:]:
        per_block_lastpos.append(hs[:, -1, :].detach().clone())
    return per_block_lastpos

# 1) Baselines
pA = prob_of_token(prompt_A, paris_id)
pB = prob_of_token(prompt_B, paris_id)

print(f'Baseline P(" Paris" | A="{prompt_A}") = {pA:.6f}')
print(f'Baseline P(" Paris" | B="{prompt_B}") = {pB:.6f}')
print()

# 2) Precompute layer outputs from A (what we will patch in)
A_block_lastpos = get_layer_outputs_at_last_pos(prompt_A)  # length 12

# 3) Activation patching: patch one layer at a time while running prompt B
def patched_prob_at_layer(layer_idx_0_based: int) -> float:
    """
    Patches GPT2Block output at layer_idx_0_based (0..11),
    replacing the last-position hidden state with PromptA's saved one.
    Then returns P(" Paris" | PromptB with patch).
    """
    block = model.transformer.h[layer_idx_0_based]
    patch_vec = A_block_lastpos[layer_idx_0_based]  # shape (1, d_model)

    def hook_fn(module, inp, out):
        # GPT2Block output can be a tensor or a tuple (hidden_states, present, ...)
        if isinstance(out, tuple):
            h = out[0]
            rest = out[1:]
        else:
            h = out
            rest = None

        # Replace last token position with patch_vec
        h = h.clone()
        h[:, -1, :] = patch_vec

        if rest is None:
            return h
        else:
            return (h, *rest)

    handle = block.register_forward_hook(hook_fn)
    try:
        # Run prompt B forward pass with the hook active
        inputs = tokenizer(prompt_B, return_tensors="pt").to(device)
        with torch.no_grad():
            outB = model(**inputs, use_cache=False)
            logits = outB.logits[:, -1, :]
            p = torch.softmax(logits, dim=-1)[0, paris_id].item()
        return p
    finally:
        handle.remove()

print("Layer-wise patching results (patching last-position residual at each block):")
print("Layer(after block) | P(' Paris' | B patched at this layer) | Delta vs baseline B")
print("-"*78)

for l in range(12):  # GPT-2 small has 12 blocks
    p_patch = patched_prob_at_layer(l)
    delta = p_patch - pB
    # Report as layer number 1..12 to match hidden_states indexing after blocks
    print(f"{l+1:>15} | {p_patch:>30.6f} | {delta:>+16.6f}")
```


## Results
```
Baseline P(" Paris" | A="The capital of France is") = 0.032245
Baseline P(" Paris" | B="The capital of Germany is") = 0.001311

Layer-wise patching results (patching last-position residual at each block):
Layer(after block) | P(' Paris' | B patched at this layer) | Delta vs baseline B
------------------------------------------------------------------------------
              1 |                       0.001312 |        +0.000001
              2 |                       0.001294 |        -0.000017
              3 |                       0.001330 |        +0.000018
              4 |                       0.001324 |        +0.000013
              5 |                       0.001383 |        +0.000072
              6 |                       0.001426 |        +0.000115
              7 |                       0.001348 |        +0.000037
              8 |                       0.001355 |        +0.000044
              9 |                       0.002086 |        +0.000775
             10 |                       0.020649 |        +0.019338
             11 |                       0.034367 |        +0.033056
             12 |                       0.000203 |        -0.001108
```


## Interpretation

In this experiment the residual stream representation of the final token ("is") in PromptB is replaced with the corresponding representation from PromptA. This patching is performed at each transformer layer (Layer 1 through Layer 12), and we measure the probability of the token " Paris" given PromptB.

The results show that patching the residual stream representations in early layers has almost no effect on the probability of " Paris". This suggests that the representations in these early layers are not yet sufficient to cause the Paris prediction. When early layer activations are replaced with those from the France prompt, later layers process this patched representation together with the remaining Germany-context representations, and the injected signal is effectively overridden.

In contrast, patching late layers significantly increases the probability of " Paris". This indicates that by these layers the residual stream at the final token already contains a representation that is sufficient to produce the correct next token under the model’s output head. We see a jump in probability in later layers which hint that the signal in "is" token at later layer contains sufficient information for model to make the probability of Paris high. 

Importantly, activation patching does not reveal what information earlier layers encode; it only identifies the layer at which the output-relevant signal becomes sufficient to influence the model’s prediction.

Interestingly, patching the final layer (Layer 12) reduces the probability of " Paris". This suggests that the final layer may play a role in reshaping or calibrating the representation before the output projection. However, the exact reason for this behavior cannot be determined from this experiment alone.


## Open questions
   Why does layer 12 behave differently?
