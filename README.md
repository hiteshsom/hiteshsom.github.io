## Interpretability Blog

### Question:
What does hidden state outputs in an LLM tell/mean/represent?


### Test
Comparison of hidden state outputs of almost same sentence.


#### Code
```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModel.from_pretrained("gpt2")

inputs = tokenizer(
    "The capital of France is Paris.",
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

hidden_states = outputs.hidden_states

inputs_berlin = tokenizer(
    "The capital of Germany is Berlin.",
    return_tensors="pt"
)

with torch.no_grad():
    outputs_berlin = model(**inputs_berlin, output_hidden_states=True)

hidden_states_berlin = outputs_berlin.hidden_states

print("The capital of France is Paris.")
for e_paris, e_berlin in zip(hidden_states, hidden_states_berlin):
    mean_per_word = (e_paris-e_berlin).mean(dim=-1)
    std_per_word  = (e_paris-e_berlin).std(dim=-1)
    l2_norm = torch.norm(e_paris-e_berlin, dim=-1)
    print("L2 Norm: ", l2_norm)
    # print("Mean: ", mean_per_word)
    # print("Std: ", std_per_word)
    print()
```

#### Output:
```
The capital of France is Paris.
L2 Norm:  tensor([[0.0000, 0.0000, 0.0000, 2.5789, 0.0000, 3.3821, 0.0000]])

L2 Norm:  tensor([[ 0.0000,  0.0000,  0.0000, 31.3441,  1.8658, 38.6030,  2.8238]])

L2 Norm:  tensor([[ 0.0000,  0.0000,  0.0000, 32.1649,  1.4644, 40.2342,  2.6202]])

L2 Norm:  tensor([[ 0.0000,  0.0000,  0.0000, 34.8764,  2.4247, 42.6538,  3.7220]])

L2 Norm:  tensor([[ 0.0000,  0.0000,  0.0000, 38.1582,  3.3115, 45.0038,  5.4377]])

L2 Norm:  tensor([[ 0.0000,  0.0000,  0.0000, 39.3070,  5.9318, 46.7198,  7.6872]])

L2 Norm:  tensor([[ 0.0000,  0.0000,  0.0000, 42.2318,  5.5535, 49.3767,  8.3979]])

L2 Norm:  tensor([[ 0.0000,  0.0000,  0.0000, 44.4158,  6.4391, 52.1156,  8.3708]])

L2 Norm:  tensor([[ 0.0000,  0.0000,  0.0000, 46.0802,  6.6939, 56.4616,  8.8441]])

L2 Norm:  tensor([[ 0.0000,  0.0000,  0.0000, 52.7074,  9.8519, 62.4894, 10.7782]])

L2 Norm:  tensor([[ 0.0000,  0.0000,  0.0000, 56.4258, 40.7219, 67.9755, 26.9553]])

L2 Norm:  tensor([[ 0.0000,  0.0000,  0.0000, 59.9862, 57.2415, 76.7896, 46.9606]])

L2 Norm:  tensor([[ 0.0000,  0.0000,  0.0000, 23.7267, 11.8551, 27.9338, 10.7161]])
```


### Observation:
<div align="justify">
While comparing outputs of every hidden layer for the sentences:

<br>Sentence1: The capital of France is Paris. 
<br>Sentence2: The capital of Germany is Berlin. 

The L2-norm of outputs between tokens of these sentences and it can be seen that for the first 3 tokens the L2-norm was 0, which seems correct since there are same tokens in both sentences and also in same order nd hence should produce same outputs and hence L2-norm is 0. Now from 4th token onwards the tokens in both sentence differ and so does the numbers in layers become more than 0 signifying some difference in both the sentence is captured in the numbers. 

Now across the layers I found the numbers to rise and drop, so basically the L2-norm for the 4th token till last token rise steadily across mid layers, whicle in later layers there is sharp increase and then there is sharp drop in last layer. 
</div>


### Inference
<div align="justify">
The reason/intuition(also guided by experimental measurements) is basically mid layer capture information representation hence every layer just adds some more representation, while later layer capture decision making for predicting next token this was understood by activation patching (ie. causal evidence, basically if you replace the later layer of model that correctly predicts with the later layers from a model that incorrectly predicts, we see decline in accuracy but if we replace the mid layers of model with correct prediction with mid layers from a model with incorrect predictions it does not effect accuracy so much). This is also guided by logit lens (decoding evidence) i.e. if we project mid layers through output head we dont have very accurate predictions but if we project late layers we get good accuracy in prediction, I believe activation patching and logit lense does not reflect what exactly mid layer is signifying but it says that atleast the later layers does something more related to decision making than mid layers do. 

So for now, its unclear about what information is mid layer signifying. 
Also the last layer just compresses outputs to remove irrelevant degrees of freedeom to project into task specific subspace. 
Now the task of LLM is to predict next token. Since the later layers were also working towards decision making and to effectively do that the model made outputs jump but the last layer, to project into task specific subspace, dropped the output. Now later layers and last layer seem to do similar thing but in exactly opposite way. So these ideas are bit conflicting and unclear.
</div>
