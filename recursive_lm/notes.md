So far we tested the recursive MiniLM idea in stages:
- Layer inspection / sensitivity
  - MiniLM has 6 layers, hidden size 384.
  - We skipped one layer at a time and measured how much the final embedding changed.
  - Average change across 3 sentences was roughly:
    - L0: 0.265
    - L1: 0.069
    - L2: 0.073
    - L3: 0.081
    - L4: 0.191
    - L5: 0.554
  - This suggested L5, L0, and L4 had the largest effect on the final representation.
- Frozen full-stack recursion
  - We reused the whole MiniLM stack repeatedly with no training.
  - On 50 CommonsenseQA examples:
T=1: accuracy=0.500, avg_margin=-0.0024
T=2: accuracy=0.420, avg_margin=-0.0253
T=3: accuracy=0.200, avg_margin=-0.0516
T=4: accuracy=0.200, avg_margin=-0.0568

- Conclusion: naive frozen recursion severely hurts.
- Frozen L4-L5 refinement recursion
  - Normal MiniLM pass first, then repeatedly applied only L4→L5.
  - Results on the same 50 CommonsenseQA examples:
T=0: accuracy=0.500, avg_margin=-0.0024
T=1: accuracy=0.400, avg_margin=-0.0101
T=2: accuracy=0.360, avg_margin=-0.0384
T=3: accuracy=0.240, avg_margin=-0.0496
T=4: accuracy=0.200, avg_margin=-0.0520

- Again, frozen recurrence degraded performance, likely due to representation drift / attractor behavior.
- Built trainable RecursiveMiniLM
  - Frozen:
    - embeddings
    - L0-L3
  - Trainable recursive block:
    - L4
    - L5
  - Recurrence:
    - T=1: L4→L5 once
    - T=2: L4→L5→L4→L5
    - etc.
  - Mean pooling + L2 normalization at the end.
- Structural tests
  - Output shape was correct:
    - [1, 384]
  - Norm was 1.0 after normalization.
  - Only L4/L5 received gradients.
  - Different recursion depths produced substantially different vectors:
cos(T=1, T=2) = 0.5119
cos(T=2, T=3) = 0.6846
cos(T=1, T=3) = 0.2850

- So recurrence is genuinely changing the latent representation.
- Frozen candidate encoder
  - Candidate labels are encoded using normal frozen MiniLM.
  - Query comes from recursive MiniLM.
  - Score is cosine similarity via q @ k.T.
- Single-step training sanity check
  - Question: “What gas do plants absorb from the atmosphere?”
  - Correct answer: carbon dioxide.
  - Before one optimizer step:
[oxygen, carbon dioxide, nitrogen, hydrogen]
[0.1087, 0.2119, 0.0962, 0.1675]

- After one step:
[0.1932, 0.3326, 0.1627, 0.1966]

- Correct-answer margin improved from about 0.044 to 0.136.
- Training on 100 random AG News examples
  - Random recursion depth during training: T ∈ {1,2,3}.
  - 5 epochs:
Epoch 1: loss=1.3230, accuracy=0.560, avg_margin=0.0246
Epoch 2: loss=1.1798, accuracy=0.760, avg_margin=0.1869
Epoch 3: loss=1.1080, accuracy=0.840, avg_margin=0.2651
Epoch 4: loss=1.0422, accuracy=0.870, avg_margin=0.3725
Epoch 5: loss=0.9973, accuracy=0.910, avg_margin=0.4457

- Evaluation on 100 unseen AG News examples
T=1: accuracy=0.870, avg_margin=0.2757
T=2: accuracy=0.880, avg_margin=0.4555
T=3: accuracy=0.840, avg_margin=0.4978
T=4: accuracy=0.850, avg_margin=0.4844

Main takeaway so far:
Frozen recursion fails badly, but after fine-tuning the shared L4-L5 block, extra recursive computation becomes useful. On unseen AG News, T=2 slightly outperformed T=1, while deeper recursion increased score separation but did not improve accuracy further.

The next logical step is what you suggested: train/evaluate across all four datasets rather than treating AG News alone as the main result.