---
title: "Programs for a Frozen Machine"
subtitle: "The description length of language and task in optimized prefixes"
author: "Georgii Sirotenko"
date: "Mechanistically Interpreting Multilingual LMs, WT 2025/26"
theme: "metropolis"
aspectratio: 169
fontsize: 10pt
---

## The idea

- A frozen LM is a fixed machine. A soft prefix that makes it do a task is a **program** for that machine.
- The shortest such program is a model‑relative **description length** of the task (Shaw et al. 2026; Cosma 2026).
- Grid: tasks {translation from English, factual recall, continuation} × output languages {ru, de, en}, Qwen2.5‑1.5B base, frozen.
- Questions: how many bits does a cell cost, how are they split between *task* and *language*, do the parts compose, how far apart are programs, where does the model read them.

## Two units, one distortion

- **Slots**: number $k$ of prefix positions (capacity).
- **Bits**: prefix = diagonal Gaussian posterior; its KL to a prior fitted to the embedding matrix is the code length (bits‑back; Hinton & van Camp 1993).
- Distortion: teacher‑forced bits per target token, normalised per cell
  $$\hat D = \frac{L - L_{\text{floor}}}{L_{\text{none}} - L_{\text{floor}}},\qquad L_{\text{floor}} = \min(\text{LoRA}, \text{best prefix})$$
- Preregistered operating points $\hat D \le 0.25$ and $\le 0.10$.
- 8 cells; ~32 A100 hours; every figure regenerated from CSVs.

## Result 1: slots are not a description length

![](../figures/curves.png)

- Translation: one slot reaches $\hat D \le 0.10$ in every language. Recall needs 2–16 and is still improving at $k = 32$.
- One trained slot beats the natural‑language prompt: recall NL prompt $\hat D$ 0.91–0.96, one slot 0.13–0.41.
- Telling the model the language or the task saves < 1 slot on average. Continuation: floor ≈ no prefix, unmeasurable.

## Result 2: description length in bits

| cell | bits at $\hat D \le 0.10$ | bits at $\hat D \le 0.25$ |
|---|---|---|
| translation‑de | 436 | 436 |
| translation‑ru | 542 | 542 |
| recall‑ru | 2,419 | 2,419 |
| recall‑de | — (best 0.14) | 3,343 |
| recall‑en | — (best 0.23) | 7,155 |

- A translation program ≈ 500 bits; Russian recall 5×; English recall dearest, also on the 4,212 items shared by all three recall test sets.
- Task ordering follows the learning gap (H1 ✓); language ordering does **not** follow exposure (H1 ✗).

## Result 3: task is the shared factor, language the cheap modifier

![](../figures/hier_bars.png)

| code for all 8 cells at $\hat D \le 0.25$ | bits |
|---|---|
| one centre per task + language deviations | 33,500 |
| eight independent programs | 37,600 |
| one centre per language + task deviations | 53,500 |

## What a language costs, what a task costs

- Given the **translation** centre: Russian 485 bits, German 662 bits. Under half a standalone program.
- Given the **recall** centre: a language costs 3,300–4,000 bits.
- Given a **language** centre: continuation 100–200 bits, translation 440–630, recall 5,300–26,200.
- Recall is the program that cannot be shared; the language‑centred code pays for it three times.
- For translation alone the hierarchy does not pay (2,733 vs 1,893 bits): the shared centre is dearer than either specialised program.

## Result 4: composition and distance

::: columns
:::: column
![](../figures/compose.png)

Zero‑shot assembly of task part + language part closes 60–73% of the gap on translation, 14–32% on recall.
::::
:::: column
![](../figures/distance.png)

Same‑task warm start costs 0.85 of scratch, same‑language 0.95. Silhouettes ≈ 0: direction only. Zero‑shot transfer = no prefix; recall prefixes break continuation.
::::
:::

## Result 5: where the program is read

![](../figures/mechanism.png)

- Patching the prefix positions into **block 0 alone** recovers 70–94% of the effect for translation and continuation, 36–58% for recall (secondary peaks at blocks 17, 25).
- A translation program is copied into the token stream in the first block; a recall program is read repeatedly, deep in the network.
- Language deviation vs task centre (H5): both peak at block 0; the depth profile splits by task, not by prefix part.

## Result 6: the objective is right, the search is hard

![](../figures/trajectories.png)

- lr 0.05 = 2 prior σ per step → KL overshoots by 20k (k=2) to 300k bits (k=8), then decays.
- Selecting the checkpoint by mean‑posterior distortion picks the high‑rate step; the mean is not the coded object (β=3: mean $\hat D$ 0.35–0.56, samples 0.06–0.17).
- Corrected protocol (lr 0.01, select on the objective, score samples): frontier **20–71% lower** with 4 runs per cell instead of 9.
- Shaw et al. (2026): 870 bits from a compiled init vs 2.8M from random init. Here the gap closes by selection and step size.

## Preregistered comparisons

| | outcome |
|---|---|
| H1 cost orderings | task ✓ (recall ≫ translation); language ✗ (English dearest) |
| H2 cheap language | slots: equal; bits: task given language 2–7× language given task on recall |
| H3 separability | partial on translation, not on recall |
| H4 task‑organised distance | direction ✓, structure ✗ (silhouette 0.004 vs −0.09) |
| H5 depth split | by task, not by prefix part |
| H6 input compressibility | one vector suffices with the task prefix (exploratory) |

## Limitations and next steps

- Two seeds; recall budget‑bound (3000 steps: $\hat D$ 0.043 → 0.032 and falling); continuation unmeasurable; distance matrix one seed.
- Variational protocol corrected after preregistration; both reported. Hierarchical rates are upper bounds (old selection rule).
- Research version, same code: scale axis 0.5B–7B, 20+ languages by resource level, per‑column mixture prior, charge $\log_2|k\text{-grid}|$ bits for $k$, step size in prior‑σ units.
- Repo: `sirotenkogeorgii/prefix-programs`, branch `prefixlab`; `01_figures.ipynb` regenerates everything in under a minute.
