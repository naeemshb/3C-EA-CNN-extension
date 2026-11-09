# Step 3 — Fashion-MNIST, random 20 %, replicate 0 (reference search budget, cheap fitness protocol)

Single replicate; engineering decision gate, not a reportable result.

## Matched conditions (trained and tested at random 20 %)

| method | activation | test acc | macro-F1 | best epoch | search cost |
|---|---|---|---|---|---|
| ReLU | `max(x, 0)` | 0.8861 | 0.8856 | 64 | 0.00 h |
| Swish | `(x * sigmoid(x))` | 0.8778 | 0.8767 | 40 | 0.00 h |
| LeakyReLU | `max(x, (0.1 * x))` | 0.8871 | 0.8870 | 64 | 0.00 h |
| ELU | `elu(x)` | 0.8753 | 0.8748 | 36 | 0.00 h |
| evolved f(z,m,c) | `sin(((x + x) / min(c, 0.5)))` | 0.8700 | 0.8705 | 10 | 3.91 h |

## Changing test conditions (frozen model and preprocessing; test accuracy)

| method | random 0% | random 40% | random 60% | block 20% | block 40% | block 60% |
|---|---|---|---|---|---|---|
| ReLU | 0.8847 | 0.8452 | 0.6793 | 0.7465 | 0.5110 | 0.2759 |
| Swish | 0.8849 | 0.8373 | 0.6678 | 0.7653 | 0.5810 | 0.3159 |
| LeakyReLU | 0.8867 | 0.8423 | 0.6628 | 0.7562 | 0.5617 | 0.2912 |
| ELU | 0.8848 | 0.8197 | 0.6257 | 0.7727 | 0.5719 | 0.3097 |
| evolved f(z,m,c) | 0.8653 | 0.8283 | 0.5758 | 0.7153 | 0.4926 | 0.2927 |

## Diagnosis

1. **The cheap fitness protocol ranks fast learners, not final accuracy.** Same data, masks, subset and seeds as the search:

| tree | cheap val (5 ep, 10k) | full val (<=30 ep, all data) |
|---|---|---|
| `max(x, 0)` | 0.7265 | 0.8785 (20 ep) |
| `(x * sigmoid(x))` | 0.7360 | 0.8745 (15 ep) |
| `max(x, (0.1 * x))` | 0.7248 | 0.8743 (14 ep) |
| `elu(x)` | 0.7402 | 0.8712 (13 ep) |
| `sin(((x + x) / min(c, 0.5)))` | 0.8240 | 0.8668 (8 ep) |

   ReLU needs ~15-20 full-data epochs to overtake the periodic winner; under every shorter protocol tried the winner is ahead:

| fitness protocol | ReLU val | ELU val | winner val | ReLU per-tree cost |
|---|---|---|---|---|
| sub10k_e5_p2_b256 (current cheap) | 0.7265 | 0.7402 | 0.8240 | 3 s |
| sub10k_e10_p3_b128 | 0.7903 | 0.7945 | 0.8458 | 6 s |
| sub10k_e15_p3_b128 | 0.8043 | 0.8210 | 0.8470 | 8 s |
| sub20k_e10_p3_b128 | 0.8227 | 0.8322 | 0.8605 | 10 s |
| sub20k_e15_p3_b128 | 0.8415 | 0.8435 | 0.8605 | 15 s |
| full_e10_p3_b128 | 0.8623 | 0.8708 | 0.8668 | 26 s |

2. **The confidence channel is dead in the winner.** Confidence is fitted on training data at rate 0.2, so c is 1.0 (observed) or 0.8 (missing) and propagated c never drops below 0.8; `min(c, 0.5)` is always 0.5 and the winner is exactly `sin(4x)`. The GP's use of c here is a constant, not a mechanism.

3. **Better formulas existed in the final population and were not selected.** Top-20 distinct formulas of the final population re-scored under the full protocol (reference ReLU full val 0.8785):

| cheap val | full val | formula |
|---|---|---|
| 0.7805 | 0.8833 (11 ep) | `((2 * (x * min(x, c))) * ((c)^3 - x))` |
| 0.7810 | 0.8827 (12 ep) | `(relu(|min(max(x, ((x)^3 * ((c / 2) - x))), x)|) + x)` |
| 0.7817 | 0.8760 (14 ep) | `(relu(sin(x)) + x)` |
| 0.7940 | 0.8755 (12 ep) | `(x + (m - sin(cos((x + x)))))` |
| 0.7737 | 0.8753 (12 ep) | `(leakyrelu(x) - (m + (max((x - x), min(relu(m), x)) - min(c, (c * (x * max((c - max(exp(log(|x|)), (0.5 / x))), (x + -0.1))))))))` |
| 0.8140 | 0.8677 (7 ep) | `sin((sin(((x + x) / (leakyrelu(c) + 0.5))) / min(c, max((tanh(x) * x), 0.5))))` |
| 0.8007 | 0.8673 (8 ep) | `sin(((x + cos((((m - 0.5) + c) * x))) / (c - 0.5)))` |
| 0.8320 | 0.8668 (8 ep) | `sin(((x + x) / min(c, 0.5)))` |
| 0.7743 | 0.8647 (8 ep) | `(leakyrelu(x) - (m + (max((x - x), min(relu(m), x)) - min(c, (c * (x * max((m - max(exp(log(|x|)), (x - m))), (x + -0.1))))))))` |
| 0.7770 | 0.8633 (11 ep) | `(m - (exp(x) + (x + m)))` |
| 0.7760 | 0.8543 (13 ep) | `(leakyrelu(x) - (m + (max((x - x), min(relu(m), x)) - min(c, x))))` |
| 0.7797 | 0.8247 (9 ep) | `((m + min(max(x, ((x)^3 * ((c / 0.5) * x))), x)) + (-0.1 + x))` |
| 0.8225 | 0.0000 (1 ep) | `sin(((x + x) / min(elu((leakyrelu(x) - (sqrt(c) - (sqrt((((tanh(x) * -0.5) - x) + x)) - x)))), (x - sqrt(max(x, (x + x)))))))` |
| 0.8140 | 0.0000 (1 ep) | `(x - (|(sqrt(x) - (c - (c * (x * max((c - max(exp(log(|x|)), (x - m))), (x + -0.1))))))| - x))` |
| 0.8000 | 0.0000 (1 ep) | `(|(sqrt(x) - (c - x))| * x)` |
| 0.7933 | 0.0000 (1 ep) | `sin(((x + cos((sqrt(x) - x))) / min(c, max((x * x), 0.5))))` |
| 0.7800 | 0.0000 (1 ep) | `(m - (|(sqrt(((x + leakyrelu(x)) / min(x, 0.5))) - (c - x))| * x))` |
| 0.7780 | 0.0000 (1 ep) | `(elu((leakyrelu(x) - (sqrt(c) - (sqrt((((tanh(x) * -0.5) - x) + x)) - x)))) - (x + (m + x)))` |
| 0.7750 | 0.0000 (1 ep) | `(c * min(sin(max(sqrt((x + c)), c)), ((x)^2 - x)))` |
| 0.7700 | 0.0000 (1 ep) | `(x - (|(sqrt(x) - (c - x))| - m))` |

   (full val 0.0000 = training failed under the full protocol although it was valid under the cheap one.)

4. Search budget: 3,000 evaluations in 3.9 h (mean 4.7 s per candidate, rising from 2 s to 8 s per candidate as trees deepened); 457 x-free candidates cost 0.15 h, so skipping them would save little here.

5. **Re-ranked candidates, final-trained and tested (diagnostic; same final protocol and seeds as the committed runs).** Top-3 of the final population by full-protocol validation:

| formula | cheap val | full val | test acc | macro-F1 | best epoch |
|---|---|---|---|---|---|
| `((2 * (x * min(x, c))) * ((c)^3 - x))` | 0.7805 | 0.8833 | 0.8830 | 0.8826 | 22 |
| `(relu(|min(max(x, ((x)^3 * ((c / 2) - x))), x)|) + x)` | 0.7810 | 0.8827 | 0.8857 | 0.8858 | 66 |
| `(relu(sin(x)) + x)` | 0.7817 | 0.8760 | 0.8791 | 0.8781 | 49 |

   Against ReLU 0.8861 / LeakyReLU 0.8871 these are ties within single-seed noise, not wins. Re-ranking recovers the baseline level that the cheap-protocol selection lost (0.870 -> 0.886); it does not establish an advantage.

## Reading of the gate

Outcome (c) of the plan: the evolved activation trails every fixed baseline under matched and shifted conditions. Two mechanisms, both artefacts of transplanting the tabular protocol unchanged: (i) the short fitness budget selects fast-plateauing activations (periodic `sin(4x)`) that ReLU overtakes after ~15-20 full-data epochs; (ii) with uniform random masks at a single rate the confidence channel is a constant (0.8 on missing pixels) and carries nothing beyond the mask. Next decision: fix the selection protocol (full-protocol re-ranking of the top-k, or a longer fitness budget) and/or the confidence definition before any replicate or dataset grid.

Diagnostic scripts and JSON: `runs/diagnostics_step3/`.

## Step 3b (2026-11-09) — selection rule changed to two-stage re-ranking (deviation D12), same search

The saved step-3 search (3,000 evaluations) was re-used; only the selection stage and the final training were re-run.
k = 20 was chosen after the diagnostic above (which had shown long-protocol winners at search ranks 11-12 and their test scores); replicate 0 is therefore not a clean test of the rule.

| selection | formula | long-protocol val | test acc | macro-F1 | extra cost |
|---|---|---|---|---|---|
| evolved, search-best (original rule) | `sin(((x + x) / min(c, 0.5)))` | – (not re-scored) | 0.8700 | 0.8705 | 0 min |
| evolved, two-stage k=10 | `(x + (m - sin(cos((x + x)))))` | 0.8755 | 0.8666 | 0.8659 | 6 min |
| evolved, two-stage k=20 (reference configs) | `((2 * (x * min(x, c))) * ((c)^3 - x))` | 0.8833 | 0.8830 | 0.8826 | 17 min |

ReLU 0.8861 / LeakyReLU 0.8871 for comparison: k = 20 ties the best fixed activation; k = 10 (pool without ranks 11-12) picked an additive-mask formula that is worse than the original rule.

### Changing test conditions, all seven models (frozen; test accuracy)

| model | random 0% | random 40% | random 60% | block 20% | block 40% | block 60% |
|---|---|---|---|---|---|---|
| ReLU | 0.8847 | 0.8452 | 0.6793 | 0.7465 | 0.5110 | 0.2759 |
| Swish | 0.8849 | 0.8373 | 0.6678 | 0.7653 | 0.5810 | 0.3159 |
| LeakyReLU | 0.8867 | 0.8423 | 0.6628 | 0.7562 | 0.5617 | 0.2912 |
| ELU | 0.8848 | 0.8197 | 0.6257 | 0.7727 | 0.5719 | 0.3097 |
| evolved, search-best (original rule) | 0.8653 | 0.8283 | 0.5758 | 0.7153 | 0.4926 | 0.2927 |
| evolved, two-stage k=10 | 0.8420 | 0.8036 | 0.4781 | 0.6573 | 0.3430 | 0.1334 |
| evolved, two-stage k=20 (reference configs) | 0.8839 | 0.8365 | 0.6943 | 0.8129 | 0.6355 | 0.3510 |

Reading (single replicate): the k=20 formula `2·x·min(x,c)·(c³−x)` is the most robust model in the table — best of all seven at random 60 % and at every block rate (block 20 %: 0.813 vs 0.773 for the best fixed activation), and it loses nothing on clean images. The k=10 formula, whose mask term enters additively (`x + m − …`), is the least robust: 0.842 on clean images (below its own matched score) and 0.133 at block 60 %. Mechanism hypothesis, to be tested with replicates and ablations: multiplicative, bounded use of the propagated confidence adapts to the local missing fraction; an additive mask term shifts every pre-activation when the rate changes.

## Step 3d (2026-11-09) — reference rule fixed: re-score EVERY distinct valid finalist (no pool size)

Pool: 90 distinct valid finalists of the step-3 search; 60 survive the long protocol (30 fail with non-finite loss); re-scoring cost 73 min (31 % of the search).

| long-protocol rank | search rank | search val | long val | formula |
|---|---|---|---|---|
| 1 | 11 | 0.7805 | 0.8833 | `((2 * (x * min(x, c))) * ((c)^3 - x))` |
| 2 | 12 | 0.7810 | 0.8827 | `(relu(|min(max(x, ((x)^3 * ((c / 2) - x))), x)|) + x)` |
| 3 | 26 | 0.7613 | 0.8797 | `((x - min(relu(x), m)) - (x + (x + (c - max(exp(x), (c + m))` |
| 4 | 38 | 0.7533 | 0.8785 | `max((x * (x - x)), x)` |
| 5 | 42 | 0.7507 | 0.8785 | `max((max(tanh(x), (x - x)) * min(x, x)), x)` |
| 6 | 68 | 0.6663 | 0.8765 | `((min(c, 0.5) * (x * min(relu(c), c))) * ((c)^3 - (x * x)))` |
| 7 | 35 | 0.7578 | 0.8763 | `((x + x) / (0.5 / ((x / 2) * x)))` |
| 8 | 13 | 0.7817 | 0.8760 | `(relu(sin(x)) + x)` |

Selected: `((2 * (x * min(x, c))) * ((c)^3 - x))` (search rank 11) — the same formula as k = 20, hence identical test (0.8830) and shift numbers (random_0pct 0.884, random_40pct 0.837, random_60pct 0.694, block_20pct 0.813, block_40pct 0.635, block_60pct 0.351). Note that ranks 38 and 42 are algebraic ReLUs (`max(x·0, x)`), scoring 0.8785 = the fixed ReLU baseline's long-protocol value: the search does produce ReLU and the long protocol ranks the selected formula 0.005 above it on validation — a single-seed margin.
