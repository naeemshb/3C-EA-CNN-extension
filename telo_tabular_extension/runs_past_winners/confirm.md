# Winners of the original runs: do they need their channels?

Screening on 12 splits, confirmation of the top 3 per dataset on 100 disjoint splits. Accuracy, mean over splits; p = two-sided Wilcoxon signed-rank, paired by split; Holm over all confirmed candidates.

## hepatitis

Screening: 46 distinct winners with both channels, 46 numerically valid, 13 at least as accurate as the best fixed activation, 17 more accurate with channels than without; median difference -0.0 points.

| formula | both | m only | c only | neither | best fixed | both − neither | p | Holm p | wins/losses | both − best fixed | p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `(min(x, (0 * c)) * (c - m))` | 0.655 | 0.660 | 0.659 | 0.663 | 0.667 (elu) | -0.8 | 0.0614 | 0.4910 | 22/32 | -1.2 | 0.1675 |
| `sin(min(min(c, x), (softplus(m))^3))` | 0.670 | 0.670 | 0.667 | 0.667 | 0.667 (elu) | +0.3 | 0.4104 | 1.0000 | 21/17 | +0.3 | 0.9609 |
| `(m + (c * x))` | 0.666 | 0.668 | 0.676 | 0.672 | 0.667 (elu) | -0.5 | 0.5638 | 1.0000 | 28/32 | -0.1 | 0.9152 |

## housevote

Screening: 36 distinct winners with both channels, 36 numerically valid, 36 at least as accurate as the best fixed activation, 14 more accurate with channels than without; median difference +0.0 points.

| formula | both | m only | c only | neither | best fixed | both − neither | p | Holm p | wins/losses | both − best fixed | p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `((x - m) + ((x - c) + max(c, (max(m, max(m, (sin(c))^2)))^2)))` | 0.934 | 0.934 | 0.933 | 0.933 | 0.886 (relu) | +0.1 | 0.9464 | 1.0000 | 23/23 | +4.8 | 0.0000 |
| `(((x + m) + c) + x)` | 0.931 | 0.932 | 0.931 | 0.932 | 0.886 (relu) | -0.0 | 0.7695 | 1.0000 | 24/29 | +4.5 | 0.0000 |
| `((min((x)^3, x) * exp(c)) + m)` | 0.935 | 0.935 | 0.935 | 0.934 | 0.886 (relu) | +0.1 | 0.6961 | 1.0000 | 39/26 | +4.9 | 0.0000 |

## horse

Screening: 20 distinct winners with both channels, 20 numerically valid, 13 at least as accurate as the best fixed activation, 11 more accurate with channels than without; median difference +0.2 points.

| formula | both | m only | c only | neither | best fixed | both − neither | p | Holm p | wins/losses | both − best fixed | p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `(((0.5 + (x + leakyrelu(max(0.5, x)))) * min(min(2, (m)^3), sin(m))) - (x / (-0.` | 0.634 | 0.632 | 0.633 | 0.637 | 0.614 (elu) | -0.3 | 0.5751 | 1.0000 | 39/46 | +2.0 | 0.0003 |
| `((x + (x + m)) / c)` | 0.625 | 0.624 | 0.635 | 0.631 | 0.614 (elu) | -0.6 | 0.1638 | 1.0000 | 36/39 | +1.1 | 0.0183 |
| `min(((m / c) + max(((c * c) * m), tanh(m))), x)` | 0.621 | 0.620 | 0.599 | 0.599 | 0.614 (elu) | +2.2 | 0.0007 | 0.0073 | 61/31 | +0.7 | 0.2165 |

## cylinderband

Screening: 36 distinct winners with both channels, 36 numerically valid, 26 at least as accurate as the best fixed activation, 17 more accurate with channels than without; median difference -0.0 points.

| formula | both | m only | c only | neither | best fixed | both − neither | p | Holm p | wins/losses | both − best fixed | p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `min(x, min(|m|, (c * ((((c / -0.1) - x) / -0.5) - sigmoid(m)))))` | 0.682 | 0.682 | 0.636 | 0.636 | 0.673 (elu) | +4.6 | 0.0000 | 0.0000 | 80/13 | +0.9 | 0.0425 |
| `sqrt(min((m + x), max(elu((2 / c)), max((c * m), 0.5))))` | 0.657 | 0.649 | 0.629 | 0.629 | 0.673 (elu) | +2.8 | 0.0000 | 0.0003 | 63/29 | -1.6 | 0.0261 |
| `(((max((-2 + max((-0.5 + m), 0.1)), m) - x) / (c - m)) / c)` | 0.681 | 0.674 | 0.672 | 0.667 | 0.673 (elu) | +1.4 | 0.0019 | 0.0169 | 60/31 | +0.8 | 0.0254 |

