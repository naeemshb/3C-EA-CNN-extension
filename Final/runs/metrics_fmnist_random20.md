# Extended test metrics (from saved predictions; one replicate)

## matched_random_20pct

| model | activation | Acc | Bal. acc | P (macro) | R (macro) | F1 (macro) | F1 (wtd) | AUC (OvR) | AP (macro) | Log loss | Brier | ECE | Kappa | MCC | Top-2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref_fashion_mnist_random20_cnn_elu | `elu(x)` | 0.8753 | 0.8753 | 0.8748 | 0.8753 | 0.8748 | 0.8748 | 0.9888 | 0.9251 | 0.3919 | 0.1854 | 0.0413 | 0.8614 | 0.8615 | 0.9644 |
| ref_fashion_mnist_random20_cnn_evolved_zmc | `sin(((x + x) / min(c, 0.5)))` | 0.8700 | 0.8700 | 0.8714 | 0.8700 | 0.8705 | 0.8705 | 0.9876 | 0.9233 | 0.3946 | 0.1924 | 0.0343 | 0.8556 | 0.8556 | 0.9554 |
| ref_fashion_mnist_random20_cnn_evolved_zmc_rr10 | `(x + (m - sin(cos((x + x)))))` | 0.8666 | 0.8666 | 0.8667 | 0.8666 | 0.8659 | 0.8659 | 0.9884 | 0.9230 | 0.4086 | 0.1967 | 0.0566 | 0.8518 | 0.8520 | 0.9611 |
| ref_fashion_mnist_random20_cnn_evolved_zmc_rr20 | `((2 * (x * min(x, c))) * ((c)^3 - ` | 0.8830 | 0.8830 | 0.8828 | 0.8830 | 0.8826 | 0.8826 | 0.9876 | 0.9209 | 0.5551 | 0.1848 | 0.0590 | 0.8700 | 0.8701 | 0.9648 |
| ref_fashion_mnist_random20_cnn_leakyrelu | `max(x, (0.1 * x))` | 0.8871 | 0.8871 | 0.8869 | 0.8871 | 0.8870 | 0.8870 | 0.9910 | 0.9386 | 0.3325 | 0.1671 | 0.0372 | 0.8746 | 0.8746 | 0.9706 |
| ref_fashion_mnist_random20_cnn_relu | `max(x, 0)` | 0.8861 | 0.8861 | 0.8855 | 0.8861 | 0.8856 | 0.8856 | 0.9906 | 0.9364 | 0.3395 | 0.1683 | 0.0374 | 0.8734 | 0.8735 | 0.9683 |
| ref_fashion_mnist_random20_cnn_swish | `(x * sigmoid(x))` | 0.8778 | 0.8778 | 0.8766 | 0.8778 | 0.8767 | 0.8767 | 0.9891 | 0.9274 | 0.3930 | 0.1816 | 0.0463 | 0.8642 | 0.8643 | 0.9654 |

## shift_block_20pct

| model | activation | Acc | Bal. acc | P (macro) | R (macro) | F1 (macro) | F1 (wtd) | AUC (OvR) | AP (macro) | Log loss | Brier | ECE | Kappa | MCC | Top-2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref_fashion_mnist_random20_cnn_elu | `elu(x)` | 0.7727 | 0.7727 | 0.7784 | 0.7727 | 0.7718 | 0.7718 | 0.9748 | 0.8464 | 0.8508 | 0.3495 | 0.1214 | 0.7474 | 0.7483 | 0.9158 |
| ref_fashion_mnist_random20_cnn_evolved_zmc | `sin(((x + x) / min(c, 0.5)))` | 0.7153 | 0.7153 | 0.7282 | 0.7153 | 0.7155 | 0.7155 | 0.9563 | 0.7910 | 0.8941 | 0.4093 | 0.0582 | 0.6837 | 0.6853 | 0.8651 |
| ref_fashion_mnist_random20_cnn_evolved_zmc_rr10 | `(x + (m - sin(cos((x + x)))))` | 0.6573 | 0.6573 | 0.6767 | 0.6573 | 0.6512 | 0.6512 | 0.9457 | 0.7215 | 1.4656 | 0.5444 | 0.2146 | 0.6192 | 0.6231 | 0.8398 |
| ref_fashion_mnist_random20_cnn_evolved_zmc_rr20 | `((2 * (x * min(x, c))) * ((c)^3 - ` | 0.8129 | 0.8129 | 0.8126 | 0.8129 | 0.8113 | 0.8113 | 0.9743 | 0.8573 | 1.4464 | 0.3169 | 0.1367 | 0.7921 | 0.7925 | 0.9328 |
| ref_fashion_mnist_random20_cnn_leakyrelu | `max(x, (0.1 * x))` | 0.7562 | 0.7562 | 0.7662 | 0.7562 | 0.7502 | 0.7502 | 0.9735 | 0.8433 | 0.9139 | 0.3732 | 0.1368 | 0.7291 | 0.7315 | 0.9063 |
| ref_fashion_mnist_random20_cnn_relu | `max(x, 0)` | 0.7465 | 0.7465 | 0.7606 | 0.7465 | 0.7434 | 0.7434 | 0.9716 | 0.8322 | 0.9786 | 0.3968 | 0.1417 | 0.7183 | 0.7208 | 0.9037 |
| ref_fashion_mnist_random20_cnn_swish | `(x * sigmoid(x))` | 0.7653 | 0.7653 | 0.7672 | 0.7653 | 0.7609 | 0.7609 | 0.9728 | 0.8353 | 0.9300 | 0.3680 | 0.1393 | 0.7392 | 0.7405 | 0.9129 |

## shift_block_40pct

| model | activation | Acc | Bal. acc | P (macro) | R (macro) | F1 (macro) | F1 (wtd) | AUC (OvR) | AP (macro) | Log loss | Brier | ECE | Kappa | MCC | Top-2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref_fashion_mnist_random20_cnn_elu | `elu(x)` | 0.5719 | 0.5719 | 0.6302 | 0.5719 | 0.5602 | 0.5602 | 0.9347 | 0.6675 | 1.8940 | 0.6519 | 0.2425 | 0.5243 | 0.5308 | 0.7618 |
| ref_fashion_mnist_random20_cnn_evolved_zmc | `sin(((x + x) / min(c, 0.5)))` | 0.4926 | 0.4926 | 0.5674 | 0.4926 | 0.4755 | 0.4755 | 0.8923 | 0.5807 | 1.7554 | 0.7222 | 0.2287 | 0.4362 | 0.4470 | 0.6856 |
| ref_fashion_mnist_random20_cnn_evolved_zmc_rr10 | `(x + (m - sin(cos((x + x)))))` | 0.3430 | 0.3430 | 0.5853 | 0.3430 | 0.3078 | 0.3078 | 0.8662 | 0.4846 | 4.3951 | 1.1154 | 0.5146 | 0.2700 | 0.3014 | 0.5993 |
| ref_fashion_mnist_random20_cnn_evolved_zmc_rr20 | `((2 * (x * min(x, c))) * ((c)^3 - ` | 0.6355 | 0.6355 | 0.6677 | 0.6355 | 0.6341 | 0.6341 | 0.9323 | 0.7021 | 1.7174 | 0.5528 | 0.1908 | 0.5950 | 0.5979 | 0.8229 |
| ref_fashion_mnist_random20_cnn_leakyrelu | `max(x, (0.1 * x))` | 0.5617 | 0.5617 | 0.6087 | 0.5617 | 0.5328 | 0.5328 | 0.9275 | 0.6567 | 2.0869 | 0.6768 | 0.2614 | 0.5130 | 0.5229 | 0.7449 |
| ref_fashion_mnist_random20_cnn_relu | `max(x, 0)` | 0.5110 | 0.5110 | 0.6041 | 0.5110 | 0.4844 | 0.4844 | 0.9139 | 0.6158 | 2.6899 | 0.7735 | 0.3136 | 0.4567 | 0.4713 | 0.6915 |
| ref_fashion_mnist_random20_cnn_swish | `(x * sigmoid(x))` | 0.5810 | 0.5810 | 0.6219 | 0.5810 | 0.5675 | 0.5675 | 0.9319 | 0.6633 | 1.8928 | 0.6520 | 0.2555 | 0.5344 | 0.5410 | 0.7815 |

## shift_block_60pct

| model | activation | Acc | Bal. acc | P (macro) | R (macro) | F1 (macro) | F1 (wtd) | AUC (OvR) | AP (macro) | Log loss | Brier | ECE | Kappa | MCC | Top-2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref_fashion_mnist_random20_cnn_elu | `elu(x)` | 0.3097 | 0.3097 | 0.4463 | 0.3097 | 0.2674 | 0.2674 | 0.8173 | 0.3748 | 3.9850 | 1.0526 | 0.4227 | 0.2330 | 0.2491 | 0.4672 |
| ref_fashion_mnist_random20_cnn_evolved_zmc | `sin(((x + x) / min(c, 0.5)))` | 0.2927 | 0.2927 | 0.4606 | 0.2927 | 0.2242 | 0.2242 | 0.7505 | 0.3210 | 3.3486 | 1.0646 | 0.4166 | 0.2141 | 0.2356 | 0.4447 |
| ref_fashion_mnist_random20_cnn_evolved_zmc_rr10 | `(x + (m - sin(cos((x + x)))))` | 0.1334 | 0.1334 | 0.3260 | 0.1334 | 0.0667 | 0.0667 | 0.7195 | 0.2345 | 11.6143 | 1.6667 | 0.8213 | 0.0371 | 0.0803 | 0.3217 |
| ref_fashion_mnist_random20_cnn_evolved_zmc_rr20 | `((2 * (x * min(x, c))) * ((c)^3 - ` | 0.3510 | 0.3510 | 0.4780 | 0.3510 | 0.3307 | 0.3307 | 0.7972 | 0.3896 | 2.1826 | 0.8237 | 0.1329 | 0.2789 | 0.2907 | 0.5347 |
| ref_fashion_mnist_random20_cnn_leakyrelu | `max(x, (0.1 * x))` | 0.2912 | 0.2912 | 0.4371 | 0.2912 | 0.2453 | 0.2453 | 0.8282 | 0.4105 | 4.1811 | 1.1159 | 0.4872 | 0.2124 | 0.2408 | 0.4816 |
| ref_fashion_mnist_random20_cnn_relu | `max(x, 0)` | 0.2759 | 0.2759 | 0.3948 | 0.2759 | 0.1892 | 0.1892 | 0.7989 | 0.3754 | 5.8249 | 1.1671 | 0.5097 | 0.1954 | 0.2187 | 0.4011 |
| ref_fashion_mnist_random20_cnn_swish | `(x * sigmoid(x))` | 0.3159 | 0.3159 | 0.4267 | 0.3159 | 0.2651 | 0.2651 | 0.8154 | 0.3709 | 4.1799 | 1.0585 | 0.4381 | 0.2399 | 0.2554 | 0.4788 |

## shift_random_0pct

| model | activation | Acc | Bal. acc | P (macro) | R (macro) | F1 (macro) | F1 (wtd) | AUC (OvR) | AP (macro) | Log loss | Brier | ECE | Kappa | MCC | Top-2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref_fashion_mnist_random20_cnn_elu | `elu(x)` | 0.8848 | 0.8848 | 0.8847 | 0.8848 | 0.8832 | 0.8832 | 0.9905 | 0.9366 | 0.4096 | 0.1768 | 0.0530 | 0.8720 | 0.8724 | 0.9676 |
| ref_fashion_mnist_random20_cnn_evolved_zmc | `sin(((x + x) / min(c, 0.5)))` | 0.8653 | 0.8653 | 0.8652 | 0.8653 | 0.8643 | 0.8643 | 0.9858 | 0.9199 | 0.4151 | 0.1995 | 0.0192 | 0.8503 | 0.8506 | 0.9554 |
| ref_fashion_mnist_random20_cnn_evolved_zmc_rr10 | `(x + (m - sin(cos((x + x)))))` | 0.8420 | 0.8420 | 0.8545 | 0.8420 | 0.8440 | 0.8440 | 0.9856 | 0.9075 | 0.5321 | 0.2421 | 0.0758 | 0.8244 | 0.8255 | 0.9453 |
| ref_fashion_mnist_random20_cnn_evolved_zmc_rr20 | `((2 * (x * min(x, c))) * ((c)^3 - ` | 0.8839 | 0.8839 | 0.8836 | 0.8839 | 0.8829 | 0.8829 | 0.9866 | 0.9204 | 1.2718 | 0.2072 | 0.0965 | 0.8710 | 0.8712 | 0.9647 |
| ref_fashion_mnist_random20_cnn_leakyrelu | `max(x, (0.1 * x))` | 0.8867 | 0.8867 | 0.8894 | 0.8867 | 0.8849 | 0.8849 | 0.9919 | 0.9460 | 0.3717 | 0.1713 | 0.0559 | 0.8741 | 0.8749 | 0.9700 |
| ref_fashion_mnist_random20_cnn_relu | `max(x, 0)` | 0.8847 | 0.8847 | 0.8878 | 0.8847 | 0.8821 | 0.8821 | 0.9913 | 0.9419 | 0.3898 | 0.1788 | 0.0545 | 0.8719 | 0.8729 | 0.9686 |
| ref_fashion_mnist_random20_cnn_swish | `(x * sigmoid(x))` | 0.8849 | 0.8849 | 0.8845 | 0.8849 | 0.8830 | 0.8830 | 0.9905 | 0.9377 | 0.4118 | 0.1775 | 0.0583 | 0.8721 | 0.8725 | 0.9692 |

## shift_random_40pct

| model | activation | Acc | Bal. acc | P (macro) | R (macro) | F1 (macro) | F1 (wtd) | AUC (OvR) | AP (macro) | Log loss | Brier | ECE | Kappa | MCC | Top-2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref_fashion_mnist_random20_cnn_elu | `elu(x)` | 0.8197 | 0.8197 | 0.8265 | 0.8197 | 0.8216 | 0.8216 | 0.9815 | 0.8846 | 0.5230 | 0.2537 | 0.0477 | 0.7997 | 0.8000 | 0.9412 |
| ref_fashion_mnist_random20_cnn_evolved_zmc | `sin(((x + x) / min(c, 0.5)))` | 0.8283 | 0.8283 | 0.8381 | 0.8283 | 0.8309 | 0.8309 | 0.9813 | 0.8889 | 0.5083 | 0.2508 | 0.0379 | 0.8092 | 0.8099 | 0.9388 |
| ref_fashion_mnist_random20_cnn_evolved_zmc_rr10 | `(x + (m - sin(cos((x + x)))))` | 0.8036 | 0.8036 | 0.8144 | 0.8036 | 0.7993 | 0.7993 | 0.9804 | 0.8757 | 0.6281 | 0.2933 | 0.0836 | 0.7818 | 0.7843 | 0.9285 |
| ref_fashion_mnist_random20_cnn_evolved_zmc_rr20 | `((2 * (x * min(x, c))) * ((c)^3 - ` | 0.8365 | 0.8365 | 0.8424 | 0.8365 | 0.8366 | 0.8366 | 0.9809 | 0.8824 | 0.4962 | 0.2389 | 0.0324 | 0.8183 | 0.8190 | 0.9478 |
| ref_fashion_mnist_random20_cnn_leakyrelu | `max(x, (0.1 * x))` | 0.8423 | 0.8423 | 0.8497 | 0.8423 | 0.8443 | 0.8443 | 0.9856 | 0.9070 | 0.4349 | 0.2241 | 0.0376 | 0.8248 | 0.8252 | 0.9526 |
| ref_fashion_mnist_random20_cnn_relu | `max(x, 0)` | 0.8452 | 0.8452 | 0.8543 | 0.8452 | 0.8475 | 0.8475 | 0.9850 | 0.9018 | 0.4430 | 0.2253 | 0.0372 | 0.8280 | 0.8286 | 0.9539 |
| ref_fashion_mnist_random20_cnn_swish | `(x * sigmoid(x))` | 0.8373 | 0.8373 | 0.8401 | 0.8373 | 0.8381 | 0.8381 | 0.9835 | 0.8949 | 0.4865 | 0.2364 | 0.0466 | 0.8192 | 0.8193 | 0.9484 |

## shift_random_60pct

| model | activation | Acc | Bal. acc | P (macro) | R (macro) | F1 (macro) | F1 (wtd) | AUC (OvR) | AP (macro) | Log loss | Brier | ECE | Kappa | MCC | Top-2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref_fashion_mnist_random20_cnn_elu | `elu(x)` | 0.6257 | 0.6257 | 0.7079 | 0.6257 | 0.6283 | 0.6283 | 0.9483 | 0.7322 | 1.2371 | 0.5180 | 0.1181 | 0.5841 | 0.5916 | 0.7996 |
| ref_fashion_mnist_random20_cnn_evolved_zmc | `sin(((x + x) / min(c, 0.5)))` | 0.5758 | 0.5758 | 0.6791 | 0.5758 | 0.5809 | 0.5809 | 0.9317 | 0.7105 | 1.3119 | 0.5717 | 0.1134 | 0.5287 | 0.5376 | 0.7586 |
| ref_fashion_mnist_random20_cnn_evolved_zmc_rr10 | `(x + (m - sin(cos((x + x)))))` | 0.4781 | 0.4781 | 0.6408 | 0.4781 | 0.4463 | 0.4463 | 0.9283 | 0.6601 | 2.3500 | 0.8014 | 0.3159 | 0.4201 | 0.4454 | 0.6956 |
| ref_fashion_mnist_random20_cnn_evolved_zmc_rr20 | `((2 * (x * min(x, c))) * ((c)^3 - ` | 0.6943 | 0.6943 | 0.7433 | 0.6943 | 0.6947 | 0.6947 | 0.9535 | 0.7573 | 1.0710 | 0.4957 | 0.2205 | 0.6603 | 0.6651 | 0.8570 |
| ref_fashion_mnist_random20_cnn_leakyrelu | `max(x, (0.1 * x))` | 0.6628 | 0.6628 | 0.7305 | 0.6628 | 0.6675 | 0.6675 | 0.9558 | 0.7739 | 1.0260 | 0.4693 | 0.0868 | 0.6253 | 0.6303 | 0.8354 |
| ref_fashion_mnist_random20_cnn_relu | `max(x, 0)` | 0.6793 | 0.6793 | 0.7423 | 0.6793 | 0.6928 | 0.6928 | 0.9565 | 0.7742 | 0.9580 | 0.4486 | 0.0865 | 0.6437 | 0.6479 | 0.8615 |
| ref_fashion_mnist_random20_cnn_swish | `(x * sigmoid(x))` | 0.6678 | 0.6678 | 0.7320 | 0.6678 | 0.6741 | 0.6741 | 0.9540 | 0.7596 | 1.0667 | 0.4680 | 0.0955 | 0.6309 | 0.6359 | 0.8360 |

Acc = accuracy; Bal. acc = balanced accuracy (mean per-class recall); P/R/F1 macro = unweighted class means; F1 (wtd) = support-weighted; AUC (OvR) = one-vs-rest ROC AUC, macro; AP = macro average precision (PR-AUC); Brier = multi-class Brier score (lower is better); ECE = expected calibration error, 15 bins (lower is better); Kappa = Cohen's kappa; MCC = Matthews correlation; Top-2 = top-2 accuracy. Fashion-MNIST test set is balanced (1,000 per class), so accuracy, balanced accuracy and micro-F1 coincide up to rounding.
