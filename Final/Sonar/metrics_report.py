# # ============================================================================
# # METRICS REPORTING
# # ============================================================================
#
# import numpy as np
# from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
#
#
# def compute_extra_metrics(y_true, preds, probs):
#     prec, rec, f1, _ = precision_recall_fscore_support(
#         y_true, preds, average="binary", zero_division=0
#     )
#     auc = roc_auc_score(y_true, probs)
#     return prec, rec, f1, auc
#
#
# def print_aggregated_results(all_results):
#     print("\n" + "=" * 120)
#     print("AGGREGATED RESULTS (MEAN ± STD)")
#     print("=" * 120)
#     print(f"{'Method':<12} {'ValAcc':<14} {'TestAcc':<14} "
#           f"{'Precision':<14} {'Recall':<14} {'Specificity':<12} {'F1':<14} {'AUC':<14}")
#     print("-" * 120)
#
#     def ms(arr):
#         return np.mean(arr), np.std(arr)
#
#     # Evolved
#     ev = {
#         "val": ms(all_results["evolved_val"]),
#         "test": ms(all_results["evolved_test"]),
#         "prec": ms(all_results["evolved_precision"]),
#         "rec": ms(all_results["evolved_recall"]),
#         "spec": ms(all_results["evolved_specificity"]),
#         "f1": ms(all_results["evolved_f1"]),
#         "auc": ms(all_results["evolved_auc"]),
#     }
#
#     print(f"{'Evolved':<12}"
#           f"{ev['val'][0]:>6.4f}±{ev['val'][1]:<7.4f}"
#           f"{ev['test'][0]:>6.4f}±{ev['test'][1]:<7.4f}"
#           f"{ev['prec'][0]:>6.4f}±{ev['prec'][1]:<7.4f}"
#           f"{ev['rec'][0]:>6.4f}±{ev['rec'][1]:<7.4f}"
#           f"{ev['spec'][0]:>6.4f}±{ev['spec'][1]:<7.4f}"
#           f"{ev['f1'][0]:>6.4f}±{ev['f1'][1]:<7.4f}"
#           f"{ev['auc'][0]:>6.4f}±{ev['auc'][1]:<7.4f}"
#     )
#
#     # ReLU
#     re = {
#         "val": ms(all_results["relu_val"]),
#         "test": ms(all_results["relu_test"]),
#         "prec": ms(all_results["relu_precision"]),
#         "rec": ms(all_results["relu_recall"]),
#         "spec": ms(all_results["relu_specificity"]),
#         "f1": ms(all_results["relu_f1"]),
#         "auc": ms(all_results["relu_auc"]),
#     }
#
#     print(f"{'ReLU':<12}"
#           f"{re['val'][0]:>6.4f}±{re['val'][1]:<7.4f}"
#           f"{re['test'][0]:>6.4f}±{re['test'][1]:<7.4f}"
#           f"{re['prec'][0]:>6.4f}±{re['prec'][1]:<7.4f}"
#           f"{re['rec'][0]:>6.4f}±{re['rec'][1]:<7.4f}"
#           f"{re['spec'][0]:>6.4f}±{ev['spec'][1]:<7.4f}"
#           f"{re['f1'][0]:>6.4f}±{re['f1'][1]:<7.4f}"
#           f"{re['auc'][0]:>6.4f}±{re['auc'][1]:<7.4f}"
#     )
#
#     # Swish
#     sw = {
#         "val": ms(all_results["swish_val"]),
#         "test": ms(all_results["swish_test"]),
#         "prec": ms(all_results["swish_precision"]),
#         "rec": ms(all_results["swish_recall"]),
#         "spec": ms(all_results["swish_specificity"]),
#         "f1": ms(all_results["swish_f1"]),
#         "auc": ms(all_results["swish_auc"]),
#     }
#
#     print(f"{'Swish':<12}"
#           f"{sw['val'][0]:>6.4f}±{sw['val'][1]:<7.4f}"
#           f"{sw['test'][0]:>6.4f}±{sw['test'][1]:<7.4f}"
#           f"{sw['prec'][0]:>6.4f}±{sw['prec'][1]:<7.4f}"
#           f"{sw['rec'][0]:>6.4f}±{sw['rec'][1]:<7.4f}"
#           f"{sw['spec'][0]:>6.4f}±{ev['spec'][1]:<7.4f}"
#           f"{sw['f1'][0]:>6.4f}±{sw['f1'][1]:<7.4f}"
#           f"{sw['auc'][0]:>6.4f}±{sw['auc'][1]:<7.4f}"
#     )
#
#     print("=" * 120)
# # ============================================================================
# # METRICS REPORTING
# # ============================================================================
#
# import numpy as np
# from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
#
#
# def compute_extra_metrics(y_true, preds, probs):
#     prec, rec, f1, _ = precision_recall_fscore_support(
#         y_true, preds, average="binary", zero_division=0
#     )
#     auc = roc_auc_score(y_true, probs)
#     return prec, rec, f1, auc
#
#
# def print_aggregated_results(all_results):
#     print("\n" + "=" * 120)
#     print("AGGREGATED RESULTS (MEAN ± STD)")
#     print("=" * 120)
#     print(f"{'Method':<12} {'ValAcc':<14} {'TestAcc':<14} "
#           f"{'Precision':<14} {'Recall':<14} {'Specificity':<12} {'F1':<14} {'AUC':<14}")
#     print("-" * 120)
#
#     def ms(arr):
#         return np.mean(arr), np.std(arr)
#
#     # Evolved
#     ev = {
#         "val": ms(all_results["evolved_val"]),
#         "test": ms(all_results["evolved_test"]),
#         "prec": ms(all_results["evolved_precision"]),
#         "rec": ms(all_results["evolved_recall"]),
#         "spec": ms(all_results["evolved_specificity"]),
#         "f1": ms(all_results["evolved_f1"]),
#         "auc": ms(all_results["evolved_auc"]),
#     }
#
#     print(f"{'Evolved':<12}"
#           f"{ev['val'][0]:>6.4f}±{ev['val'][1]:<7.4f}"
#           f"{ev['test'][0]:>6.4f}±{ev['test'][1]:<7.4f}"
#           f"{ev['prec'][0]:>6.4f}±{ev['prec'][1]:<7.4f}"
#           f"{ev['rec'][0]:>6.4f}±{ev['rec'][1]:<7.4f}"
#           f"{ev['spec'][0]:>6.4f}±{ev['spec'][1]:<7.4f}"
#           f"{ev['f1'][0]:>6.4f}±{ev['f1'][1]:<7.4f}"
#           f"{ev['auc'][0]:>6.4f}±{ev['auc'][1]:<7.4f}"
#     )
#
#     # ReLU
#     re = {
#         "val": ms(all_results["relu_val"]),
#         "test": ms(all_results["relu_test"]),
#         "prec": ms(all_results["relu_precision"]),
#         "rec": ms(all_results["relu_recall"]),
#         "spec": ms(all_results["relu_specificity"]),
#         "f1": ms(all_results["relu_f1"]),
#         "auc": ms(all_results["relu_auc"]),
#     }
#
#     print(f"{'ReLU':<12}"
#           f"{re['val'][0]:>6.4f}±{re['val'][1]:<7.4f}"
#           f"{re['test'][0]:>6.4f}±{re['test'][1]:<7.4f}"
#           f"{re['prec'][0]:>6.4f}±{re['prec'][1]:<7.4f}"
#           f"{re['rec'][0]:>6.4f}±{re['rec'][1]:<7.4f}"
#           f"{re['spec'][0]:>6.4f}±{ev['spec'][1]:<7.4f}"
#           f"{re['f1'][0]:>6.4f}±{re['f1'][1]:<7.4f}"
#           f"{re['auc'][0]:>6.4f}±{re['auc'][1]:<7.4f}"
#     )
#
#     # Swish
#     sw = {
#         "val": ms(all_results["swish_val"]),
#         "test": ms(all_results["swish_test"]),
#         "prec": ms(all_results["swish_precision"]),
#         "rec": ms(all_results["swish_recall"]),
#         "spec": ms(all_results["swish_specificity"]),
#         "f1": ms(all_results["swish_f1"]),
#         "auc": ms(all_results["swish_auc"]),
#     }
#
#     print(f"{'Swish':<12}"
#           f"{sw['val'][0]:>6.4f}±{sw['val'][1]:<7.4f}"
#           f"{sw['test'][0]:>6.4f}±{sw['test'][1]:<7.4f}"
#           f"{sw['prec'][0]:>6.4f}±{sw['prec'][1]:<7.4f}"
#           f"{sw['rec'][0]:>6.4f}±{sw['rec'][1]:<7.4f}"
#           f"{sw['spec'][0]:>6.4f}±{ev['spec'][1]:<7.4f}"
#           f"{sw['f1'][0]:>6.4f}±{sw['f1'][1]:<7.4f}"
#           f"{sw['auc'][0]:>6.4f}±{sw['auc'][1]:<7.4f}"
#     )
#
#     print("=" * 120)
# metrics_report.py

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def compute_extra_metrics(y_true, preds, probs):
    """
    Compute precision, recall, F1, and AUC for binary classification.
    y_true : (N,) numpy array with 0/1 labels
    preds  : (N,) numpy array predicted labels (0/1)
    probs  : (N,) numpy array probabilities of class 1
    """
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_true, probs)
    return prec, rec, f1, auc



def print_aggregated_results(all_results):
    """
    Print mean ± std for:
        - Validation Accuracy
        - Test Accuracy
        - Precision
        - Recall
        - F1-score
        - AUC
    for: Evolved, ReLU, Swish
    """

    print("\n" + "=" * 120)
    print("AGGREGATED RESULTS (MEAN ± STD)")
    print("=" * 120)
    print(f"{'Method':<12} {'ValAcc':<14} {'TestAcc':<14} "
          f"{'Precision':<14} {'Recall':<14} {'Specificity':<12} {'F1':<14} {'AUC':<14}")
    print("-" * 120)

    def ms(arr):
        return np.mean(arr), np.std(arr)

    # Evolved
    ev = {
        "val": ms(all_results["evolved_val"]),
        "test": ms(all_results["evolved_test"]),
        "prec": ms(all_results["evolved_precision"]),
        "rec": ms(all_results["evolved_recall"]),
        "spec": ms(all_results["evolved_specificity"]),
        "f1": ms(all_results["evolved_f1"]),
        "auc": ms(all_results["evolved_auc"]),
    }

    print(f"{'Evolved':<12}"
          f"{ev['val'][0]:>6.4f}±{ev['val'][1]:<7.4f}"
          f"{ev['test'][0]:>6.4f}±{ev['test'][1]:<7.4f}"
          f"{ev['prec'][0]:>6.4f}±{ev['prec'][1]:<7.4f}"
          f"{ev['rec'][0]:>6.4f}±{ev['rec'][1]:<7.4f}"
          f"{ev['spec'][0]:>6.4f}±{ev['spec'][1]:<7.4f}"
          f"{ev['f1'][0]:>6.4f}±{ev['f1'][1]:<7.4f}"
          f"{ev['auc'][0]:>6.4f}±{ev['auc'][1]:<7.4f}"
    )

    # ReLU
    re = {
        "val": ms(all_results["relu_val"]),
        "test": ms(all_results["relu_test"]),
        "prec": ms(all_results["relu_precision"]),
        "rec": ms(all_results["relu_recall"]),
        "spec": ms(all_results["relu_specificity"]),
        "f1": ms(all_results["relu_f1"]),
        "auc": ms(all_results["relu_auc"]),
    }

    print(f"{'ReLU':<12}"
          f"{re['val'][0]:>6.4f}±{re['val'][1]:<7.4f}"
          f"{re['test'][0]:>6.4f}±{re['test'][1]:<7.4f}"
          f"{re['prec'][0]:>6.4f}±{re['prec'][1]:<7.4f}"
          f"{re['rec'][0]:>6.4f}±{re['rec'][1]:<7.4f}"
          f"{re['spec'][0]:>6.4f}±{ev['spec'][1]:<7.4f}"
          f"{re['f1'][0]:>6.4f}±{re['f1'][1]:<7.4f}"
          f"{re['auc'][0]:>6.4f}±{re['auc'][1]:<7.4f}"
    )

    # Swish
    sw = {
        "val": ms(all_results["swish_val"]),
        "test": ms(all_results["swish_test"]),
        "prec": ms(all_results["swish_precision"]),
        "rec": ms(all_results["swish_recall"]),
        "spec": ms(all_results["swish_specificity"]),
        "f1": ms(all_results["swish_f1"]),
        "auc": ms(all_results["swish_auc"]),
    }

    print(f"{'Swish':<12}"
          f"{sw['val'][0]:>6.4f}±{sw['val'][1]:<7.4f}"
          f"{sw['test'][0]:>6.4f}±{sw['test'][1]:<7.4f}"
          f"{sw['prec'][0]:>6.4f}±{sw['prec'][1]:<7.4f}"
          f"{sw['rec'][0]:>6.4f}±{sw['rec'][1]:<7.4f}"
          f"{sw['spec'][0]:>6.4f}±{ev['spec'][1]:<7.4f}"
          f"{sw['f1'][0]:>6.4f}±{sw['f1'][1]:<7.4f}"
          f"{sw['auc'][0]:>6.4f}±{sw['auc'][1]:<7.4f}"
    )
    # LeakyReLU
    lr = {
        "val": ms(all_results["leakyrelu_val"]),
        "test": ms(all_results["leakyrelu_test"]),
        "prec": ms(all_results["leakyrelu_precision"]),
        "rec": ms(all_results["leakyrelu_recall"]),
        "spec": ms(all_results["leakyrelu_specificity"]),
        "f1": ms(all_results["leakyrelu_f1"]),
        "auc": ms(all_results["leakyrelu_auc"]),
    }

    print(f"{'LeakyReLU':<12}"
          f"{lr['val'][0]:>6.4f}±{lr['val'][1]:<7.4f}"
          f"{lr['test'][0]:>6.4f}±{lr['test'][1]:<7.4f}"
          f"{lr['prec'][0]:>6.4f}±{lr['prec'][1]:<7.4f}"
          f"{lr['rec'][0]:>6.4f}±{lr['rec'][1]:<7.4f}"
          f"{lr['spec'][0]:>6.4f}±{lr['spec'][1]:<7.4f}"
          f"{lr['f1'][0]:>6.4f}±{lr['f1'][1]:<7.4f}"
          f"{lr['auc'][0]:>6.4f}±{lr['auc'][1]:<7.4f}"
          )

    # ELU
    el = {
        "val": ms(all_results["elu_val"]),
        "test": ms(all_results["elu_test"]),
        "prec": ms(all_results["elu_precision"]),
        "rec": ms(all_results["elu_recall"]),
        "spec": ms(all_results["elu_specificity"]),
        "f1": ms(all_results["elu_f1"]),
        "auc": ms(all_results["elu_auc"]),
    }

    print(f"{'ELU':<12}"
          f"{el['val'][0]:>6.4f}±{el['val'][1]:<7.4f}"
          f"{el['test'][0]:>6.4f}±{el['test'][1]:<7.4f}"
          f"{el['prec'][0]:>6.4f}±{el['prec'][1]:<7.4f}"
          f"{el['rec'][0]:>6.4f}±{el['rec'][1]:<7.4f}"
          f"{el['spec'][0]:>6.4f}±{el['spec'][1]:<7.4f}"
          f"{el['f1'][0]:>6.4f}±{el['f1'][1]:<7.4f}"
          f"{el['auc'][0]:>6.4f}±{el['auc'][1]:<7.4f}"
          )
    print("=" * 120)
