# Proposed TELO Extension: Learning Imputation Reliability for Tabular 3C-EA

Date: 2026-11-09  
Status: Research proposal. The proposed improvements have not yet been established experimentally.

## Recommendation

Return to the tabular MLP and build the GECCO 2026 journal extension around better confidence estimation, supported by a deeper analysis of ChannelProp and evaluation under changed missingness.

The central research question is:

> Does estimating the reliability of each imputed value help evolved activations handle missing data better?

The proposed package has three connected contributions:

1. An entry-specific confidence estimator.
2. A theoretical and experimental explanation of when confidence and ChannelProp help.
3. Evaluation when test-time missingness differs from training.

Initially retain mean imputation, the MLP, the original single-objective GP search, and ChannelProp. This makes the effect of changing confidence easier to isolate.

## 1. Estimate confidence for each missing entry

### Current method

The original method assigns confidence 1 to observed entries. For a missing entry in feature j, it assigns:

`c_ij = max(0.1, 1 - r_j)`

Here, `r_j` is the feature's missingness frequency in the training data.

Two missing entries in the same column therefore receive identical confidence even if their imputations have different reliability.

### Proposed change

Estimate confidence separately for each missing entry, using the other observed features and the missingness pattern.

For a feature missing in 30% of training records:

| Record | Original confidence | Proposed confidence, illustrative |
|---|---:|---:|
| Observed features suggest the imputed value is likely accurate | 0.70 | 0.90 |
| Observed features suggest the imputed value is unreliable | 0.70 | 0.30 |

These proposed scores are examples, not measured results or manually assigned values.

### Training procedure

1. Deliberately hide some observed values within the training partition.
2. Impute the hidden entries with the existing imputation method.
3. Compare the imputations with their known original values.
4. Train a small model to predict imputation reliability from the remaining observed features and the missingness pattern.
5. Generate training confidence scores using held-out folds, so a record does not help fit its own confidence estimate. Fit the imputer and other learned preprocessing within the corresponding training folds as well.
6. Fit the final estimator using only the outer training partition and freeze it before applying it to validation or test records.

A possible interpretable target is the probability that an imputation falls within a predefined, feature-scaled error tolerance:

`P(abs(imputed_value - true_value) <= tolerance_j | observed_features, mask)`

Define numeric tolerances using training data. Any tolerance selection or estimator tuning must exclude the test data. Check whether the resulting probabilities are calibrated on held-out, deliberately hidden values.

Observed entries retain confidence 1 under the original assumption that observed values are reliable. This score concerns imputation reliability; it is not the classifier's probability of predicting the correct class.

## 2. Explain when confidence and ChannelProp add useful information

### A theoretical observation

If every feature uses the same missingness rate r, and the original confidence floor is inactive, then:

`c = 1 - r * m`

ChannelProp applies the same row-normalized nonnegative routing matrix A to both channels. Since its rows sum to one:

`c_out = A * c = A * (1 - r * m) = 1 - r * m_out`

Thus the relationship is preserved through ChannelProp layers. In this special case, confidence is completely determined by missingness.

With empirical feature-specific rates, the relationship is generally approximate rather than exact. This observation does not imply that the original method cannot help; it identifies a condition under which the confidence channel contains no additional information beyond the mask.

The extension should investigate whether entry-specific confidence distinguishes records that the missingness channel treats alike.

### New analyses

- Check whether high confidence corresponds to smaller imputation errors.
- Examine whether evolved formulas respond to confidence over values actually encountered during inference.
- Replace learned confidence with feature-level averages to test the value of entry-specific variation.
- Shuffle confidence among missing entries of the same feature as a diagnostic control.
- Track how confidence differences change through successive layers.
- Compare propagation throughout the network with confidence supplied only at the input or first layer.

The original paper already contains no-confidence and no-ChannelProp ablations. The new experiments should explain the mechanism more directly, rather than simply repeat those comparisons.

A weighted average of calibrated input scores is not automatically a calibrated probability of hidden-unit correctness. Treat propagated confidence as a reliability summary unless a stronger interpretation is justified.

## 3. Test changes in missingness after training

The original work already evaluates MCAR, MAR, MNAR, and different missingness rates. The extension should explicitly study a mismatch between training and deployment conditions.

| Training condition | Test conditions |
|---|---|
| 20% random missingness | 20%, 40%, and 60% random missingness |
| Randomly missing individual features | Groups of related features missing together |
| One specified missingness mechanism | A different specified mechanism |

Keep the trained classifier, confidence estimator, imputation statistics, and other preprocessing fixed during these tests. Specify the test conditions before examining their results.

This evaluates robustness to changed conditions, rather than performance after retraining for each condition.

### Limits of confidence evaluation

For genuinely missing values, the true values are unavailable, so their imputation errors cannot be measured directly. Begin confidence evaluation with deliberately hidden, known values.

Calibration under random masking does not guarantee calibration under MNAR or a changed missingness distribution. Evaluate these conditions separately and state the assumptions and limitations.

## 4. Start with a small tabular pilot

Use WDBC and PIMA first, keeping their existing preprocessing conventions explicit and consistent across methods.

First evaluate the confidence estimator itself, before spending time on GP. If it does not distinguish reliable from unreliable imputations on held-out training data, investigate that failure before launching a large search.

Then compare:

| Variant | Purpose |
|---|---|
| Original 3C-EA | Establish the reference |
| 3C-EA with learned confidence | Test the proposed change |
| Learned confidence replaced by feature-level averages | Test whether entry-specific information matters |
| Learned confidence with propagation disabled | Test whether carrying it through the network matters |

Retain the fixed-activation baselines for context. Use paired data splits, missingness masks, seeds, and comparable training/search budgets.

Existing activation formulas can support inexpensive preliminary tests. Such tests assess those formulas under changed confidence; they do not establish how reliably GP discovers useful formulas. The final method comparison should evolve activations under each confidence scheme.

If the pilot is informative, expand to Sonar, Glass, and selected naturally incomplete datasets already used in the project.

## 5. Report more than classification accuracy

The evaluation should connect the proposed mechanism to the final task:

- **Confidence quality:** reliability plots and a proper probability score for the chosen imputation-success event; relationship between confidence and reconstruction error.
- **Classification:** AUC, accuracy, and macro-F1 as appropriate to the dataset.
- **Robustness:** changes in performance under the predefined missingness shifts.
- **Mechanism:** confidence use by evolved expressions and the effect of propagation.
- **Cost:** confidence-estimator fitting, preprocessing, GP search, final training, and inference.

Use repeated runs and paired comparisons for confirmatory experiments. Keep development decisions separate from final test evaluation. Do not infer a dependable improvement from a single seed.

## 6. Distinction from the planned GECCO 2027 paper

The current GECCO 2027 draft focuses on multi-objective activation search, a structural archive, and ensembles. It retains the original frequency-based confidence rule.

The proposed TELO extension focuses on the reliability information supplied to individual evolved activations and its propagation through the network.

| Aspect | Proposed TELO extension | Current GECCO 2027 direction |
|---|---|---|
| Main contribution | Entry-specific confidence and its mechanism | Multi-objective search, archive, and ensembles |
| Search | Original single-objective GP initially | Multi-objective GP |
| Prediction | A single selected activation/model initially | Ensembles of independently trained models |
| Central analysis | Imputation reliability and ChannelProp behavior | Search trade-offs and ensemble contributions |

This is a scope distinction based on the current draft, not a blanket guarantee of non-overlap. Describe shared foundations and cross-reference the related work appropriately.

## 7. Contribution and novelty assessment

Learning imputation uncertainty is an existing research area. The novelty claim should therefore concern the specific integration with evolved three-channel activations and ChannelProp, supported by controlled evidence.

The proposed journal contribution is:

> An entry-specific confidence method, a theoretical and experimental explanation of ChannelProp, and evaluation under changed missingness.

Together, these could provide a substantive extension toward the invitation's requirement of at least 30% new content. Neither the number of added pages nor this proposal alone guarantees that the requirement is satisfied; the completed contribution, evidence, and editorial assessment matter.

The immediate next step is the small confidence-estimation pilot, followed by a decision based on validation evidence.

## References and context

- [GECCO 2026 paper: Evolving Multi-Channel Confidence-Aware Activation Functions for Missing Data with Channel Propagation](https://arxiv.org/abs/2602.13864). The paper identifies combining heuristic and learned confidence estimates as future work.
- [From Data Imputation to Data Cleaning: Automated Cleaning of Tabular Data Improves Downstream Predictive Performance](https://proceedings.mlr.press/v238/jager24a.html), AISTATS 2024. Related work on imputation and uncertainty calibration.
- [Beyond Accuracy: An Empirical Study of Uncertainty Estimation in Imputation](https://arxiv.org/abs/2511.21607). Related work examining imputation uncertainty, calibration, and computational trade-offs.
- GECCO 2027 draft reviewed in the discussion: *Trustworthy Missing-Data Classification via Quality-Diversity Ensembles of Evolved Confidence-Aware Activations*.

