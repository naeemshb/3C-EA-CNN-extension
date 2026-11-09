import numpy as np
import pytest

from cnn_extension.data import (ImputationPreprocessor, generate_masks, load_dataset, load_masks, make_synthetic,
                                save_masks, spawn_corruption_seeds, stratified_split)


@pytest.mark.parametrize("pattern", ["random", "block", "rows", "cols"])
def test_masks_are_reproducible_and_rate_is_recorded(pattern):
    a = generate_masks(50, 1, 28, 28, pattern, 0.4, seed=123)
    b = generate_masks(50, 1, 28, 28, pattern, 0.4, seed=123)
    c = generate_masks(50, 1, 28, 28, pattern, 0.4, seed=124)
    assert a.dtype == bool and a.shape == (50, 1, 28, 28)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    realized = a.mean()
    assert 0.3 < realized < 0.5
    if pattern == "random":
        assert abs(realized - 0.4) < 0.02
    if pattern in ("rows", "cols"):
        assert abs(realized - round(0.4 * 28) / 28) < 1e-9     # quantised, exactly recorded
    if pattern == "block":
        per_image = a.reshape(50, -1).mean(1)
        assert np.allclose(per_image, per_image[0])            # one rectangle of fixed size per image


def test_rgb_masks_hide_all_colour_channels_at_a_location():
    m = generate_masks(10, 3, 32, 32, "random", 0.3, seed=1)
    assert np.array_equal(m[:, 0], m[:, 1]) and np.array_equal(m[:, 0], m[:, 2])


def test_clean_rate_and_invalid_pattern():
    assert generate_masks(5, 1, 8, 8, "block", 0.0, seed=0).sum() == 0
    with pytest.raises(ValueError):
        generate_masks(5, 1, 8, 8, "diagonal", 0.2, seed=0)


def test_masks_do_not_depend_on_image_content():
    # masks come from (pattern, rate, seed) only; images are never consulted
    assert np.array_equal(generate_masks(4, 1, 6, 6, "random", 0.5, 7), generate_masks(4, 1, 6, 6, "random", 0.5, 7))


def test_mask_save_load_roundtrip(tmp_path):
    m = generate_masks(7, 3, 9, 9, "rows", 0.3, seed=5)
    save_masks(str(tmp_path / "m.npz"), m, {"pattern": "rows"})
    m2, info = load_masks(str(tmp_path / "m.npz"))
    assert np.array_equal(m, m2) and info["pattern"] == "rows"


def test_corruption_seeds_are_deterministic_and_distinct():
    s1, s2 = spawn_corruption_seeds(2000), spawn_corruption_seeds(2000)
    assert s1 == s2 and len({s1["train"], s1["val"], s1["test"]}) == 3
    assert spawn_corruption_seeds(2001) != s1


def test_stratified_split_preserves_class_proportions():
    y = np.repeat(np.arange(5), [100, 50, 80, 20, 10])
    tr, va = stratified_split(y, 0.2, seed=3)
    assert len(np.intersect1d(tr, va)) == 0 and len(tr) + len(va) == len(y)
    for cls, n in zip(range(5), [100, 50, 80, 20, 10]):
        assert (y[va] == cls).sum() == round(0.2 * n)
    tr2, va2 = stratified_split(y, 0.2, seed=3)
    assert np.array_equal(va, va2)


def _fit_case(seed=0, n=40, c=1, h=6, w=6, rate=0.3):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, size=(n, c, h, w)).astype(np.float32)
    m = rng.random((n, c, h, w)) < rate
    return x, m


def test_preprocessor_uses_only_observed_training_entries():
    x, m = _fit_case()
    pre = ImputationPreprocessor().fit(x, m)
    obs = ~m
    manual_mean = (x * obs).sum(0) / obs.sum(0)
    assert np.allclose(pre.feature_mean, manual_mean, atol=1e-6)
    assert np.allclose(pre.missing_rate, m.mean(0))
    assert np.allclose(pre.conf_missing, np.maximum(0.1, 1 - m.mean(0)))
    # changing the HIDDEN ground truth under the mask changes nothing
    x2 = x.copy()
    x2[m] = 123.0
    pre2 = ImputationPreprocessor().fit(x2, m)
    assert np.array_equal(pre.feature_mean, pre2.feature_mean) and pre.fingerprint() == pre2.fingerprint()


def test_transform_replaces_missing_and_builds_confidence():
    x, m = _fit_case(seed=1)
    pre = ImputationPreprocessor().fit(x, m)
    x_imp, m_f, conf = pre.transform(x, m)
    assert x_imp.dtype == np.float32 and conf.dtype == np.float32 and m_f.dtype == np.float32
    assert np.array_equal(x_imp[~m], x[~m])                               # observed values untouched
    assert np.allclose(x_imp[m], np.broadcast_to(pre.feature_mean, x.shape)[m])
    assert np.all(conf[~m] == 1.0)
    assert np.allclose(conf[m], np.broadcast_to(pre.conf_missing, x.shape)[m])
    assert np.all((conf >= 0.1) & (conf <= 1.0))
    assert np.array_equal(m_f, m.astype(np.float32))


def test_black_observed_pixel_is_not_treated_as_missing():
    x, m = _fit_case(seed=2)
    x[:, 0, 0, 0] = 0.0
    m[:, 0, 0, 0] = False
    pre = ImputationPreprocessor().fit(x, m)
    x_imp, m_f, conf = pre.transform(x, m)
    assert np.all(x_imp[:, 0, 0, 0] == 0.0) and np.all(conf[:, 0, 0, 0] == 1.0) and np.all(m_f[:, 0, 0, 0] == 0.0)


def test_validation_and_test_values_cannot_change_fitted_state():
    x, m = _fit_case(seed=3)
    pre = ImputationPreprocessor().fit(x, m)
    fp = pre.fingerprint()
    xv, mv = _fit_case(seed=99, n=10)
    pre.transform(xv * 5.0, mv)             # arbitrary "validation" values
    pre.transform(xv * 0.0, ~mv)            # arbitrary "test" values
    assert pre.fingerprint() == fp


def test_fallbacks_for_never_observed_features_and_channels():
    x, m = _fit_case(seed=4, c=2, h=4, w=4, rate=0.2)
    m[:, 0, 1, 1] = True                    # feature never observed -> channel mean
    m[:, 1] = True                          # whole channel never observed -> constant 0.5
    pre = ImputationPreprocessor().fit(x, m)
    obs0 = ~m[:, 0]
    channel_mean = (x[:, 0] * obs0).sum() / obs0.sum()
    assert np.isclose(pre.feature_mean[0, 1, 1], channel_mean, atol=1e-6)
    assert np.all(pre.feature_mean[1] == 0.5)
    assert pre.fallback_counts == {"features_without_observed_train_entries": 17,
                                   "features_using_channel_fallback_constant": 16}
    assert np.all(pre.conf_missing[1] == 0.1)        # r = 1 -> max(0.1, 0)


def test_preprocessor_save_load_roundtrip(tmp_path):
    x, m = _fit_case(seed=5, c=3)
    pre = ImputationPreprocessor().fit(x, m)
    pre.save(str(tmp_path / "pre.npz"))
    pre2 = ImputationPreprocessor.load(str(tmp_path / "pre.npz"))
    assert pre2.fingerprint() == pre.fingerprint() and pre2.tau == 0.1
    a = pre.transform(x, m)
    b = pre2.transform(x, m)
    assert all(np.array_equal(u, v) for u, v in zip(a, b))


def test_synthetic_dataset_and_loader_metadata():
    splits, meta = load_dataset("synthetic", val_fraction=0.1, split_seed=0)
    assert meta["in_channels"] == 1 and meta["num_classes"] == 2 and meta["sizes"]["train"] == 160
    x = splits["train"].x
    assert x.dtype == np.float32 and x.min() >= 0 and x.max() <= 1 and x.shape[1:] == (1, 28, 28)
    assert set(np.unique(splits["train"].y)) == {0, 1}
    s2 = make_synthetic(seed=0)
    assert np.array_equal(s2["train"].x, x)
    with pytest.raises(ValueError):
        load_dataset("imagenet")
