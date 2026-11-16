from flightdelay.config import is_holdout_month, load_config


def test_config_loads_and_has_required_keys():
    cfg = load_config()
    for key in ["project", "paths", "data", "periods", "target", "allowed_feature_columns"]:
        assert key in cfg


def test_periods_are_in_time_order_and_do_not_overlap():
    p = load_config()["periods"]
    assert p["initial_training"][1] < p["replay"][0]
    assert p["replay"][1] < p["final_holdout"][0]
    assert p["development_selection"][1] < p["final_holdout"][0]


def test_holdout_guard():
    cfg = load_config()
    assert is_holdout_month("2025-01", cfg)
    assert is_holdout_month("2025-12", cfg)
    assert not is_holdout_month("2024-12", cfg)


def test_target_is_not_a_feature():
    cfg = load_config()
    assert cfg["target"] not in cfg["allowed_feature_columns"]
