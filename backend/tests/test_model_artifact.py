from loan_prediction.service import get_bundle


def test_model_bundle_has_champion_and_challenger():
    bundle = get_bundle()
    assert "champion" in bundle
    assert "challenger" in bundle
    assert bundle["metadata"]["champion_model"]
