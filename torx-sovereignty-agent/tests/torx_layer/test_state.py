import pytest
from src.torx_layer.state import PBit, PDit, PMode, STORAGE_PRECISION, _normalise, shannon_entropy

def test_pbit_basic():
    pb = PBit(0.7, label="test")
    assert pb.p == round(0.7, STORAGE_PRECISION)
    assert not pb.is_certain
    assert pb.entropy > 0
    json = pb.to_json()
    assert json["type"] == "pbit"
    assert json["p"] == pb.p
    # certain true/false
    assert PBit.certain(True).is_certainly_true()
    assert PBit.certain(False).is_certainly_false()

def test_pdit_constructors_and_props():
    outcomes = ("a", "b", "c")
    pd = PDit.uniform(outcomes, label="u")
    assert sum(pd.probs) == 1.0
    assert all(abs(p - 1/3) <= 1e-9 for p in pd.probs)
    # certain
    pd2 = PDit.certain(outcomes, "b")
    assert pd2.prob("b") == 1.0
    assert pd2.max_prob == 1.0
    # from_scores softmax
    scores = {"a": 2.0, "b": 1.0}
    pd3 = PDit.from_scores(scores, temperature=0.5)
    assert pd3.argmax in scores
    assert pd3.margin >= 0
    # errors
    with pytest.raises(ValueError):
        PDit.certain(outcomes, "x")
    with pytest.raises(ValueError):
        PDit.from_scores({}, temperature=1.0)

def test_pmode_creation_and_methods():
    dims = ("x", "y")
    mean = (0.2, 0.8)
    var = (0.1, 0.2)
    pm = PMode.from_diagonal(dims, mean, var, label="test")
    assert pm.value("x") == round(0.2, STORAGE_PRECISION)
    assert pm.uncertainty("y") == round((0.2) ** 0.5, STORAGE_PRECISION)
    # confidence derived from variance
    conf = pm.confidence
    assert 0.0 <= conf <= 1.0
    # clamped
    pm2 = pm.clamped(lo=0.0, hi=0.5)
    assert pm2.mean[1] == 0.5
    # errors
    with pytest.raises(KeyError):
        pm.value("z")
    with pytest.raises(KeyError):
        pm.uncertainty("z")

def test_normalise_and_entropy():
    # normalise ensures sum to 1
    weights = [0.2, 0.3, 0.5]
    norm = _normalise(weights)
    assert abs(sum(norm) - 1.0) < 1e-9
    # entropy of uniform distribution
    ent = shannon_entropy([0.5, 0.5])
    assert round(ent, 6) == round(1.0, 6)
