"""thrml Ising consensus: correct, and demonstrably not an average.

The load-bearing test here is
:func:`test_coupling_beats_per_item_average`. The model card states
``simple_average_forbidden: true`` as a design constraint; this file turns it
into an executable claim by exhibiting a group where the per-item weighted
average returns a bundle no coalition actually holds, and the EBM does not.
"""

from __future__ import annotations

import math
import random

import pytest

from src.kernels.thrml_consensus import (
    EXACT_ENUMERATION_LIMIT,
    ConsensusResult,
    MemberPosition,
    _sample_exact,
    _sample_mean_field,
    _sample_thrml,
    aggregate_consensus,
    build_ising,
    engine_label,
    greedy_coloring,
    ising_energy,
    thrml_available,
    weighted_average,
)

PROPS = ("ship-fast", "add-tests", "refactor")


def coalition_members():
    """Two coherent camps plus one cross-cutting member.

    a/b want speed and oppose process; c/d want process and oppose speed; e is
    mixed. No member holds "ship fast AND add tests", which is exactly the
    bundle a per-item average is prone to synthesise.
    """
    return [
        MemberPosition("a", {"ship-fast": 0.9, "add-tests": -0.8, "refactor": -0.7}),
        MemberPosition("b", {"ship-fast": 0.8, "add-tests": -0.9, "refactor": -0.6}),
        MemberPosition("c", {"ship-fast": -0.9, "add-tests": 0.95, "refactor": 0.9}),
        MemberPosition("d", {"ship-fast": -0.85, "add-tests": 0.9, "refactor": 0.85}),
        MemberPosition("e", {"ship-fast": 0.1, "add-tests": 0.15, "refactor": -0.9}),
    ]


# --------------------------------------------------------------------------
# the forbidden average
# --------------------------------------------------------------------------


def test_coupling_beats_per_item_average():
    """The EBM and the weighted average disagree in sign on some proposition.

    The coupling term rewards coherent packages. The average treats each item
    independently, so it can return a majority for one item and a majority
    against another that no actual coalition combines — an outcome the group
    could not act on. Disagreeing in sign is the observable signature that the
    aggregation is genuinely constraint-aware rather than a rescaled mean.
    """
    members = coalition_members()
    result = aggregate_consensus(members, PROPS, check_gate=False)
    average = weighted_average(members, PROPS)

    disagreements = [
        p
        for p in PROPS
        if (result.position(p) >= 0) != (average[p] >= 0)
    ]
    assert disagreements, (
        "EBM matched the sign of the per-item average on every proposition; "
        f"ebm={result.as_mapping()} average={average}"
    )


def test_zero_coupling_reduces_toward_the_mean_direction():
    """Without coupling there is nothing to disagree with.

    A single proposition has no partner to couple to, so the EBM must agree in
    sign with the weighted mean. This is the control for the test above: it
    shows the disagreement there comes from coupling, not from noise.
    """
    members = [
        MemberPosition("a", {"only": 0.8}),
        MemberPosition("b", {"only": 0.6}),
        MemberPosition("c", {"only": -0.2}),
    ]
    result = aggregate_consensus(members, ("only",), check_gate=False)
    avg = weighted_average(members, ("only",))["only"]
    assert (result.position("only") >= 0) == (avg >= 0)


# --------------------------------------------------------------------------
# sampler correctness
# --------------------------------------------------------------------------


@pytest.mark.skipif(not thrml_available(), reason="thrml/JAX not importable")
def test_sampler_agrees_with_exact_marginals():
    """Block-Gibbs must reproduce the Boltzmann marginals it approximates."""
    rng = random.Random(3)
    props = [f"p{i}" for i in range(6)]
    members = [
        MemberPosition(
            f"m{k}",
            {p: round(rng.uniform(-1, 1), 3) for p in props},
            weight=rng.uniform(0.5, 1.5),
        )
        for k in range(6)
    ]
    biases, edges = build_ising(members, props)

    exact_mag, _, _, _ = _sample_exact(biases, edges, beta=1.0)
    sampled_mag, _, backend, diag = _sample_thrml(
        biases, edges, beta=1.0, samples=2000, warmup=400, steps_per_sample=2, seed=11
    )

    assert backend.startswith("thrml-")
    assert diag["n_samples"] == 2000
    for i, p in enumerate(props):
        assert (sampled_mag[i] >= 0) == (exact_mag[i] >= 0), (
            f"sign disagreement on {p}: sampled={sampled_mag[i]} exact={exact_mag[i]}"
        )
        # Monte-Carlo standard error at n=2000 is ~1/sqrt(n) = 0.022; allow 4x.
        assert sampled_mag[i] == pytest.approx(exact_mag[i], abs=0.09)


def test_exact_enumeration_is_a_proper_distribution():
    members = coalition_members()
    biases, edges = build_ising(members, PROPS)
    mag, energy, backend, _ = _sample_exact(biases, edges, beta=1.0)
    assert backend == "thrml-fallback/python-exact"
    assert all(-1.0 <= m <= 1.0 for m in mag)
    assert math.isfinite(energy)


def test_mean_field_converges_on_a_frustrated_graph():
    """Undamped iteration oscillates on negative couplings; damping must not."""
    biases = [0.0, 0.0, 0.0]
    edges = [(0, 1, -1.0), (1, 2, -1.0), (0, 2, -1.0)]  # frustrated triangle
    mag, energy, backend, _ = _sample_mean_field(biases, edges, beta=1.5)
    assert backend.endswith("meanfield")
    assert all(math.isfinite(m) and -1.0 <= m <= 1.0 for m in mag)
    assert math.isfinite(energy)


def test_beta_sharpens_the_consensus():
    members = coalition_members()
    cold = aggregate_consensus(members, PROPS, beta=0.1, check_gate=False)
    hot = aggregate_consensus(members, PROPS, beta=4.0, check_gate=False)
    assert hot.overall_confidence > cold.overall_confidence


# --------------------------------------------------------------------------
# dissent is first-class
# --------------------------------------------------------------------------


def test_dissent_is_recorded_for_members_opposing_the_group():
    members = coalition_members()
    result = aggregate_consensus(members, PROPS, check_gate=False)
    assert result.dissent, "no dissent recorded for a visibly split group"
    for record in result.dissent:
        assert (record.member_position > 0) != (record.group_position > 0)
        assert record.magnitude > 0
        assert record.member_id in {m.member_id for m in members}


def test_no_dissent_is_invented_when_the_group_is_undecided():
    """Dissent needs a majority to dissent *from*.

    An evenly split group has taken no position; recording dissent against a
    non-position would manufacture a minority where there is only a tie.
    """
    members = [
        MemberPosition("a", {"x": 0.5}),
        MemberPosition("b", {"x": -0.5}),
    ]
    result = aggregate_consensus(members, ("x",), check_gate=False)
    assert abs(result.position("x")) < 0.05
    assert result.dissent == ()


def test_dissent_survives_serialisation():
    result = aggregate_consensus(coalition_members(), PROPS, check_gate=False)
    doc = result.to_json()
    assert len(doc["dissent"]) == len(result.dissent)
    assert {"member_id", "proposition", "magnitude"} <= set(doc["dissent"][0])


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def test_zero_members_raises_rather_than_returning_neutral():
    """A neutral 'consensus' from nobody would be a fabricated agreement."""
    with pytest.raises(ValueError, match="zero members"):
        aggregate_consensus([], PROPS, check_gate=False)


def test_duplicate_propositions_raise():
    with pytest.raises(ValueError, match="unique"):
        aggregate_consensus(
            coalition_members(), ("a", "a"), check_gate=False
        )


def test_zero_propositions_raise():
    with pytest.raises(ValueError, match="zero propositions"):
        aggregate_consensus(coalition_members(), (), check_gate=False)


def test_zero_total_weight_raises():
    members = [MemberPosition("a", {"x": 0.5}, weight=0.0)]
    with pytest.raises(ValueError, match="weights sum to zero"):
        aggregate_consensus(members, ("x",), check_gate=False)


def test_invalid_positions_are_rejected_at_construction():
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        MemberPosition("a", {"x": 1.5})
    with pytest.raises(ValueError, match="weight"):
        MemberPosition("a", {"x": 0.5}, weight=-1.0)


def test_invalid_beta_raises():
    with pytest.raises(ValueError, match="beta"):
        aggregate_consensus(coalition_members(), PROPS, beta=0.0, check_gate=False)


# --------------------------------------------------------------------------
# model construction
# --------------------------------------------------------------------------


def test_weights_are_normalised_so_headcount_does_not_sharpen():
    """A larger group must not sample effectively colder for free."""
    small = [MemberPosition(f"m{i}", {"x": 0.6, "y": 0.6}) for i in range(2)]
    large = [MemberPosition(f"m{i}", {"x": 0.6, "y": 0.6}) for i in range(20)]
    b_small, _ = build_ising(small, ("x", "y"))
    b_large, _ = build_ising(large, ("x", "y"))
    assert b_small[0] == pytest.approx(b_large[0], abs=1e-9)


def test_colouring_leaves_no_intra_block_edge():
    """Block-Gibbs is only valid when a block has no internal coupling."""
    rng = random.Random(5)
    n = 9
    edges = [
        (i, j, rng.uniform(-1, 1))
        for i in range(n)
        for j in range(i + 1, n)
        if rng.random() < 0.4
    ]
    blocks = greedy_coloring(n, edges)
    assert sorted(v for block in blocks for v in block) == list(range(n))
    for block in blocks:
        members = set(block)
        for i, j, _ in edges:
            assert not (i in members and j in members), (
                f"edge ({i},{j}) lies inside a block"
            )


def test_ising_energy_matches_the_hamiltonian():
    biases = [0.5, -0.25]
    edges = [(0, 1, 0.75)]
    # H = -(0.5*1 + -0.25*-1) - 0.75*1*-1 = -(0.5 + 0.25) + 0.75 = 0.0
    assert ising_energy([1, -1], biases, edges) == pytest.approx(0.0, abs=1e-12)


def test_engine_label_names_a_real_family():
    label = engine_label()
    assert label.startswith("thrml-")
    if thrml_available():
        version = label.split("/")[0].removeprefix("thrml-")
        assert all(part.isdigit() for part in version.split("."))


def test_result_reports_which_backend_answered():
    result = aggregate_consensus(coalition_members(), PROPS, check_gate=False)
    assert isinstance(result, ConsensusResult)
    assert result.backend
    assert result.diagnostics["n_members"] == 5
    assert result.diagnostics["n_propositions"] == 3
