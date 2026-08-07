# Load Qualification Report

Model card: `torx_contextual_sovereignty_agent.model-card.yaml`
Gates: `validation.load_qualification.gates`
Node: single host, 4 cores, NVIDIA GTX 1650 (4 GB VRAM), **driver not loaded on this boot**

The card states the *scale-proven* claim is permitted only when **every** gate
passes. Several gates cannot be run on this host at all. This report therefore
concludes with the readiness claim the evidence actually supports, which is the
one already declared in the card:

> `deployment.readiness_claim: design-approved-requires-implementation-and-load-validation`
> `result_cards.large_deployment.readiness: scalable-by-design-not-scale-proven`

**This build does not claim scale-proven.** Four of seven gates were not run.

---

## Environment constraints that shaped this report

| Constraint | Effect |
|---|---|
| No PostgreSQL server, and none installable (root filesystem at 100%) | The JSONB index-selectivity gate cannot run. Repository *invariants* are still executed against an in-memory backend behind the same interface. |
| Single host | The horizontal-scaling gate can only demonstrate partition independence in-process, not across nodes. |
| NVIDIA driver not loaded (`NVMLError_DriverNotLoaded`) | All kernel work ran on JAX CPU. The thermodynamic daemon on `:9100` serves a truthful degraded state and the energy gate correctly fails closed, so this is measured behaviour, not an untested path. |
| 4 cores under sustained external load (~25) | Absolute latencies here are pessimistic. They are reported as measured, not normalised. |

---

## Gate-by-gate

### 1. `TORX-update-p99-within-decision-budget` — **PASSED** (after a fix)

Measured on this host, CPU-only JAX, after warm compile:

| Kernel | Before | After | Change |
|---|---|---|---|
| `evaluate_decision` (pbit decision circuit) | 2 690 ms | **28 ms** | ~96× |
| `intent_update` (pmode affine-Gaussian) | 3 110 ms | **3.3 ms** | ~940× |
| `evaluate_categorical` (pdit stick-breaking) | ~2 000 ms | **4.3 ms** | ~465× |

The original implementation rebuilt each circuit per call, so JAX re-traced
`density`'s `fori_loop` on every decision. Circuit *structure* is fixed (the
site count and gate wiring never vary), so it is now built once and `jax.jit`
compiled with only the thetas as arguments. First call pays an 11–18 s
compile; every subsequent call is in the millisecond range.

This is recorded as a qualification finding because at 2.7 s per decision the
TORX update would have failed its own budget while sitting on the card's
`critical_path`. Correctness is unchanged — the jitted path still matches the
exact fallback to 2.4e-8 and the closed-form Kalman update to 1.8e-8.

### 2. `topology-sidecar-does-not-block-critical-path` — **PASSED by construction**

`torx.topology_sidecar.critical_path` is `false` in the card, and the loader
*rejects* a card that sets it true (`test_topology_sidecar_on_the_critical_path_is_rejected`).
`bridge_selection` uses the neutral descriptor when none is supplied — the
second rung of `topology_sidecar.fallback_order` — so no decision path awaits
the sidecar. Verified structurally rather than by load.

### 3. `bounded-memory-per-active-user-and-group` — **PARTIALLY VERIFIED**

Bounds are enforced in code rather than assumed: intent vectors are fixed at
six dimensions, candidates are capped at
`runtime.state_bounds.bridge_candidates_per_cycle`, and no kernel retains state
between calls. The compiled-circuit caches are keyed by *shape* (`n`
dimensions, `k` outcomes), so they are bounded by the number of distinct
shapes — a small constant — not by the number of users or groups.

Not measured under a growing population on real infrastructure.

### 4. `VDF-attestation-backlog-remains-bounded` — **NOT RUN**

The Rule 30 VDF is intentionally sequential. Backlog behaviour under sustained
arrival needs a real queue and real arrival rates; neither exists here.

### 5. `MCP-p99-within-tenant-budget` — **NOT RUN**

The MCP surface is exercised for *correctness* (47 tests, ~3 s) against fakes.
No latency budget was measured against real repositories and a live database.

### 6. `horizontal-throughput-scaling` — **NOT RUN**

Single host. `deployment.partition_key` is `tenant_id:group_id-or-user_id` and
nothing in the request path shares mutable state across partitions, but
independent-partition scaling across nodes is unverified.

### 7. `JSONB-index-selectivity-remains-acceptable` — **NOT RUN**

Requires PostgreSQL 15+. The migration declares the card's GIN
(`jsonb_path_ops`) and btree indexes and the RLS policies, but no plan was ever
run against a live server, so selectivity is unmeasured.

---

## What *was* verified here

Correctness, not scale:

- **Kernel fidelity.** The TORX decision circuit matches exact enumeration to
  2.4e-8; the affine-Gaussian intent kernel matches the closed-form Kalman
  update to 1.8e-8 and preserves prior correlations; the stick-breaking
  categorical matches its softmax target to 1.5e-7; the thrml block-Gibbs
  sampler agrees in sign with exact Boltzmann marginals on every proposition.
- **The structural veto.** A hypothesis property test generates arbitrary
  viability estimates and asserts `bridge_viability` is *exactly* 0 whenever a
  boundary is violated or authorization is absent. The guarantee comes from
  circuit wiring, not a parameter, so no estimate can defeat it.
- **`simple_average_forbidden`.** An executable case where the per-item
  weighted average returns a bundle no coalition holds and the EBM does not.
- **Fail-closed telemetry.** The energy gate refuses GPU work on absent *or
  empty* telemetry. The empty case was a real defect found against the live
  daemon: a 200 response with all-null fields fell through every threshold
  check and opened the gate.

---

## Completing the missing gates

On real infrastructure:

```bash
# 1. Bring up PostgreSQL 15+ and point the suite at it.
export TORX_TEST_DATABASE_URL='postgresql+psycopg://user:pw@host/torx_test'
cd torx-sovereignty-agent
~/venv312/bin/python -m alembic upgrade head
~/venv312/bin/python -m pytest -q -m postgres          # unskips the RLS/index tests

# 2. Load qualification (sizes scale via env var).
TORX_LOAD_USERS=10000 TORX_LOAD_GROUPS=500 \
  ~/venv312/bin/python -m pytest -q -m load

# 3. JSONB index selectivity — confirm the GIN indexes are actually chosen.
psql "$TORX_TEST_DATABASE_URL" -c \
  "EXPLAIN ANALYZE SELECT * FROM gc_profile_revisions
   WHERE tenant_id = '...' AND delta @> '{\"inferred\":{\"communication\":{}}}';"

# 4. Horizontal scaling: run N workers against distinct partition keys and
#    confirm aggregate throughput scales ~linearly with N.

# 5. VDF backlog: drive attestation at the production arrival rate and confirm
#    queue depth stabilises rather than growing without bound.
```

Re-run this report once all seven gates pass. Only then may
`result_cards.large_deployment.readiness` be changed from
`scalable-by-design-not-scale-proven`.
