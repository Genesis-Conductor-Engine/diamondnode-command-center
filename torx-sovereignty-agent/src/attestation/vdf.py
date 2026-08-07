"""Rule 30 verifiable delay function for event attestation.

``rule30_vdf.verification`` in the card makes two claims the code has to make
true: **deterministic** and **offline_verification**. Both are enforced by
construction here.

* *Deterministic.* :func:`prove` is a pure function of
  ``(document, seed, difficulty, width)`` — the CA is a plain bit loop with no
  clock, no ``random``, and no backend. Proofs for the same inputs are
  byte-identical, which is what lets a later ``attestation.verify`` agree with
  an earlier ``prove`` without either side storing "the answer".
* *Offline.* :func:`verify` needs only the envelope. It does not re-hash the
  original document (the envelope's ``input_hash`` is that value); it re-runs
  the Rule 30 chain from the envelope's own ``seed``/``difficulty``/``width``
  and checks the recomputed ``output`` and checkpoint chain against what the
  envelope claims. When the caller does hand the original document over,
  :func:`verify` additionally recomputes ``input_hash`` and rejects a mismatch
  — a document that does not hash to its envelope's ``input_hash`` is not the
  document the envelope attests.

**Rule 30 VDF construction.** The seed is expanded into a ``width``-bit ring.
Rule 30 (``next = left ^ (center | right)``) is applied ``difficulty`` times
with wraparound boundaries so the state never collapses to all-zeros and the
output is a fixed-width function of the input. ``output`` is the SHA-256 of the
final packed state; ``proof`` is a chain of intermediate state digests taken
every ``checkpoint_every`` steps. Verification recomputes the whole run and
compares both, so tampering with ``seed``, ``difficulty``, ``output`` or any
checkpoint is caught even though nothing about the original document is needed.

**Honest limits (``rule30_vdf.non_claims``).** This is not a randomness source
(the seed is fixed, not sampled), not a consensus mechanism, and — critically —
not a proof that the underlying inference was *correct*. It binds the *bytes*
of an event to an attestation so a history rewrite is detectable; it says
nothing about whether those bytes described a good decision.
"""

from __future__ import annotations

import hashlib
import uuid
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any, Iterator, Mapping

from src.observability.metrics import MetricRegistry
from src.observability.tracing import SpanAttributes, span
from src.personalization.evidence import canonical_json, content_hash

#: ``verifier_version`` written into every envelope. The persistence tests and
#: the card's ``proof_envelope`` fixtures reference this exact string.
VERIFIER_VERSION = "rule30-vdf/1"

#: Fields the canonical input must exclude before hashing
#: (``rule30_vdf.canonical_input.excluded_fields``). The proof must never be
#: part of the bytes it attests; ``transient-runtime-handles`` are process-local
#: objects (open files, sockets, device handles) that cannot be part of a
#: durable, portable document.
EXCLUDED_FIELDS = ("vdf_proof", "transient-runtime-handles")

#: A proof is only useful as a *delay* if the chain is long enough that skipping
#: it was the point. Small enough to stay snappy in tests, large enough that a
#: real call site is not emitting a one-step proof.
DEFAULT_DIFFICULTY = 1024

#: Ring width in bits. 256 keeps the state comfortably inside one or two machine
#: words per row and gives ``difficulty`` a non-trivial value to chew on.
DEFAULT_WIDTH = 256

#: How often (in steps) a state digest is appended to ``proof.checkpoints``.
CHECKPOINT_EVERY = 64


class AttestationError(ValueError):
    """An envelope is structurally unusable: a required field is missing."""


def _strip_excluded(value: Any) -> Any:
    """Recursively remove :data:`EXCLUDED_FIELDS` from a document.

    Applied before canonicalisation so ``input_hash`` never depends on bytes
    that cannot outlive the process (runtime handles) or that would be
    self-referential (the proof of the hash it is hashed into).
    """
    if isinstance(value, Mapping):
        return {
            str(k): _strip_excluded(v)
            for k, v in value.items()
            if str(k) not in EXCLUDED_FIELDS
        }
    if isinstance(value, (list, tuple)):
        return [_strip_excluded(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return {_strip_excluded(v) for v in value}
    return value


def input_hash(document: Any) -> str:
    """The hash the envelope attests: SHA-256 of the excluded-stripped document.

    Uses the same canonical encoding as every other content hash in the system
    (``personalization.evidence.content_hash``), so a document written by one
    component verifies when read by another.
    """
    return content_hash(_strip_excluded(document))


def _expand_seed(seed: str, width: int) -> list[int]:
    """Deterministically expand a seed string into ``width`` bits.

    Repeated SHA-256 in counter style: the first digest seeds the first
    ``width`` bits, each subsequent digest re-feeds the previous one, so the
    expansion is fixed-width and free of the seed's own length.
    """
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    bits: list[int] = []
    while len(bits) < width:
        bits.extend((byte >> k) & 1 for byte in digest for k in range(8))
        digest = hashlib.sha256(digest).digest()
    return bits[:width]


def _pack_bits(bits: list[int]) -> bytes:
    """LSB-first packing of a bit list into bytes."""
    out = bytearray()
    for i in range(0, len(bits), 8):
        chunk = bits[i : i + 8]
        b = 0
        for shift, bit in enumerate(chunk):
            b |= bit << shift
        out.append(b)
    return bytes(out)


def _state_digest(bits: list[int]) -> str:
    return hashlib.sha256(_pack_bits(bits)).hexdigest()


def _rule30_step(bits: list[int]) -> list[int]:
    """One Rule 30 generation on a ring: ``next = left ^ (center | right)``."""
    width = len(bits)
    return [
        bits[(i - 1) % width] ^ (bits[i] | bits[(i + 1) % width])
        for i in range(width)
    ]


def _utc_z(dt: datetime | None) -> str:
    """Canonical UTC timestamp (``Z`` suffix), matching canonical datetime form."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def prove(
    document: Any,
    *,
    seed: str | None = None,
    difficulty: int = DEFAULT_DIFFICULTY,
    width: int = DEFAULT_WIDTH,
    checkpoint_every: int = CHECKPOINT_EVERY,
    proof_id: str | uuid.UUID | None = None,
    event_id: str | uuid.UUID | None = None,
    created_at: datetime | None = None,
    tenant_id: str | None = None,
    registry: MetricRegistry | None = None,
    card: Any = None,
) -> dict[str, Any]:
    """Attest a document: run the Rule 30 delay chain and return the envelope.

    The envelope carries exactly the card's ``proof_envelope.required_fields``:
    ``proof_id``, ``event_id``, ``input_hash``, ``seed``, ``difficulty``,
    ``output``, ``proof``, ``created_at``, ``verifier_version``.

    ``seed`` defaults to the document's ``input_hash`` so the same document
    always attests identically (and proving twice for the same event is
    idempotent). Supply an explicit ``seed`` when two different documents must
    not share a proof chain.
    """
    if difficulty < 1:
        raise ValueError(f"difficulty must be >= 1, got {difficulty}")
    if width < 3:
        raise ValueError(f"width must be >= 3, got {width}")
    if checkpoint_every < 1:
        raise ValueError(f"checkpoint_every must be >= 1, got {checkpoint_every}")

    digest = input_hash(document)
    chain_seed = seed if seed is not None else digest
    timer = (
        registry.timed("gc_vdf_attestation_seconds")
        if registry is not None
        else nullcontext()
    )
    with timer:
        bits = _expand_seed(chain_seed, width)
        checkpoints: list[str] = []
        for step in range(1, difficulty + 1):
            bits = _rule30_step(bits)
            if step % checkpoint_every == 0:
                checkpoints.append(_state_digest(bits))

    envelope = {
        "proof_id": str(proof_id or uuid.uuid4()),
        "event_id": str(event_id or uuid.uuid4()),
        "input_hash": digest,
        "seed": chain_seed,
        "difficulty": difficulty,
        "output": _state_digest(bits),
        "proof": {
            "width": width,
            "checkpoint_every": checkpoint_every,
            "checkpoints": checkpoints,
        },
        "created_at": _utc_z(created_at),
        "verifier_version": VERIFIER_VERSION,
    }

    return envelope


def _envelope_fields(envelope: Mapping[str, Any]) -> tuple[str, ...]:
    required = (
        "proof_id",
        "event_id",
        "input_hash",
        "seed",
        "difficulty",
        "output",
        "proof",
        "created_at",
        "verifier_version",
    )
    missing = [f for f in required if f not in envelope]
    if missing:
        raise AttestationError(
            "envelope is missing required proof_envelope fields: "
            + ", ".join(missing)
        )
    return required


def _recompute_chain(
    seed: str,
    difficulty: int,
    width: int,
    checkpoint_every: int,
) -> tuple[str, list[str]]:
    """Re-run the Rule 30 chain. Returns ``(output, checkpoints)``."""
    bits = _expand_seed(seed, width)
    checkpoints: list[str] = []
    for step in range(1, difficulty + 1):
        bits = _rule30_step(bits)
        if step % checkpoint_every == 0:
            checkpoints.append(_state_digest(bits))
    return _state_digest(bits), checkpoints


def verify(
    envelope: Mapping[str, Any],
    *,
    document: Any = None,
    tenant_id: str | None = None,
    registry: MetricRegistry | None = None,
    card: Any = None,
) -> bool:
    """Deterministically re-verify an envelope; emits the audit span.

    Returns ``True`` only if the recomputed Rule 30 chain matches the envelope's
    ``output`` and ``proof.checkpoints``. When ``document`` is supplied, the
    document's ``input_hash`` must also match. On success the span
    ``gc.vdf.attested`` is emitted; on failure ``gc.vdf.verification_failed`` —
    both carrying ``vdf.proof_id`` so a trace links the audit record to the
    proof that was checked.
    """
    _envelope_fields(envelope)
    proof = envelope["proof"]
    if not isinstance(proof, Mapping):
        raise AttestationError("envelope 'proof' must be an object")
    for field in ("width", "checkpoint_every"):
        if field not in proof:
            raise AttestationError(f"envelope proof is missing field {field!r}")

    width = int(proof["width"])
    checkpoint_every = int(proof["checkpoint_every"])
    difficulty = int(envelope["difficulty"])

    if document is not None and input_hash(document) != envelope["input_hash"]:
        ok = False
        reason = "document does not hash to the envelope's input_hash"
    else:
        timer = (
            registry.timed("gc_vdf_attestation_seconds")
            if registry is not None
            else nullcontext()
        )
        try:
            with timer:
                output, checkpoints = _recompute_chain(
                    envelope["seed"], difficulty, width, checkpoint_every
                )
            ok = output == envelope["output"] and checkpoints == list(
                proof.get("checkpoints", ())
            )
            reason = "" if ok else "recomputed Rule 30 chain does not match"
        except (ValueError, TypeError):
            ok = False
            reason = "envelope contains non-numeric difficulty/width"

    _emit_verification(ok, envelope, tenant_id, registry, card, reason=reason)
    return ok


def _emit_verification(
    ok: bool,
    envelope: Mapping[str, Any],
    tenant_id: str | None,
    registry: MetricRegistry | None,
    card: Any,
    *,
    reason: str,
) -> None:
    """Emit the audit span matching the card's ``vdf.*`` audit events.

    Emission is optional: ``tenant_id`` scopes a span to a tenant's trace store
    and is required by :class:`SpanAttributes`, so pure offline computation may
    verify without emitting. Callers with a tenant always pass it.
    """
    if tenant_id is None:
        return
    event = "gc.vdf.attested" if ok else "gc.vdf.verification_failed"
    with span(
        event,
        SpanAttributes(tenant_id=tenant_id, vdf_proof_id=str(envelope["proof_id"])),
        registry=registry,
        card=card,
    ):
        pass
    if not ok:
        _verification_failure_buffer.append(
            (str(envelope["proof_id"]), reason)
        )


#: In-process record of failed verifications, bounded. Mirrors the span ring
#: buffer's shape so a HUD or degraded-operation path can read the last failure
#: without reaching into the trace backend.
_verification_failure_buffer: list[tuple[str, str]] = []


def clear_verification_failures() -> None:
    _verification_failure_buffer.clear()


def verification_failures() -> tuple[tuple[str, str], ...]:
    return tuple(_verification_failure_buffer[-10:])
