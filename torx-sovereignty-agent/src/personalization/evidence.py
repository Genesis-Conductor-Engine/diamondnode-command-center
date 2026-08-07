"""Structured evidence — the only thing permitted to move a profile field.

``security_and_privacy.threat_controls.prompt_injection`` makes two claims:
``profile_mutation_requires_structured_evidence`` and
``untrusted_text_cannot_directly_write_profile``. A comment saying "we sanitise
input" would not make either true. This module makes them true *in the type
system*:

* The only thing a profile writer can consume is a :class:`FeatureObservation`,
  whose ``value`` is a bounded ``float`` on a feature name drawn from the card's
  own ``personalization.feature_domains`` catalogue. There is no constructor,
  coercion, or accessor that turns text into one.
* Free text is carried as a :class:`ProvenanceNote`. It has no numeric accessor,
  no ``__float__``, and — unless tenant policy explicitly opts in — does not even
  retain the text, only a digest and a length. ``prohibited_persistence``
  forbids "raw message bodies unless explicitly configured by tenant policy",
  and ``data_minimization.store_features_not_raw_content`` is the same rule from
  the other direction.
* A source on the ``untrusted-text`` channel cannot produce an observation at
  all. That path is rejected structurally rather than scored down, because a
  scored-down injection is still an injection that succeeds at sufficient
  volume.

**Prohibited targets.** ``personalization.prohibited_persistence`` names ten
categories that may never be inferred or stored. Screening is a positive
denylist check with a stable reason code per category — not a heuristic
classifier — because a rejection has to be explainable in an audit record
(``personalization.observation.rejected``) and stable across releases. The
matcher deliberately catches near misses: an attacker who cannot write
``political_belief`` will try ``politcal_belief``, ``p0litical-belief``, or
``politicalbelief``, and a guard that only does exact string equality is a guard
that is trivially stepped around. :func:`assert_denylist_covers_card` fails the
build if the card ever grows a category this module does not screen.

**Deduplication** uses the card's own key,
``inference_update.event_deduplication_key`` = ``tenant_id:user_id:event_id``.
The event stream is ``at-least-once`` (``runtime.event_model``), so a redelivered
event that updated the profile twice would let a single interaction move a
preference as far as two.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Iterable, Iterator, Mapping, Sequence

from src.model_card import AgentModelCard, cached_model_card
from src.torx_layer.state import STORAGE_PRECISION

# --------------------------------------------------------------------------
# stable reason codes
# --------------------------------------------------------------------------

#: Codes for rejections that are *not* a prohibited-target hit. Stable strings:
#: they are written into audit records and asserted on by other services, so they
#: are treated like the card's own ``mcp.errors.stable_codes``.
MALFORMED_EVENT = "MALFORMED_EVENT"
UNTRUSTED_TEXT_NOT_A_VALUE = "UNTRUSTED_TEXT_NOT_A_VALUE"
VALUE_OUT_OF_RANGE = "VALUE_OUT_OF_RANGE"
UNKNOWN_FEATURE = "UNKNOWN_FEATURE"
UNAUTHENTICATED_SOURCE = "UNAUTHENTICATED_SOURCE"
DUPLICATE_EVENT = "DUPLICATE_EVENT"


class EvidenceRejected(ValueError):
    """Structured refusal to admit an observation.

    Carries the stable ``code`` so the caller can record it verbatim instead of
    matching on message text.
    """

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class EvidenceRejection:
    """An auditable rejection — the payload of ``observation.rejected``."""

    code: str
    detail: str
    dedup_key: str | None = None
    matched_term: str | None = None
    match_kind: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "detail": self.detail,
            "dedup_key": self.dedup_key,
            "matched_term": self.matched_term,
            "match_kind": self.match_kind,
        }


# --------------------------------------------------------------------------
# canonical encoding (shared with inference and the compiler)
# --------------------------------------------------------------------------


def canonicalize(value: Any) -> Any:
    """Recursively put a document into the card's canonical form.

    ``rule30_vdf.canonical_input`` requires UTF-8 canonical JSON with
    lexicographic keys and normalised floats. Floats are rounded to
    ``STORAGE_PRECISION`` here for the same reason ``torx_layer.state`` rounds
    them: a value produced on the JAX path and the same value produced by the
    pure-Python fallback must serialise to identical bytes, or every attestation
    would be backend-specific.
    """
    if isinstance(value, Mapping):
        return {
            str(k): canonicalize(v)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (set, frozenset)):
        items = [canonicalize(v) for v in value]
        return sorted(items, key=lambda x: json.dumps(x, sort_keys=True))
    if isinstance(value, (list, tuple)):
        return [canonicalize(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float is not canonically encodable: {value!r}")
        return round(value, STORAGE_PRECISION)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError(f"naive datetime is not canonically encodable: {value!r}")
        return (
            value.astimezone(timezone.utc)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        )
    raise TypeError(f"{type(value).__name__} has no canonical encoding")


def canonical_json(document: Any) -> str:
    """The exact byte string the Rule 30 VDF and every content hash sign."""
    return json.dumps(
        canonicalize(document),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_hash(document: Any) -> str:
    """SHA-256 over :func:`canonical_json`, hex."""
    return hashlib.sha256(canonical_json(document).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# text normalisation used only by the denylist screen
# --------------------------------------------------------------------------

# Homoglyph substitutions an evader reaches for first. Folding them *before*
# matching means ``p0litical`` and ``political`` screen identically.
_LEET = str.maketrans(
    {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t",
     "8": "b", "$": "s", "@": "a", "|": "l", "!": "i"}
)

_NON_ALPHA = re.compile(r"[^a-z]+")


def _fold(raw: str) -> tuple[tuple[str, ...], str]:
    """Return ``(tokens, squashed)`` for denylist comparison.

    Unicode is decomposed and stripped to ASCII so ``pоlitical`` with a Cyrillic
    ``о`` folds to the Latin form; separators, digits and case are discarded so
    ``Political-Belief_2``, ``politicalBelief`` and ``political belief`` all
    reduce to the same tokens.
    """
    # Split camelCase first — after case folding the boundary is gone, and
    # ``sexualOrientation`` would become one unsplittable run.
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(raw))
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().translate(_LEET)
    tokens = tuple(t for t in _NON_ALPHA.sub(" ", text).split() if t)
    return tokens, "".join(tokens)


#: Tokens that appear in the card's own catalogue (or in obviously benign field
#: names) and must never be reported as a near miss. ``region`` and ``residency``
#: are the load-bearing entries: ``data_residency`` is a declared hard-boundary
#: category, and ``region`` is one edit away from ``religion``.
_SAFE_TOKENS = frozenset(
    {
        "region", "regions", "residency", "resident", "structure", "structured",
        "sequence", "sequencing", "preference", "preferences", "frequency",
        "threshold", "thresholds", "parallelism", "abstraction", "verification",
        "expression", "iteration", "artifact", "format", "challenge",
        "technicality", "verbosity", "planning", "depth", "conflict", "style",
        "consensus", "consent", "dissent", "bridge", "receptivity", "proposal",
        "execution", "confirmation", "autonomy", "communication", "workflow",
        "collaboration", "examples", "versus", "language", "goals",
        "accessibility", "requirements", "permission", "permissions",
        "expansion", "preserve", "boundary", "boundaries", "session", "urgency",
        "cognitive", "granted", "within", "explicit", "inferred", "profile",
    }
)

# Ratio above which two folded strings are treated as the same word. 0.88 is
# tight enough that ``region``/``religion`` (0.857) stays clear and loose enough
# that a single transposition or dropped letter still matches.
_NEAR_MISS_RATIO = 0.88
_MIN_NEAR_MISS_LEN = 5
_MIN_SQUASHED_LEN = 6


@dataclass(frozen=True, slots=True)
class ProhibitedTarget:
    """One ``prohibited_persistence`` category and how to recognise it."""

    code: str
    card_clause: str
    phrases: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.phrases:
            raise ValueError(f"{self.code}: a target with no phrases screens nothing")


#: The card's ten prohibited categories, each with a stable code. ``card_clause``
#: is the verbatim card string; :func:`assert_denylist_covers_card` checks the
#: two lists still line up, so adding a category to the card without adding a
#: screen here is a test failure rather than a silent hole.
PROHIBITED_TARGETS: tuple[ProhibitedTarget, ...] = (
    ProhibitedTarget(
        "PROHIBITED_CREDENTIALS",
        "credentials",
        ("credential", "credentials", "password", "passwd", "passphrase",
         "api key", "apikey", "client secret", "auth secret", "login secret"),
    ),
    ProhibitedTarget(
        "PROHIBITED_PRIVATE_KEY",
        "private keys",
        ("private key", "privkey", "secret key", "signing key", "keypair",
         "seed phrase", "mnemonic", "recovery phrase"),
    ),
    ProhibitedTarget(
        "PROHIBITED_ACCESS_TOKEN",
        "access tokens",
        ("access token", "bearer token", "refresh token", "session token",
         "oauth token", "id token", "jwt"),
    ),
    ProhibitedTarget(
        "PROHIBITED_RAW_CONTENT",
        "raw message bodies unless explicitly configured by tenant policy",
        ("raw message", "message body", "raw body", "raw text", "chat log",
         "transcript", "verbatim message"),
    ),
    ProhibitedTarget(
        "PROHIBITED_BIOMETRIC",
        "biometric identity templates",
        ("biometric", "faceprint", "face template", "fingerprint", "voiceprint",
         "iris scan", "retina scan", "gait signature"),
    ),
    ProhibitedTarget(
        "PROHIBITED_MEDICAL",
        "medical diagnosis inference",
        ("diagnosis", "diagnoses", "diagnostic", "medical condition",
         "mental illness", "disorder", "prescription", "medication",
         "health condition"),
    ),
    ProhibitedTarget(
        "PROHIBITED_PROTECTED_CLASS",
        "protected-class inference",
        ("protected class", "race", "racial", "ethnicity", "ethnic origin",
         "national origin", "caste", "immigration status", "disability status",
         "pregnancy status", "gender identity"),
    ),
    ProhibitedTarget(
        "PROHIBITED_POLITICAL",
        "political belief inference",
        ("political", "politics", "political belief", "party affiliation",
         "partisan lean", "voting intention", "ideology"),
    ),
    ProhibitedTarget(
        "PROHIBITED_RELIGIOUS",
        "religious belief inference",
        ("religion", "religious", "religious belief", "faith tradition",
         "denomination", "observance level"),
    ),
    ProhibitedTarget(
        "PROHIBITED_SEXUAL_ORIENTATION",
        "sexual orientation inference",
        ("sexual orientation", "sexuality", "sexual preference", "lgbt",
         "lgbtq", "homosexual", "heterosexual", "bisexual"),
    ),
)


@dataclass(frozen=True, slots=True)
class ProhibitedMatch:
    """Why a name was refused, in a form an audit record can quote."""

    code: str
    card_clause: str
    phrase: str
    match_kind: str  # "exact" | "squashed" | "near-miss"
    fragment: str

    def as_rejection(self, dedup_key: str | None = None) -> EvidenceRejection:
        return EvidenceRejection(
            code=self.code,
            detail=(
                f"{self.fragment!r} names a prohibited-inference target "
                f"({self.card_clause!r}) via {self.match_kind} match on "
                f"{self.phrase!r}"
            ),
            dedup_key=dedup_key,
            matched_term=self.phrase,
            match_kind=self.match_kind,
        )


def _contains_ngram(tokens: Sequence[str], phrase_tokens: Sequence[str]) -> bool:
    n = len(phrase_tokens)
    return any(
        tuple(tokens[i : i + n]) == tuple(phrase_tokens)
        for i in range(len(tokens) - n + 1)
    )


def _near(a: str, b: str) -> bool:
    return SequenceMatcher(None, a, b).ratio() >= _NEAR_MISS_RATIO


def screen_fragment(fragment: str) -> ProhibitedMatch | None:
    """Screen one identifier against every prohibited category.

    Three passes, cheapest first: exact token n-gram, separator-stripped
    containment, then bounded fuzzy comparison. The fuzzy pass skips tokens in
    :data:`_SAFE_TOKENS` — those are names the card itself uses, and a guard that
    rejects the card's own feature catalogue is worse than no guard, because it
    would push operators to disable it.
    """
    tokens, squashed = _fold(fragment)
    if not tokens:
        return None
    for target in PROHIBITED_TARGETS:
        for phrase in target.phrases:
            p_tokens, p_squashed = _fold(phrase)
            if _contains_ngram(tokens, p_tokens):
                return ProhibitedMatch(
                    target.code, target.card_clause, phrase, "exact", fragment
                )
            if len(p_squashed) >= _MIN_SQUASHED_LEN and p_squashed in squashed:
                return ProhibitedMatch(
                    target.code, target.card_clause, phrase, "squashed", fragment
                )
    for target in PROHIBITED_TARGETS:
        for phrase in target.phrases:
            p_tokens, p_squashed = _fold(phrase)
            if len(p_tokens) == 1:
                if len(p_squashed) < _MIN_NEAR_MISS_LEN:
                    continue
                for tok in tokens:
                    if len(tok) < _MIN_NEAR_MISS_LEN or tok in _SAFE_TOKENS:
                        continue
                    if _near(tok, p_squashed):
                        return ProhibitedMatch(
                            target.code, target.card_clause, phrase,
                            "near-miss", fragment,
                        )
                continue
            # Multi-word phrase: compare same-length windows of the folded
            # tokens, so one typo anywhere inside the phrase still matches.
            n = len(p_tokens)
            for i in range(len(tokens) - n + 1):
                window = "".join(tokens[i : i + n])
                if len(window) >= _MIN_SQUASHED_LEN and _near(window, p_squashed):
                    return ProhibitedMatch(
                        target.code, target.card_clause, phrase, "near-miss", fragment
                    )
    return None


def screen(*fragments: str) -> ProhibitedMatch | None:
    """First prohibited match across several identifiers, or ``None``."""
    for fragment in fragments:
        hit = screen_fragment(fragment)
        if hit is not None:
            return hit
    return None


def assert_denylist_covers_card(card: AgentModelCard | None = None) -> None:
    """Fail loudly when the card declares a category nothing here screens."""
    card = card or cached_model_card()
    declared = set(card.personalization.prohibited_persistence)
    screened = {t.card_clause for t in PROHIBITED_TARGETS}
    missing = declared - screened
    if missing:
        raise ValueError(
            "personalization.prohibited_persistence declares categories with no "
            f"denylist screen: {sorted(missing)}"
        )
    stale = screened - declared
    if stale:
        raise ValueError(
            f"denylist screens categories the card no longer declares: {sorted(stale)}"
        )


# --------------------------------------------------------------------------
# provenance: free text, structurally unable to become a value
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProvenanceNote:
    """An opaque, non-writable note attached to an observation.

    Deliberately anaemic. It exposes a digest, a length and a label — enough for
    ``profile.explain`` to say "this came from a tool result at 14:02 whose body
    hashed to ``a1b2…``" and enough for an operator to correlate with their own
    logs, and nothing more. There is no accessor that yields a number, so no
    amount of downstream code can route this into a profile field.

    ``text`` is accepted at construction and **discarded** unless
    ``tenant_allows_raw_text`` is set, matching the card's
    "raw message bodies unless explicitly configured by tenant policy".
    """

    label: str
    digest: str
    char_count: int
    retained_text: str | None = None

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("provenance note requires a label naming its origin")
        if self.char_count < 0:
            raise ValueError(f"char_count must be >= 0, got {self.char_count}")

    @classmethod
    def from_untrusted_text(
        cls, text: str, *, label: str, tenant_allows_raw_text: bool = False
    ) -> ProvenanceNote:
        body = str(text)
        return cls(
            label=label,
            digest=hashlib.sha256(body.encode("utf-8")).hexdigest(),
            char_count=len(body),
            retained_text=body if tenant_allows_raw_text else None,
        )

    def to_json(self) -> dict[str, Any]:
        """Never emits the body unless tenant policy retained it."""
        doc: dict[str, Any] = {
            "label": self.label,
            "digest": self.digest,
            "char_count": self.char_count,
            "raw_retained": self.retained_text is not None,
        }
        if self.retained_text is not None:
            doc["text"] = self.retained_text
        return doc


# --------------------------------------------------------------------------
# feature catalogue
# --------------------------------------------------------------------------

#: Domains whose feature names are user-owned rather than drawn from the card's
#: inferred-feature catalogue. Names here are free-form — and therefore screened
#: against the denylist exactly like every other name.
EXPLICIT_DOMAIN = "explicit"
HARD_BOUNDARY_DOMAIN = "hard_boundaries"
OPEN_NAME_DOMAINS = (EXPLICIT_DOMAIN, HARD_BOUNDARY_DOMAIN)


def _canonical_name(raw: str) -> str:
    """``examples-versus-abstraction`` -> ``examples_versus_abstraction``.

    The card writes feature names with hyphens in ``feature_domains`` and with
    underscores in ``jsonb_contracts.user_profile.example``. One spelling has to
    win for the JSONB key; the example's underscore form does, because that is
    the shape already published as the storage contract.
    """
    return str(raw).strip().replace("-", "_").replace(" ", "_")


def feature_catalog(card: AgentModelCard | None = None) -> dict[str, frozenset[str]]:
    """``{domain: {feature names}}`` from ``personalization.feature_domains``."""
    card = card or cached_model_card()
    return {
        _canonical_name(domain): frozenset(_canonical_name(n) for n in names)
        for domain, names in card.personalization.feature_domains.items()
    }


@dataclass(frozen=True, slots=True)
class FeatureObservation:
    """One range-checked, typed measurement of one catalogued feature.

    This is the *only* type a profile writer accepts. ``value`` is a float on
    ``[0, 1]`` because every feature the card declares is a bounded preference
    scalar; symbolic settings (language, declared goals) are user-owned and
    arrive through ``profile.correct``, not through inference, so no parser here
    ever has to turn a string into stored state.
    """

    domain: str
    name: str
    value: float
    value_type: str = "scalar"  # "scalar" | "boolean"

    def __post_init__(self) -> None:
        domain = _canonical_name(self.domain)
        name = _canonical_name(self.name)
        if not domain or not name:
            raise EvidenceRejected(
                MALFORMED_EVENT, "observation requires both a domain and a name"
            )
        if self.value_type not in ("scalar", "boolean"):
            raise EvidenceRejected(
                MALFORMED_EVENT,
                f"value_type must be 'scalar' or 'boolean', got {self.value_type!r}",
            )
        hit = screen(domain, name)
        if hit is not None:
            raise EvidenceRejected(hit.code, hit.as_rejection().detail)
        catalog = feature_catalog()
        if domain in catalog:
            if name not in catalog[domain]:
                raise EvidenceRejected(
                    UNKNOWN_FEATURE,
                    f"{domain}.{name} is not in the card's feature catalogue "
                    f"({sorted(catalog[domain])})",
                )
        elif domain not in OPEN_NAME_DOMAINS:
            raise EvidenceRejected(
                UNKNOWN_FEATURE,
                f"unknown feature domain {domain!r}; expected one of "
                f"{sorted(set(catalog) | set(OPEN_NAME_DOMAINS))}",
            )
        if isinstance(self.value, bool):
            # ``True`` is an ``int`` in Python. A boolean must come through
            # :meth:`boolean` so ``value_type`` records it and the profile
            # renders it as a boolean rather than as 1.0.
            raise EvidenceRejected(
                MALFORMED_EVENT,
                f"{domain}.{name}: build boolean observations with "
                "FeatureObservation.boolean() so the value type is recorded",
            )
        if not isinstance(self.value, (int, float)):
            raise EvidenceRejected(
                UNTRUSTED_TEXT_NOT_A_VALUE,
                f"observation value must be a real number, got "
                f"{type(self.value).__name__}: only typed, range-checked "
                "measurements may reach a profile field",
            )
        value = float(self.value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise EvidenceRejected(
                VALUE_OUT_OF_RANGE,
                f"{domain}.{name} = {self.value!r} is outside the declared "
                "[0, 1] range",
            )
        if self.value_type == "boolean" and value not in (0.0, 1.0):
            raise EvidenceRejected(
                VALUE_OUT_OF_RANGE,
                f"{domain}.{name} is boolean but carries {value!r}",
            )
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "value", round(value, STORAGE_PRECISION))

    @classmethod
    def boolean(cls, domain: str, name: str, flag: bool) -> FeatureObservation:
        return cls(domain, name, 1.0 if flag else 0.0, "boolean")

    @property
    def key(self) -> str:
        """``domain.name`` — the profile's feature key."""
        return f"{self.domain}.{self.name}"

    @property
    def is_boundary(self) -> bool:
        return self.domain == HARD_BOUNDARY_DOMAIN

    def rendered_value(self) -> Any:
        return bool(self.value) if self.value_type == "boolean" else self.value

    def to_json(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "name": self.name,
            "value": self.value,
            "value_type": self.value_type,
        }


# --------------------------------------------------------------------------
# source and event
# --------------------------------------------------------------------------

#: Where an observation came from. ``untrusted-text`` exists so that a caller can
#: *name* the untrusted path honestly; it is then refused, rather than the caller
#: having to omit the field and the guard having to guess.
SOURCE_CHANNELS: tuple[str, ...] = (
    "user-declaration",
    "tool-result",
    "system-telemetry",
    "third-party",
    "untrusted-text",
)

EVIDENCE_KINDS: tuple[str, ...] = (
    "explicit-declaration",
    "explicit-boundary",
    "user-correction",
    "observed-behavior",
    "tool-outcome",
    "session-signal",
)

#: Kinds carrying the user's own word. ``profile_layers.explicit`` gives these
#: "highest-user-owned" authority, which the contradiction policy then uses.
EXPLICIT_KINDS: frozenset[str] = frozenset(
    {"explicit-declaration", "explicit-boundary", "user-correction"}
)


@dataclass(frozen=True, slots=True)
class EvidenceSource:
    """Attribution for one observation.

    ``trust`` is the card's ``poisoning.evidence_source_weighting``: telemetry
    the node produced itself is worth more than a third-party assertion, and the
    weight multiplies into the confidence the inference layer thresholds on.
    """

    source_id: str
    channel: str
    trust: float = 1.0
    authenticated: bool = True

    def __post_init__(self) -> None:
        if not str(self.source_id).strip():
            raise EvidenceRejected(
                MALFORMED_EVENT,
                "source_id is required: 'source evidence is attributable' is a "
                "precondition of automatic mutation",
            )
        if self.channel not in SOURCE_CHANNELS:
            raise EvidenceRejected(
                MALFORMED_EVENT,
                f"unknown source channel {self.channel!r}; expected one of "
                f"{list(SOURCE_CHANNELS)}",
            )
        if not math.isfinite(self.trust) or not 0.0 <= self.trust <= 1.0:
            raise EvidenceRejected(
                MALFORMED_EVENT, f"source trust must be in [0, 1], got {self.trust!r}"
            )
        object.__setattr__(self, "trust", round(float(self.trust), STORAGE_PRECISION))

    def to_json(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "channel": self.channel,
            "trust": self.trust,
            "authenticated": self.authenticated,
        }


@dataclass(frozen=True, slots=True)
class EvidenceEvent:
    """One eligible interaction event, already reduced to typed features.

    ``implied_capability_grants`` exists so that an attempt to widen the
    permission envelope is *representable and therefore refusable*. If the type
    could not express "this evidence suggests granting ``bridge:apply``", the
    request would arrive as an untyped side effect somewhere further downstream
    where nothing checks it. Here the inference layer rejects any non-empty grant
    outright, while revocations — narrowing — are allowed through.
    """

    tenant_id: str
    user_id: str
    event_id: str
    observed_at: datetime
    kind: str
    observation: FeatureObservation
    source: EvidenceSource
    source_confidence: float
    provenance: tuple[ProvenanceNote, ...] = ()
    implied_capability_grants: frozenset[str] = frozenset()
    implied_capability_revocations: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        for field_name in ("tenant_id", "user_id", "event_id"):
            if not str(getattr(self, field_name)).strip():
                raise EvidenceRejected(
                    MALFORMED_EVENT,
                    f"{field_name} is required to form the deduplication key "
                    "tenant_id:user_id:event_id",
                )
        if self.kind not in EVIDENCE_KINDS:
            raise EvidenceRejected(
                MALFORMED_EVENT,
                f"unknown evidence kind {self.kind!r}; expected one of "
                f"{list(EVIDENCE_KINDS)}",
            )
        if not isinstance(self.observed_at, datetime) or self.observed_at.tzinfo is None:
            raise EvidenceRejected(
                MALFORMED_EVENT,
                "observed_at must be a timezone-aware datetime; a naive "
                "timestamp cannot be compared against a decay half-life",
            )
        if self.source.channel == "untrusted-text":
            raise EvidenceRejected(
                UNTRUSTED_TEXT_NOT_A_VALUE,
                "a source on the 'untrusted-text' channel may never produce a "
                "feature observation; carry it as a ProvenanceNote instead",
            )
        if self.kind in EXPLICIT_KINDS and not self.source.authenticated:
            raise EvidenceRejected(
                UNAUTHENTICATED_SOURCE,
                f"kind {self.kind!r} asserts a user-owned value but the source "
                "is not authenticated",
            )
        if not math.isfinite(self.source_confidence) or not (
            0.0 <= self.source_confidence <= 1.0
        ):
            raise EvidenceRejected(
                MALFORMED_EVENT,
                f"source_confidence must be in [0, 1], got "
                f"{self.source_confidence!r}",
            )
        grants = frozenset(str(c) for c in self.implied_capability_grants)
        revocations = frozenset(str(c) for c in self.implied_capability_revocations)
        overlap = grants & revocations
        if overlap:
            raise EvidenceRejected(
                MALFORMED_EVENT,
                f"capabilities both granted and revoked: {sorted(overlap)}",
            )
        object.__setattr__(
            self, "source_confidence", round(float(self.source_confidence), STORAGE_PRECISION)
        )
        object.__setattr__(self, "provenance", tuple(self.provenance))
        object.__setattr__(self, "implied_capability_grants", grants)
        object.__setattr__(self, "implied_capability_revocations", revocations)

    @property
    def dedup_key(self) -> str:
        """``inference_update.event_deduplication_key``, verbatim."""
        return f"{self.tenant_id}:{self.user_id}:{self.event_id}"

    @property
    def effective_confidence(self) -> float:
        """Source confidence weighted by source trust.

        The product, not the maximum: a highly self-assured assertion from a
        barely-trusted source must not clear the persistence threshold on its own
        confidence alone. This is the number the inference layer compares against
        the card's thresholds.
        """
        return round(self.source_confidence * self.source.trust, STORAGE_PRECISION)

    @property
    def is_explicit(self) -> bool:
        return self.kind in EXPLICIT_KINDS

    def to_json(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "event_id": self.event_id,
            "dedup_key": self.dedup_key,
            "observed_at": self.observed_at,
            "kind": self.kind,
            "observation": self.observation.to_json(),
            "source": self.source.to_json(),
            "source_confidence": self.source_confidence,
            "effective_confidence": self.effective_confidence,
            "provenance": [p.to_json() for p in self.provenance],
            "implied_capability_grants": sorted(self.implied_capability_grants),
            "implied_capability_revocations": sorted(self.implied_capability_revocations),
        }


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

#: Payload keys whose contents are free text. They are folded into provenance
#: notes and can never reach ``FeatureObservation.value``.
_TEXT_KEYS = ("note", "text", "message", "utterance", "rationale", "excerpt")


def parse_evidence(
    payload: Mapping[str, Any], *, tenant_allows_raw_text: bool = False
) -> EvidenceEvent:
    """Build an :class:`EvidenceEvent` from a structured payload.

    Raises :class:`EvidenceRejected` with a stable code. Free-text keys in the
    payload become :class:`ProvenanceNote` instances; a payload that tries to
    supply text *as the value* is refused rather than parsed, because "parse the
    number out of the sentence" is precisely the step the prompt-injection
    control forbids.
    """
    if not isinstance(payload, Mapping):
        raise EvidenceRejected(
            MALFORMED_EVENT, f"payload must be a mapping, got {type(payload).__name__}"
        )
    try:
        feature = payload["feature"]
        domain = feature["domain"]
        name = feature["name"]
        raw_value = feature["value"]
    except (KeyError, TypeError) as exc:
        raise EvidenceRejected(
            MALFORMED_EVENT,
            "payload requires feature.{domain,name,value}; missing "
            f"{exc.args[0] if exc.args else exc}",
        ) from exc

    if isinstance(raw_value, str):
        raise EvidenceRejected(
            UNTRUSTED_TEXT_NOT_A_VALUE,
            f"feature.value arrived as text ({raw_value!r}); a profile field is "
            "only writable from a typed, range-checked measurement",
        )
    value_type = str(feature.get("value_type", "scalar"))
    if isinstance(raw_value, bool):
        observation = FeatureObservation.boolean(domain, name, raw_value)
    else:
        observation = FeatureObservation(domain, name, raw_value, value_type)

    source_doc = payload.get("source")
    if not isinstance(source_doc, Mapping):
        raise EvidenceRejected(
            MALFORMED_EVENT, "payload requires a 'source' mapping (attributability)"
        )
    source = EvidenceSource(
        source_id=str(source_doc.get("source_id", "")),
        channel=str(source_doc.get("channel", "")),
        trust=float(source_doc.get("trust", 1.0)),
        authenticated=bool(source_doc.get("authenticated", True)),
    )

    observed_at = payload.get("observed_at")
    if isinstance(observed_at, str):
        try:
            observed_at = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise EvidenceRejected(
                MALFORMED_EVENT, f"observed_at is not ISO-8601: {observed_at!r}"
            ) from exc

    notes = [
        ProvenanceNote.from_untrusted_text(
            str(payload[key]), label=key, tenant_allows_raw_text=tenant_allows_raw_text
        )
        for key in _TEXT_KEYS
        if payload.get(key) is not None
    ]

    return EvidenceEvent(
        tenant_id=str(payload.get("tenant_id", "")),
        user_id=str(payload.get("user_id", "")),
        event_id=str(payload.get("event_id", "")),
        observed_at=observed_at,
        kind=str(payload.get("kind", "")),
        observation=observation,
        source=source,
        source_confidence=float(payload.get("source_confidence", 0.0)),
        provenance=tuple(notes),
        implied_capability_grants=frozenset(
            payload.get("implied_capability_grants", ()) or ()
        ),
        implied_capability_revocations=frozenset(
            payload.get("implied_capability_revocations", ()) or ()
        ),
    )


def try_parse_evidence(
    payload: Mapping[str, Any], *, tenant_allows_raw_text: bool = False
) -> tuple[EvidenceEvent | None, EvidenceRejection | None]:
    """Non-raising :func:`parse_evidence`, for stream consumers.

    A rejected event must still be *recorded* — ``observability.audit_events``
    lists ``personalization.observation.rejected`` — so the rejection is returned
    as data rather than as an exception the consumer might swallow.
    """
    try:
        return parse_evidence(payload, tenant_allows_raw_text=tenant_allows_raw_text), None
    except EvidenceRejected as exc:
        key = None
        if isinstance(payload, Mapping):
            parts = [payload.get("tenant_id"), payload.get("user_id"), payload.get("event_id")]
            if all(p is not None for p in parts):
                key = ":".join(str(p) for p in parts)
        return None, EvidenceRejection(code=exc.code, detail=exc.detail, dedup_key=key)


# --------------------------------------------------------------------------
# deduplication
# --------------------------------------------------------------------------


@dataclass(slots=True)
class EvidenceLedger:
    """Append-only admission log keyed on ``tenant_id:user_id:event_id``.

    Deliberately not a cache with eviction: ``runtime.event_model`` promises
    at-least-once delivery and idempotent consumers, and an evicting dedup window
    silently stops being idempotent exactly when redelivery is most likely (after
    a long outage). Bounding this set is a persistence concern, handled by the
    revision table's own uniqueness constraint, not by forgetting here.
    """

    _seen: dict[str, datetime] = field(default_factory=dict)

    def admit(self, event: EvidenceEvent) -> bool:
        """True when the event is new; False when it is a redelivery."""
        if event.dedup_key in self._seen:
            return False
        self._seen[event.dedup_key] = event.observed_at
        return True

    def rejection_for(self, event: EvidenceEvent) -> EvidenceRejection:
        return EvidenceRejection(
            code=DUPLICATE_EVENT,
            detail=(
                f"{event.dedup_key} was already applied; at-least-once delivery "
                "means a redelivery must not move the profile a second time"
            ),
            dedup_key=event.dedup_key,
        )

    def __contains__(self, key: object) -> bool:
        return str(key) in self._seen

    def __len__(self) -> int:
        return len(self._seen)

    def keys(self) -> Iterator[str]:
        return iter(self._seen)


def deduplicate(
    events: Iterable[EvidenceEvent],
) -> tuple[tuple[EvidenceEvent, ...], tuple[EvidenceRejection, ...]]:
    """Split a batch into first-seen events and duplicate rejections."""
    ledger = EvidenceLedger()
    accepted: list[EvidenceEvent] = []
    rejected: list[EvidenceRejection] = []
    for event in events:
        if ledger.admit(event):
            accepted.append(event)
        else:
            rejected.append(ledger.rejection_for(event))
    return tuple(accepted), tuple(rejected)
