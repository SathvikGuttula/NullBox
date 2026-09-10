"""
Contextual fraud signals — everything about the call that is not the audio.

A voice-clone attack is rarely just a voice. It arrives from an unknown number,
at an odd hour, asking for an urgent transfer to a beneficiary nobody has paid
before, from a device the account has never seen. Each of those is weak on its
own and unremarkable in isolation; together they are the difference between a
model that flags synthetic speech and a system that detects fraud.

This is the branch that lets VoxShield escalate a *genuine human* voice — the
attacker who impersonates by social engineering rather than by synthesis, and
whom neither anti-spoof nor speaker verification will ever flag if they are
calling about someone else's account.

Two disciplines carried over from the rest of the project
---------------------------------------------------------

**Unknown is not innocent, and it is not guilty either.** Every signal is
optional. A signal that was never collected contributes nothing and is listed
in ``missing`` rather than being scored as zero risk, because "we did not check
whether the device was known" and "the device was known" are very different
statements and must not produce the same number.

**The weights are placeholders and say so.** They encode ordinary fraud-analyst
intuition, not anything fitted to data, and they carry a version string that
travels with each assessment. Fit them when there is labelled fraud data to fit
them on; until then the structure is the contribution, not the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar


def _type_name(expected) -> str:
    if isinstance(expected, tuple):
        return " or ".join(t.__name__ for t in expected)
    return expected.__name__


@dataclass
class CallContext:
    """
    What is known about a call besides its audio.

    Every field is optional. ``None`` means "not collected", which is treated
    differently from a known-safe value.
    """

    # -- who is calling ----------------------------------------------------
    caller_known: bool | None = None
    trusted_contact: bool | None = None
    caller_country: str | None = None
    expected_country: str | None = None

    # -- how they are calling ---------------------------------------------
    device_known: bool | None = None
    recent_password_reset: bool | None = None
    failed_authentication_count: int | None = None

    # -- when -------------------------------------------------------------
    hour_of_day: int | None = None          # 0-23, local to the account
    outside_business_hours: bool | None = None

    # -- what they are asking for -----------------------------------------
    transaction_amount: float | None = None
    typical_transaction_amount: float | None = None
    beneficiary_known: bool | None = None
    beneficiary_age_days: int | None = None

    def known_signals(self) -> int:
        return sum(1 for value in vars(self).values() if value is not None)

    # Field -> (accepted python types, optional inclusive range).
    #
    # A dataclass does not check types, so CallContext(transaction_amount="lots")
    # constructs happily and then fails several layers away - in the policy
    # engine, comparing a string to a float, as a 500 with no useful message.
    # The boundary where untrusted JSON arrives is the place to catch that.
    _FIELD_RULES: ClassVar[dict] = {
        "caller_known": (bool, None),
        "trusted_contact": (bool, None),
        "caller_country": (str, None),
        "expected_country": (str, None),
        "device_known": (bool, None),
        "recent_password_reset": (bool, None),
        "failed_authentication_count": (int, (0, 10_000)),
        "hour_of_day": (int, (0, 23)),
        "outside_business_hours": (bool, None),
        "transaction_amount": ((int, float), (0, 1e15)),
        "typical_transaction_amount": ((int, float), (0, 1e15)),
        "beneficiary_known": (bool, None),
        "beneficiary_age_days": (int, (0, 100_000)),
    }

    @classmethod
    def from_dict(cls, fields: dict) -> "CallContext":
        """
        Build from untrusted JSON, or raise ``ValueError`` saying exactly why.

        Unknown keys are rejected rather than ignored: a caller who sends
        ``{"caller_is_known": false}`` and gets a 200 back would reasonably
        assume the signal was counted, and it was not.
        """

        if not isinstance(fields, dict):
            raise ValueError("context must be a JSON object")

        known = set(cls._FIELD_RULES)
        unknown = set(fields) - known

        if unknown:
            raise ValueError(
                f"unknown context fields: {sorted(unknown)}. "
                f"Accepted: {sorted(known)}"
            )

        cleaned: dict = {}

        for name, value in fields.items():
            if value is None:
                continue

            expected, bounds = cls._FIELD_RULES[name]

            # bool is a subclass of int in Python, so an unguarded isinstance
            # check would let True through as failed_authentication_count.
            if expected is not bool and isinstance(value, bool):
                raise ValueError(
                    f"{name} must be {_type_name(expected)}, got a boolean"
                )

            if not isinstance(value, expected):
                raise ValueError(
                    f"{name} must be {_type_name(expected)}, got "
                    f"{type(value).__name__} ({value!r})"
                )

            if bounds is not None and not bounds[0] <= value <= bounds[1]:
                raise ValueError(
                    f"{name} must be between {bounds[0]} and {bounds[1]}, got {value}"
                )

            cleaned[name] = value

        return cls(**cleaned)


@dataclass
class ContextWeights:
    """
    Points each signal can add to the context risk, out of 100.

    Placeholders. Analyst intuition, not fitted values.
    """

    unknown_caller: float = 14.0
    untrusted_contact: float = 8.0
    unexpected_country: float = 12.0

    unknown_device: float = 12.0
    recent_password_reset: float = 14.0
    failed_authentications: float = 12.0

    odd_hour: float = 8.0

    unusual_amount: float = 14.0
    unknown_beneficiary: float = 12.0
    new_beneficiary: float = 10.0

    version: str = "v0-UNCALIBRATED"

    @property
    def calibrated(self) -> bool:
        return "UNCALIBRATED" not in self.version


@dataclass
class ContextAssessment:
    risk: float                      # 0-1, for the fusion layer
    points: float                    # raw points scored
    possible: float                  # points that could have been scored
    reasons: list[str] = field(default_factory=list)
    signals_used: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    version: str = "v0-UNCALIBRATED"

    def to_dict(self) -> dict:
        return {
            "risk": round(self.risk, 4),
            "points": round(self.points, 1),
            "possible": round(self.possible, 1),
            "reasons": self.reasons,
            "signals_used": self.signals_used,
            "missing": self.missing,
            "version": self.version,
            "calibrated": "UNCALIBRATED" not in self.version,
        }


class ContextAnalyzer:
    """
    Turn call metadata into a risk in [0, 1] with an explanation per signal.

    The score is the fraction of *available* risk that was actually triggered,
    not a fraction of all conceivable risk. A call where only two signals were
    collected and both look bad should read as high context risk; dividing by
    the full weight of ten signals would dilute it to nearly nothing and hide
    the very thing the operator needs to see. ``signals_used`` and ``missing``
    are reported so the denominator is never a mystery.
    """

    def __init__(self, weights: ContextWeights | None = None) -> None:
        self.weights = weights or ContextWeights()

    def assess(self, context: CallContext) -> ContextAssessment:
        weights = self.weights

        points = 0.0
        possible = 0.0
        reasons: list[str] = []
        used: list[str] = []
        missing: list[str] = []

        def score(name: str, value, weight: float, triggered, describe) -> None:
            nonlocal points, possible

            if value is None:
                missing.append(name)
                return

            possible += weight
            used.append(name)

            if triggered(value):
                points += weight
                reasons.append(describe(value))

        # -- who ------------------------------------------------------------

        score(
            "caller_known", context.caller_known, weights.unknown_caller,
            lambda v: not v,
            lambda v: "Caller number is not recognised",
        )
        score(
            "trusted_contact", context.trusted_contact, weights.untrusted_contact,
            lambda v: not v,
            lambda v: "Caller is not on the account's trusted-contact list",
        )

        # Country only carries information when there is an expectation to
        # compare against - "called from Ireland" is meaningless alone.
        if context.caller_country is not None and context.expected_country is not None:
            possible += weights.unexpected_country
            used.append("caller_country")
            if context.caller_country.upper() != context.expected_country.upper():
                points += weights.unexpected_country
                reasons.append(
                    f"Call originates from {context.caller_country}, "
                    f"not the expected {context.expected_country}"
                )
        else:
            missing.append("caller_country")

        # -- how ------------------------------------------------------------

        score(
            "device_known", context.device_known, weights.unknown_device,
            lambda v: not v,
            lambda v: "Call is from a device the account has not used before",
        )
        score(
            "recent_password_reset", context.recent_password_reset,
            weights.recent_password_reset,
            lambda v: bool(v),
            lambda v: "Account password was reset recently",
        )
        score(
            "failed_authentication_count", context.failed_authentication_count,
            weights.failed_authentications,
            lambda v: v >= 2,
            lambda v: f"{v} failed authentication attempts on this account",
        )

        # -- when -----------------------------------------------------------

        if context.outside_business_hours is not None:
            score(
                "outside_business_hours", context.outside_business_hours,
                weights.odd_hour,
                lambda v: bool(v),
                lambda v: "Call is outside normal business hours",
            )
        elif context.hour_of_day is not None:
            possible += weights.odd_hour
            used.append("hour_of_day")
            if context.hour_of_day < 7 or context.hour_of_day >= 21:
                points += weights.odd_hour
                reasons.append(
                    f"Call placed at {context.hour_of_day:02d}:00, outside "
                    f"normal hours"
                )
        else:
            missing.append("hour_of_day")

        # -- what -----------------------------------------------------------

        if (
            context.transaction_amount is not None
            and context.typical_transaction_amount is not None
            and context.typical_transaction_amount > 0
        ):
            possible += weights.unusual_amount
            used.append("transaction_amount")

            ratio = context.transaction_amount / context.typical_transaction_amount
            if ratio >= 5.0:
                points += weights.unusual_amount
                reasons.append(
                    f"Requested amount is {ratio:.0f}x the account's typical "
                    f"transaction"
                )
            elif ratio >= 2.0:
                points += weights.unusual_amount * 0.5
                reasons.append(
                    f"Requested amount is {ratio:.1f}x the account's typical "
                    f"transaction"
                )
        else:
            missing.append("transaction_amount")

        score(
            "beneficiary_known", context.beneficiary_known,
            weights.unknown_beneficiary,
            lambda v: not v,
            lambda v: "Payment is to a beneficiary this account has never paid",
        )
        score(
            "beneficiary_age_days", context.beneficiary_age_days,
            weights.new_beneficiary,
            lambda v: v <= 7,
            lambda v: (
                "Beneficiary was added today"
                if v <= 0
                else f"Beneficiary was added {v} day(s) ago"
            ),
        )

        risk = points / possible if possible > 0 else 0.0

        if not used:
            reasons.append("No call context was provided")

        return ContextAssessment(
            risk=min(max(risk, 0.0), 1.0),
            points=points,
            possible=possible,
            reasons=reasons,
            signals_used=used,
            missing=missing,
            version=weights.version,
        )
