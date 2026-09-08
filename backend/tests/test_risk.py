from app.ml.risk import HIGH, LOW, SUSPICIOUS, RiskThresholds, TemporalRiskEngine


def test_score_is_bounded():
    engine = TemporalRiskEngine()

    result = engine.update(0.95)

    assert 0 <= result["risk_score"] <= 100


def test_low_probability_stays_low():
    engine = TemporalRiskEngine()

    for _ in range(20):
        result = engine.update(0.10)

    assert result["risk_level"] == LOW


def test_sustained_high_probability_escalates():
    engine = TemporalRiskEngine()

    for _ in range(20):
        result = engine.update(0.97)

    assert result["risk_level"] == HIGH
    assert result["risk_score"] > 85


def test_a_single_spike_does_not_escalate():
    """
    One bad window must not trip the alarm.

    A cough, a codec glitch or a moment of clipping can spike a single
    window. Escalating on that is what makes a fraud system get switched off.
    """

    engine = TemporalRiskEngine()

    for _ in range(5):
        engine.update(0.05)

    result = engine.update(1.0)

    assert result["risk_level"] == LOW


def test_hysteresis_prevents_flapping():
    """
    A score hovering at the boundary must not flip the level every window.

    Once SUSPICIOUS, the score has to fall a clear margin below the boundary
    before it downgrades.
    """

    thresholds = RiskThresholds(suspicious=60.0, high=85.0, margin=5.0)
    engine = TemporalRiskEngine(smoothing_alpha=1.0, thresholds=thresholds)

    engine.update(0.65)
    assert engine.level == SUSPICIOUS

    # 0.58 -> 58, below the 60 boundary but inside the 5-point margin.
    engine.update(0.58)
    assert engine.level == SUSPICIOUS

    engine.update(0.50)
    assert engine.level == LOW


def test_reasons_are_populated():
    engine = TemporalRiskEngine()

    for _ in range(10):
        result = engine.update(0.95)

    assert result["reasons"]
    assert any("synthetic" in reason.lower() for reason in result["reasons"])


def test_quiet_call_explains_itself():
    engine = TemporalRiskEngine()

    result = engine.update(0.02)

    assert result["reasons"] == ["No synthetic-speech indicators detected"]


def test_out_of_range_probabilities_are_clamped():
    engine = TemporalRiskEngine()

    assert engine.update(5.0)["raw_probability"] == 1.0
    assert engine.update(-2.0)["raw_probability"] == 0.0


def test_reset_returns_to_the_initial_state():
    engine = TemporalRiskEngine()

    for _ in range(10):
        engine.update(0.99)

    engine.reset()

    assert engine.smoothed == 0.0
    assert engine.level == LOW
    assert engine.update(0.1)["samples_seen"] == 1
