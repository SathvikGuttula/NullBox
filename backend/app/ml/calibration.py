class ProbabilityCalibrator:

    def __init__(
        self,
        temperature: float = 1.0,
    ):
        self.temperature = max(
            temperature,
            1e-6,
        )

    def calibrate(
        self,
        probability: float,
    ) -> float:

        probability = min(
            max(probability, 0.0),
            1.0,
        )

        return probability