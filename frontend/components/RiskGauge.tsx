
"use client";

interface RiskGaugeProps {
    score: number;
}

export default function RiskGauge({
    score,
}: RiskGaugeProps) {
    const getLabel = () => {
        if (score <= 25) return "SAFE";
        if (score <= 50) return "LOW";
        if (score <= 70) return "SUSPICIOUS";
        if (score <= 85) return "HIGH";
        return "CRITICAL";
    };

    return (
        <div
            style={{
                width: 240,
                height: 240,
                borderRadius: "50%",
                border: "8px solid #1f2937",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                flexDirection: "column",
                margin: "0 auto",
            }}
        >
            <div
                style={{
                    fontSize: 56,
                    fontWeight: 800,
                }}
            >
                {score}
            </div>

            <div
                style={{
                    fontSize: 14,
                    letterSpacing: 2,
                    fontWeight: 700,
                }}
            >
                {getLabel()}
            </div>
        </div>
    );
}