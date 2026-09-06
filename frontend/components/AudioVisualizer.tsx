"use client";

interface AudioVisualizerProps {
  level: number;
  active: boolean;
}

export default function AudioVisualizer({
  level,
  active,
}: AudioVisualizerProps) {

  const normalized = Math.min(
    Math.max(level * 1000, 0),
    100
  );

  const bars = Array.from(
    { length: 32 },
    (_, index) => {

      const wave =
        Math.sin(index * 0.8) * 0.35 + 0.65;

      return Math.max(
        4,
        normalized * wave
      );
    }
  );

  return (
    <div
      style={{
        height: 120,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: 3,
        padding: 20,
        background: "#080b10",
        borderRadius: 12,
      }}
    >

      {bars.map((height, index) => (

        <div
          key={index}
          style={{
            width: 4,
            height: active
              ? `${height}%`
              : "4px",

            background:
              active
                ? "#64748b"
                : "#1f2937",

            borderRadius: 4,

            transition:
              "height 80ms linear",
          }}
        />

      ))}

    </div>
  );
}