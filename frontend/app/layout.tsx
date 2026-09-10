import type { Metadata, Viewport } from "next";
import { Archivo, JetBrains_Mono } from "next/font/google";

import "./globals.css";

/**
 * Two families, three roles.
 *
 * Archivo is a grotesque with genuine industrial character — it carries both
 * the display and UI roles through its weight range rather than pulling in a
 * third family, which keeps the page cohesive. JetBrains Mono handles every
 * figure in the product: risk scores, thresholds, similarities, file paths.
 * Numbers here are instrument readouts and they are set as such — tabular, so
 * a value changing during a live call does not reflow the row it sits in.
 *
 * Loaded through next/font so they are self-hosted and there is no flash of
 * fallback text on first paint.
 */

const display = Archivo({
  subsets: ["latin"],
  weight: ["600", "700"],
  variable: "--font-display",
  display: "swap",
});

const ui = Archivo({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-ui",
  display: "swap",
});

const mono = JetBrains_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "VoxShield",
  description:
    "Voice integrity, speaker identity and call context, fused into one decision.",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f6f5f8" },
    { media: "(prefers-color-scheme: dark)", color: "#0b0a10" },
  ],
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      className={`${display.variable} ${ui.variable} ${mono.variable}`}
    >
      <head>
        {/*
          Applied before first paint so an explicit theme choice does not flash
          the system theme on reload. Deliberately tiny and dependency-free —
          anything that runs here blocks rendering.
        */}
        <script
          dangerouslySetInnerHTML={{
            __html:
              "try{var t=localStorage.getItem('voxshield-theme');if(t&&t!=='system')document.documentElement.setAttribute('data-theme',t)}catch(e){}",
          }}
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
