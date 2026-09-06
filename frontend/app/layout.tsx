
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
    title: "VoxShield",
    description:
        "Real-time AI voice integrity and impersonation prevention platform",
};

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <html lang="en">
            <body>{children}</body>
        </html>
    );
}