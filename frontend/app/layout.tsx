import type { Metadata } from "next";
import "./globals.css";

import { Geist, Geist_Mono } from "next/font/google";
import Navbar from "@/components/Navbar";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});
export const metadata = {
  title: "Prediction",
  description: "Generated by create next app",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} flex flex-col antialiased m-0 min-h-screen font-[family-name:var(--font-geist-sans)] bg-neutral-200 `}
      >
        <Navbar />
        {children}
        <footer className="w-full h-[40px] flex items-center justify-center bg-gray-800 text-white text-sm mt-auto">
          &copy; 2025 Thesis Software
        </footer>
      </body>
    </html>
  );
}
