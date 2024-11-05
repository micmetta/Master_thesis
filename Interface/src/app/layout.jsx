"use client";
import { MantineProvider } from '@mantine/core';
import localFont from "next/font/local";
import "./globals.css";
import AppShellLayout from './AppShellLayout';
import "@mantine/core/styles.css";
var geistSans = localFont({
    src: "./fonts/GeistVF.woff",
    variable: "--font-geist-sans",
    weight: "100 900",
});
var geistMono = localFont({
    src: "./fonts/GeistMonoVF.woff",
    variable: "--font-geist-mono",
    weight: "100 900",
});
export default function RootLayout(_a) {
    var children = _a.children;
    return (<html lang="en">
      <body className={"".concat(geistSans.variable, " ").concat(geistMono.variable, " antialiased")}>
        {/* avvolgo l'intera applicazione con MantineProvider */}
        <MantineProvider>
          <AppShellLayout>{children}</AppShellLayout>
        </MantineProvider>
      </body>
    </html>);
}
