"use client";

import { AppShell, Burger, MantineProvider, Group } from '@mantine/core'; 
import { useDisclosure, useHeadroom} from '@mantine/hooks';
import localFont from "next/font/local";
import "./globals.css";
import logo from "@/assets/logo.png";
import classes from "./app-shell.module.css";
import Image from "next/image";
import { useRouter } from "next/navigation";
import Link from "next/link";
import AppShellLayout from './AppShellLayout';
import "@mantine/core/styles.css";

const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});
const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  
  const [opened, { toggle }] = useDisclosure(); // Uso del hook lato client
  const pinned = useHeadroom({ fixedAt: 40 });
  const router = useRouter();

  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        {/* avvolgo l'intera applicazione con MantineProvider */}
        <MantineProvider>
          <AppShellLayout>{children}</AppShellLayout>
        </MantineProvider>
      </body>
    </html>
  );
}
