"use client";

import { AppShell, Burger, Group } from "@mantine/core";
import { useDisclosure, useHeadroom } from "@mantine/hooks";
import { useRouter } from "next/navigation";
import logo from "@/assets/logo.png";
import Image from "next/image";

export default function AppShellLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [opened, { toggle }] = useDisclosure();
  const pinned = useHeadroom({ fixedAt: 40 });
  const router = useRouter();

  return (
    <AppShell
      styles={{
        root: { backgroundColor: "#ECE9EA" },
        header: { backgroundColor: "#ECE9EA" },
        navbar: { backgroundColor: "#ECE9EA" },
        main: { backgroundColor: "#ECE9EA" },
      }}
      header={{ height: 60, collapsed: !pinned, offset: opened }}
      navbar={{
        width: 300,
        breakpoint: "md",
        collapsed: { desktop: true, mobile: !opened },
      }}
      padding="md"
    >
      <AppShell.Header>
        <Group h="100%" px="md">
          <Burger opened={opened} onClick={toggle} hiddenFrom="md" size="sm" />
          <Group justify="space-between" style={{ flex: 1 }}>
            <Image
              onClick={() => router.push("/")}
              src={logo}
              alt="logo"
              width={280}
            />
          </Group>
        </Group>
      </AppShell.Header>

      <AppShell.Main>{children}</AppShell.Main>
    </AppShell>
  );
}
