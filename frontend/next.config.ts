import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  experimental: {
    agentUpgrade: "security",
  },
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "thesvg.org",
        pathname: "/icons/**",
      },
    ],
  },
};

export default nextConfig;
