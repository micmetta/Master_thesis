// @ts-check
import { PHASE_PRODUCTION_SERVER } from 'next/constants.js';
import pkg from './src/instrumentation.mjs';
const { register } = pkg;

/** @type {import('next').NextConfig} */
const nextConfig = (phase, { defaultConfig }) => {
  if (phase === PHASE_PRODUCTION_SERVER) {
    console.log('(instrumentation.mjs - PHASE_PRODUCTION_SERVER) Backend started and then I call register() which is in instrumentation.mjs');
    register(); // I run the `register()` function only when the app is in production
  }

  return {
    reactStrictMode: true,
    webpack(config) {
      return config;
    },
  };
};

export default nextConfig;
