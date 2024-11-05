// @ts-check
import { PHASE_PRODUCTION_SERVER } from 'next/constants.js';
import pkg from './src/instrumentation.mjs';
const { register } = pkg;

/** @type {import('next').NextConfig} */
const nextConfig = (phase, { defaultConfig }) => {
  if (phase === PHASE_PRODUCTION_SERVER) {
    console.log('Avvio del batch polling all\'avvio del backend in produzione');
    register(); // eseguo la funzione `register()` solo quando l'app è in fase di production
  }

  return {
    reactStrictMode: true,
    webpack(config) {
      return config;
    },
  };
};

export default nextConfig;
