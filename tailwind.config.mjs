/** @type {import('tailwindcss').Config} */
import { addDynamicIconSelectors } from "@iconify/tailwind";
import typography from "@tailwindcss/typography";
import daisyUI from "daisyui";
import { SITE_THEME } from "./src/config";

export default {
  content: ["./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}"],
  theme: {
    extend: {},
  },
  safelist: [
    "alert",
    "alert-info",
    "alert-success",
    "alert-warning",
    "alert-error",
  ],
  plugins: [daisyUI, typography, addDynamicIconSelectors()],
  daisyui: {
    themes: [
      {
        frostilight: {
          primary: "#6f8cff",
          "primary-content": "#0b1220",
          secondary: "#4adfb8",
          "secondary-content": "#052217",
          accent: "#a78bfa",
          "accent-content": "#0f0b1c",
          neutral: "#1b2433",
          "neutral-content": "#f5f7ff",
          "base-100": "#f6f8ff",
          "base-200": "#e8eefb",
          "base-300": "#d4dced",
          "base-content": "#0f172a",
          info: "#3abff8",
          "info-content": "#00131a",
          success: "#36d399",
          "success-content": "#062017",
          warning: "#fbbd23",
          "warning-content": "#1a1200",
          error: "#f87272",
          "error-content": "#1a0707",
          "--rounded-box": "1rem",
          "--rounded-btn": "0.9rem",
          "--rounded-badge": "1rem",
          "--animation-btn": "0.2s",
          "--animation-input": "0.2s",
          "--btn-focus-scale": "0.98",
          "--border-btn": "1px",
          "--tab-border": "1px",
          "--tab-radius": "0.6rem",
        },
      },
      {
        frostinight: {
          primary: "#7aa5ff",
          "primary-content": "#0b1020",
          secondary: "#67f2d0",
          "secondary-content": "#032018",
          accent: "#b39bff",
          "accent-content": "#120a24",
          neutral: "#0c1220",
          "neutral-content": "#e6edff",
          "base-100": "#0b0f1a",
          "base-200": "#0f1626",
          "base-300": "#172238",
          "base-content": "#e6edff",
          info: "#5bbcff",
          "info-content": "#07131f",
          success: "#57e6b3",
          "success-content": "#071a14",
          warning: "#ffd166",
          "warning-content": "#1f1605",
          error: "#ff8b8b",
          "error-content": "#1f0b0b",
          "--rounded-box": "1rem",
          "--rounded-btn": "0.9rem",
          "--rounded-badge": "1rem",
          "--animation-btn": "0.2s",
          "--animation-input": "0.2s",
          "--btn-focus-scale": "0.98",
          "--border-btn": "1px",
          "--tab-border": "1px",
          "--tab-radius": "0.6rem",
        },
      },
    ],
    darkTheme: SITE_THEME.dark, // name of one of the included themes for dark mode
    logs: false, // Shows info about daisyUI version and used config in the console when building your CSS
  },
};
