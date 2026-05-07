# Repository Guidelines

## Project Structure & Module Organization
This repository is an Astro-based personal blog. Application routes live in `src/pages/`, shared layouts and UI components are in `src/layouts/` and `src/components/`, and reusable helpers are under `src/utils/` and `src/plugins/`. Blog content is stored as MDX in `src/content/blog/`. Static assets such as images, icons, and PDFs belong in `public/`. Site-wide settings are primarily driven by `frosti.config.yaml`, with additional typed exports in `src/config.ts`.

## Build, Test, and Development Commands
Use `pnpm install` to install dependencies. Run `pnpm dev` for local development and `pnpm preview` to serve the production build locally. Use `pnpm check` to run Astro type and content checks. Use `pnpm build` to validate the project, generate the static site, and build the Pagefind search index into `dist/`. If search assets need to be copied into `public/pagefind`, run `pnpm search:index`. Use `pnpm search:clean` to remove generated search output before rebuilding.

## Coding Style & Naming Conventions
The project uses TypeScript, Astro, Tailwind, and SCSS. ESLint is configured through `eslint.config.js` with Antfu's preset: use double quotes, semicolons, and explicit arrow-function parentheses. Follow the existing naming pattern: components and pages use PascalCase filenames such as `BaseLayout.astro`, utility modules use camelCase such as `blogUtils.ts`, and content filenames should remain descriptive and stable for routing. Keep configuration and content edits small and localized.

## Testing Guidelines
There is no dedicated unit test suite in this repository today. Treat `pnpm check` and `pnpm build` as the required validation steps for code, routing, and content changes. After editing search, blog metadata, or generated pages, also verify the relevant page locally with `pnpm dev`. For content additions, confirm the MDX file renders correctly and appears in the expected archive, tag, or category page.

## Commit & Pull Request Guidelines
Recent history follows short conventional-style messages such as `feat: add a card css` and `chore:update blog`. Prefer `feat:`, `fix:`, or `chore:` prefixes with a concise summary. Pull requests should describe what changed, why it changed, and how it was validated. Include screenshots for layout or styling changes, and call out any changes to `frosti.config.yaml`, `public/`, or content collections because they affect the published site directly.
