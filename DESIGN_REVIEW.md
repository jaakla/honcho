# Design Review: Honcho UI

Reviewed against: no `DESIGN_BRIEF.md` present in repo  
Assessment basis: professional quality for a data-heavy admin/explorer UI  
Date: 2026-04-21

## Screenshots Captured

- `/Users/jaak/mygit/honcho/ui/screenshots/review-dashboard-desktop-localhost-1280.png`
- `/Users/jaak/mygit/honcho/ui/screenshots/review-dashboard-mobile-localhost-375.png`
- `/Users/jaak/mygit/honcho/ui/screenshots/review-settings-desktop-1280.png`
- `/Users/jaak/mygit/honcho/ui/screenshots/review-settings-tablet-768.png`
- `/Users/jaak/mygit/honcho/ui/screenshots/review-settings-mobile-localhost-375.png`

## Overall Verdict

The current UI reads as an internal scaffold, not a production-quality explorer. The strongest issue is not visual taste but structural stability: several screens are rendering different trees on the server and client, which produces hydration errors and weakens first paint. Visually, the app is competent but generic, under-art-directed, and too empty in its primary states.

## Must Fix

### 1. Hydration is unstable across core routes

- Evidence: `ui/.next/dev/logs/next-development.log` records hydration mismatch errors for the dashboard and sessions routes.
- Root cause: [`ui/hooks/use-config.ts`](/Users/jaak/mygit/honcho/ui/hooks/use-config.ts:6) returns `undefined` on first render, while route components branch to materially different markup once the client mounts. See [`ui/app/(app)/page.tsx`](/Users/jaak/mygit/honcho/ui/app/(app)/page.tsx:97) and [`ui/app/(app)/sessions/page.tsx`](/Users/jaak/mygit/honcho/ui/app/(app)/sessions/page.tsx:18).
- Impact: users get a fragile first impression, React is forced to regenerate trees, and the UI feels unreliable before any styling concerns are even considered.

### 2. The dashboard welcome state wastes almost the entire viewport

- Evidence: `review-dashboard-desktop-localhost-1280.png` and `review-dashboard-mobile-localhost-375.png`.
- The default dashboard is mostly blank canvas with a small centered message and one button. The primary surface communicates “unfinished” rather than “ready to explore”.
- Relevant implementation: [`ui/app/(app)/page.tsx`](/Users/jaak/mygit/honcho/ui/app/(app)/page.tsx:99).
- Impact: this is the product’s first screen and currently has almost no hierarchy, orientation, or confidence-building content.

### 3. The mobile settings screen is not adapted tightly enough for 375px

- Evidence: `review-settings-mobile-localhost-375.png`.
- The intro copy clips at the right edge, the form keeps desktop-like spacing, and the page feels squeezed rather than intentionally reorganized for mobile.
- Relevant implementation: [`ui/app/(app)/settings/page.tsx`](/Users/jaak/mygit/honcho/ui/app/(app)/settings/page.tsx:47).
- Impact: the only non-empty screen in the app already feels borderline at mobile width, which is a clear sign the layout was not tuned from the smallest breakpoint upward.

## Should Fix

### 4. The global Dialectic drawer competes with the core product before it earns that priority

- Evidence: visible on every captured screen.
- The bottom-fixed drawer is always present, even when no workspace is configured and the user is still in setup.
- Relevant implementation: [`ui/app/(app)/layout.tsx`](/Users/jaak/mygit/honcho/ui/app/(app)/layout.tsx:38) and [`ui/components/dialectic-drawer.tsx`](/Users/jaak/mygit/honcho/ui/components/dialectic-drawer.tsx:81).
- Impact: it adds visual weight to a secondary tool before the primary workflow is usable. For a professional admin UI, that is backwards.

### 5. The visual system is mostly stock shadcn dark mode, with little product character

- Evidence: dashboard and settings screenshots.
- The app uses grayscale shadcn variables plus `Geist`, and the root layout forces dark mode globally.
- Relevant implementation: [`ui/app/layout.tsx`](/Users/jaak/mygit/honcho/ui/app/layout.tsx:27) and [`ui/app/globals.css`](/Users/jaak/mygit/honcho/ui/app/globals.css:51).
- Impact: the UI is clean enough, but it does not feel designed. It feels generated from defaults.

## Could Improve

### 6. Form and utility rows need stronger responsive and accessibility discipline

- The settings labels are visually adjacent to fields but not programmatically bound with `htmlFor`/`id`. See [`ui/app/(app)/settings/page.tsx`](/Users/jaak/mygit/honcho/ui/app/(app)/settings/page.tsx:57).
- Search/action rows rely on rigid horizontal flex layouts that will get tight quickly on smaller screens. See [`ui/app/(app)/search/page.tsx`](/Users/jaak/mygit/honcho/ui/app/(app)/search/page.tsx:87) and [`ui/app/(app)/conclusions/page.tsx`](/Users/jaak/mygit/honcho/ui/app/(app)/conclusions/page.tsx:101).
- These are not the top problem today, but they reinforce the prototype feel.

## Recommended Direction

For this product, a restrained Swiss or Dieter Rams direction would fit best:

- keep the dark mode optional instead of mandatory
- use one restrained accent color for state and navigation focus
- replace the empty dashboard with a denser onboarding surface
- demote or hide the Dialectic drawer until a peer/workspace is active
- eliminate hydration mismatches before any visual polish work

## Summary

Current quality level: internal tooling prototype.  
Blocking issue: hydration mismatch across main routes.  
Most important design issue: the app does not use its primary surfaces to orient the user or build confidence.
