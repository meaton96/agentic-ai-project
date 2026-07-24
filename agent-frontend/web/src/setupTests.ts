import '@testing-library/jest-dom/vitest'

// jsdom has no ResizeObserver — @xyflow/react (Builder screen) requires one
// to measure node/canvas dimensions. A no-op stub is enough for tests: they
// assert on rendered content/interactions, not on real layout measurements.
if (typeof globalThis.ResizeObserver === 'undefined') {
  globalThis.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}
