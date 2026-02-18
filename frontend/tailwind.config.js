/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        ink: '#0f172a',
        mist: '#f1f5f9',
        accent: '#0ea5e9',
        ember: '#f97316',
      },
      fontFamily: {
        display: ['Sora', 'ui-sans-serif', 'system-ui'],
        body: ['Manrope', 'ui-sans-serif', 'system-ui'],
      },
      boxShadow: {
        soft: '0 12px 40px rgba(14, 165, 233, 0.15)',
      },
      keyframes: {
        pulseUp: {
          '0%': { transform: 'scale(0.96)', opacity: '0.6' },
          '100%': { transform: 'scale(1)', opacity: '1' },
        },
      },
      animation: {
        pulseUp: 'pulseUp 0.45s ease-out',
      },
    },
  },
  plugins: [],
}
