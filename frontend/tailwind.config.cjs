/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: '#1e40af',
        secondary: '#7c3aed',
        accent: '#06b6d4',
        neutral: '#f3f4f6',
        dark: '#1f2937',
      }
    },
  },
  plugins: [],
}
