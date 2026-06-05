module.exports = {
  root: true,
  env: { browser: true, es2020: true, node: true },
  extends: [
    'eslint:recommended',
    'plugin:react/recommended',
    'plugin:react/jsx-runtime',
    'plugin:react-hooks/recommended',
  ],
  ignorePatterns: ['dist', '.eslintrc.cjs', 'postcss.config.js', 'tailwind.config.js', 'vite.config.js'],
  parserOptions: { ecmaVersion: 'latest', sourceType: 'module' },
  settings: { react: { version: '18.2' } },
  plugins: ['react-refresh'],
  rules: {
    'react-refresh/only-export-components': [
      'warn',
      { allowConstantExport: true },
    ],
    'react/prop-types': 'off', // Turn off prop-types check since it's a modern React app
    'react/no-unescaped-entities': 'off', // Allow quotes and apostrophes in text nodes
    'no-unused-vars': 'off', // Disable unused vars check for compilation flexibility
    'react-hooks/exhaustive-deps': 'off', // Disable exhaustive deps hook warning
  },
}
