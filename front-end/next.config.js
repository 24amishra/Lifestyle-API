/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  trailingSlash: true,
  images: {
    unoptimized: true
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
  // Or specifically disable this rule
  eslint: {
    rules: {
      'react/no-unescaped-entities': 'off'
    }
  }
}

module.exports = nextConfig