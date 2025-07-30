/** @type {import('next').NextConfig} */
const nextConfig = {
  // Remove this line: output: 'export',
  trailingSlash: true,
  images: {
    unoptimized: true
  },
  eslint: {
    ignoreDuringBuilds: true,
  }
}

module.exports = nextConfig