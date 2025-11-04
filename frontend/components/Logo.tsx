import React from 'react'

export function Logo({ size = 32 }: { size?: number }) {
  const s = size
  return (
    <svg width={s} height={s} viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="g" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0%" stopColor="#2563eb" />
          <stop offset="100%" stopColor="#22d3ee" />
        </linearGradient>
      </defs>
      <rect x="4" y="4" width="56" height="56" rx="14" fill="url(#g)" />
      <g fill="#fff">
        <circle cx="24" cy="28" r="6" opacity="0.95" />
        <circle cx="40" cy="36" r="8" opacity="0.9" />
        <path d="M18 46c6-8 22-8 28 0" opacity="0.85" />
      </g>
    </svg>
  )
}



