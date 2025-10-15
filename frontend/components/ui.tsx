import { ReactNode, HTMLAttributes, ButtonHTMLAttributes } from 'react'
import clsx from 'classnames'

export function Card({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div {...props} className={clsx('rounded-xl border bg-white/70 dark:bg-gray-900/60 backdrop-blur p-4 shadow-sm', className)} />
}

export function Button({ className, variant = 'primary', ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: 'primary' | 'outline' | 'ghost' }) {
  const base = 'inline-flex items-center justify-center rounded-lg text-sm font-medium transition-colors focus:outline-none disabled:opacity-50 disabled:pointer-events-none h-10 px-4'
  const styles = {
    primary: 'bg-blue-600 text-white hover:bg-blue-700',
    outline: 'border hover:bg-gray-50 dark:hover:bg-gray-800',
    ghost: 'hover:bg-gray-100 dark:hover:bg-gray-800',
  }[variant]
  return <button {...props} className={clsx(base, styles, className)} />
}

export function Chip({ children }: { children: ReactNode }) {
  return <span className="px-3 py-1 rounded-full border bg-white/60 dark:bg-gray-900/50 text-sm">{children}</span>
}

export function Sheet({ open, onClose, children }: { open: boolean; onClose: () => void; children: ReactNode }) {
  return (
    <>
      <div className={clsx('fixed inset-0 bg-black/30 transition-opacity', open ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none')} onClick={onClose} />
      <div className={clsx('fixed right-0 top-0 h-full w-[360px] max-w-[85%] bg-white dark:bg-gray-900 border-l shadow-xl transition-transform', open ? 'translate-x-0' : 'translate-x-full')}>
        <div className="p-4 h-full overflow-auto">{children}</div>
      </div>
    </>
  )
}

export function Switch({ checked, onChange }: { checked: boolean; onChange: (c: boolean) => void }) {
  return (
    <button onClick={() => onChange(!checked)} className={clsx('h-6 w-11 rounded-full border transition-colors', checked ? 'bg-blue-600 border-blue-600' : 'bg-gray-300 border-gray-300')}>
      <span className={clsx('block h-5 w-5 bg-white rounded-full shadow transform transition-transform', checked ? 'translate-x-5' : 'translate-x-0')} />
    </button>
  )
}

export function useToasts() {
  if (typeof window === 'undefined') return { toast: (_: string) => { } }
  return {
    toast: (msg: string) => {
      const el = document.createElement('div')
      el.className = 'fixed bottom-4 left-1/2 -translate-x-1/2 z-50'
      el.innerHTML = `<div class="px-4 py-2 rounded-lg border bg-white dark:bg-gray-900 shadow">${msg.replace(/</g, '&lt;')}</div>`
      document.body.appendChild(el)
      setTimeout(() => el.remove(), 2500)
    }
  }
}



