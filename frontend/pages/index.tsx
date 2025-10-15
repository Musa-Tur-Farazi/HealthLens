import { useCallback, useEffect, useRef, useState } from 'react'
import clsx from 'classnames'
import { diag } from '../lib/api'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { UploadCloud, Sun, Moon } from 'lucide-react'
import { Logo } from '@/components/Logo'
import { motion } from 'framer-motion'

type Modality = 'disease' | 'xray'

export default function Home() {
  const [imageB64, setImageB64] = useState<string | null>(null)
  const [fileMeta, setFileMeta] = useState<string>('')
  const [symptoms, setSymptoms] = useState('')
  const [modality, setModality] = useState<Modality | null>(null)
  const [includeCam] = useState(true)
  const [loading, setLoading] = useState<'idle' | 'uploading' | 'classifying' | 'generating'>('idle')
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<any>(null)
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const toast = (m: string) => console.log(m)
  const [categoryTouched, setCategoryTouched] = useState(false)

  // Theme management (persistent)
  const [theme, setTheme] = useState<'light' | 'dark'>('light')
  const [mounted, setMounted] = useState(false)

  // Hydration-safe theme initialization
  useEffect(() => {
    setMounted(true)
    const stored = localStorage.getItem('hl:theme') as 'light' | 'dark' | null
    if (stored) {
      setTheme(stored)
    } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      setTheme('dark')
    }
  }, [])

  useEffect(() => {
    if (!mounted) return
    const root = document.documentElement
    if (theme === 'dark') root.classList.add('dark'); else root.classList.remove('dark')
    localStorage.setItem('hl:theme', theme)
  }, [theme, mounted])

  useEffect(() => {
    try { const saved = localStorage.getItem('hl:last'); if (saved) setResult(JSON.parse(saved)) } catch { }
  }, [])
  useEffect(() => { if (result) localStorage.setItem('hl:last', JSON.stringify(result)) }, [result])

  const onPick = useCallback(() => inputRef.current?.click(), [])

  const onFile = useCallback(async (f: File) => {
    if (!f) return
    if (f.size > 8 * 1024 * 1024) {
      setError('File too large (max 8MB).')
      return
    }
    setError(null)
    setResult(null)
    setLoading('uploading')
    setFileMeta(`${f.name} — ${(f.size / 1024).toFixed(0)} KB`)
    const reader = new FileReader()
    reader.onload = () => {
      setImageB64(reader.result as string)
      setLoading('idle')
    }
    reader.onerror = () => {
      setError('Failed to read file')
      setLoading('idle')
    }
    reader.readAsDataURL(f)
  }, [])

  const send = useCallback(async () => {
    if (!imageB64) { setError('Please upload an image.'); return }
    if (!modality) { setCategoryTouched(true); setError('Please select an image type.'); return }
    setError(null)
    setLoading('classifying')
    try {
      const data = await diag({
        modality: modality,
        symptoms,
        topk: 3,
        include_cam: includeCam,
        image_b64: imageB64,
      })
      setLoading('generating')
      setResult(data)
    } catch (e: any) {
      setError(e?.message || 'Request failed')
    } finally {
      setLoading('idle')
    }
  }, [imageB64, modality, symptoms, includeCam])

  const topk = result?.topk as Array<{ label: string, prob: number }> | undefined
  const locked = !!result

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-b from-gray-100 to-gray-200 dark:from-gray-950 dark:to-gray-900">
      <header className="border-b px-4 py-3 flex items-center justify-between backdrop-blur bg-white/60 dark:bg-gray-900/50">
        <div className="flex items-center gap-3">
          <Logo size={28} />
          <h1 className="font-semibold tracking-tight">HealthLens</h1>
        </div>
        <div>
          <Button variant="outline" className="gap-2" onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')}>
            {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            {theme === 'dark' ? 'Light' : 'Dark'}
          </Button>
        </div>
      </header>

      <main className="flex-1 max-w-7xl w-full mx-auto p-6 space-y-6">
        <div className="grid md:grid-cols-2 gap-8 items-start">
          <div className="space-y-4">
            <div
              className={clsx(
                'rounded-2xl border border-dashed p-6 text-center h-[420px] flex flex-col items-center justify-center backdrop-blur-md shadow-sm transition-all',
                'bg-white/60 dark:bg-gray-900/40 hover:shadow-md',
                dragOver ? 'ring-2 ring-blue-500' : 'ring-0'
              )}
              onClick={locked ? undefined : onPick}
              onDragOver={(e) => { if (locked) return; e.preventDefault(); setDragOver(true) }}
              onDragLeave={() => { if (locked) return; setDragOver(false) }}
              onDrop={(e) => { if (locked) return; e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files?.[0]; if (f) onFile(f) }}
            >
              {imageB64 ? (
                <div className="relative w-full h-full flex items-center justify-center">
                  <img src={imageB64} alt="preview" className="max-h-[360px] object-contain" />
                  {locked && (
                    <div className="absolute bottom-2 left-1/2 -translate-x-1/2 text-[11px] px-2 py-1 rounded-full border bg-white/70 dark:bg-gray-900/60">
                      Image locked — press Reset to change
                    </div>
                  )}
                </div>
              ) : (
                <>
                  <UploadCloud className="h-7 w-7 mb-2 text-blue-600" />
                  <p className="font-medium">Drag & drop or click to upload</p>
                  <p className="text-sm text-gray-500">JPEG/PNG, up to 8MB</p>
                </>
              )}
              <input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f) }} />
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-300 h-5">{fileMeta}</div>

            {/* Category Selection */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">Select image type</label>
                {!modality && categoryTouched && (
                  <span className="text-xs text-red-600">Please select one</span>
                )}
              </div>
              <div className="grid grid-cols-2 gap-3">
                <button
                  onClick={() => { setModality('disease'); setCategoryTouched(true); setError(null) }}
                  disabled={locked}
                  className={clsx(
                    'px-4 py-3 rounded-xl border-2 transition-all text-sm font-medium',
                    'bg-white/60 dark:bg-gray-900/40 backdrop-blur-sm',
                    modality === 'disease'
                      ? 'border-blue-500 ring-2 ring-blue-500/50 text-blue-700 dark:text-blue-300'
                      : (!modality && categoryTouched ? 'border-red-500' : 'border-gray-300 dark:border-gray-700 hover:border-blue-400'),
                    locked ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
                  )}
                >
                  Skin Lesion/Histopathological Image
                </button>
                <button
                  onClick={() => { setModality('xray'); setCategoryTouched(true); setError(null) }}
                  disabled={locked}
                  className={clsx(
                    'px-4 py-3 rounded-xl border-2 transition-all text-sm font-medium',
                    'bg-white/60 dark:bg-gray-900/40 backdrop-blur-sm',
                    modality === 'xray'
                      ? 'border-blue-500 ring-2 ring-blue-500/50 text-blue-700 dark:text-blue-300'
                      : (!modality && categoryTouched ? 'border-red-500' : 'border-gray-300 dark:border-gray-700 hover:border-blue-400'),
                    locked ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
                  )}
                >
                  X-Ray Image
                </button>
              </div>
            </div>

            <Textarea
              value={symptoms}
              onChange={e => setSymptoms(e.target.value)}
              rows={8}
              placeholder="Describe symptoms (optional)"
              className="rounded-2xl border bg-white/60 dark:bg-gray-900/50 backdrop-blur-sm shadow-sm focus:ring-2 focus:ring-blue-500"
            />
            <div className="flex gap-2">
              <Button onClick={send} disabled={loading !== 'idle'}>
                {loading === 'idle' && (result ? 'Retry Prompt' : 'Send')}
                {loading === 'uploading' && 'Uploading…'}
                {loading === 'classifying' && 'Classifying…'}
                {loading === 'generating' && 'Generating…'}
              </Button>
              <Button variant="outline" onClick={() => {
                try { localStorage.removeItem('hl:last') } catch { }
                setImageB64(null)
                setResult(null)
                setSymptoms('')
                setFileMeta('')
                setDragOver(false)
                setError(null)
                setModality(null)
                if (inputRef.current) inputRef.current.value = ''
              }}>Reset</Button>
            </div>
            {error && <div className="text-red-600 text-sm">{error}</div>}
          </div>

          <div className="space-y-4">
            {result ? (
              <div className="space-y-3">
                {topk && topk.length > 0 && (
                  <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }}>
                    <Card className="p-4">
                      <h3 className="font-semibold mb-3">Top predictions</h3>
                      <div className="grid grid-cols-2 gap-4 items-center">
                        <motion.div initial={{ scale: 0.9 }} animate={{ scale: 1 }} transition={{ duration: 0.4 }}>
                          <PieChart data={topk.slice(0, 3)} />
                        </motion.div>
                        <div className="space-y-2">
                          {topk.slice(0, 3).map((t, i) => (
                            <div key={i} className="flex items-center justify-between text-sm">
                              <div className="flex items-center gap-2">
                                <span className="inline-block h-3 w-3 rounded-full" style={{ background: getColor(i) }} />
                                <span>{t.label}</span>
                              </div>
                              <span className="font-medium">{(t.prob * 100).toFixed(1)}%</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </Card>
                  </motion.div>
                )}
                {topk && topk.length > 0 && (
                  <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35, delay: 0.03 }}>
                    <Card className="p-4">
                      <div className="flex items-start justify-between gap-4">
                        <div>
                          <div className="text-xs uppercase tracking-wide text-gray-500">Primary diagnosis</div>
                          <div className="text-lg font-semibold mt-1">{topk[0].label}</div>
                        </div>
                        <div className="text-right">
                          <div className="text-xs text-gray-500">Probability</div>
                          <div className="text-lg font-semibold">{(topk[0].prob * 100).toFixed(1)}%</div>
                        </div>
                      </div>
                      {result?.report?.next_steps && (
                        <div className="mt-4">
                          <div className="font-semibold">Recommended next steps</div>
                          <ul className="list-disc pl-5 mt-1 space-y-1">
                            {toArray(result.report.next_steps).map((s: string, i: number) => (
                              <li key={i}>{s}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </Card>
                  </motion.div>
                )}
                {result.cam_b64 && (
                  <div>
                    <img src={result.cam_b64} alt="Grad-CAM" className="rounded-xl border max-h-64 object-contain shadow" />
                  </div>
                )}
                {result.report && (
                  <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35, delay: 0.05 }}>
                    <Card className="p-4">
                      <div className="space-y-4 text-sm">
                        <div>
                          <div className="font-semibold">Impression</div>
                          <div className="whitespace-pre-wrap">{formatMaybeArray(result.report.impression)}</div>
                        </div>
                        {result.report.findings && (
                          <div>
                            <div className="font-semibold">Findings</div>
                            <ul className="list-disc pl-5 mt-1 space-y-1">
                              {toArray(result.report.findings).map((f: string, i: number) => (
                                <li key={i}>{f}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                        {result.report.disease_summary && (
                          <div>
                            <div className="font-semibold">About the Disease</div>
                            <div className="whitespace-pre-wrap bg-blue-50 dark:bg-blue-900/20 p-3 rounded-lg border-l-4 border-blue-400">
                              {formatMaybeArray(result.report.disease_summary)}
                            </div>
                          </div>
                        )}
                        {result.report.red_flags && (
                          <div>
                            <div className="font-semibold">Red Flags</div>
                            <ul className="list-disc pl-5 mt-1 space-y-1 text-red-700 dark:text-red-400">
                              {toArray(result.report.red_flags).map((f: string, i: number) => (
                                <li key={i}>{f}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                        <div>
                          <div className="font-semibold">Next Steps</div>
                          <div className="whitespace-pre-wrap">{formatMaybeArray(result.report.next_steps)}</div>
                        </div>
                        <div>
                          <div className="font-semibold">Disclaimer</div>
                          <div className="whitespace-pre-wrap text-gray-600 dark:text-gray-400 text-xs">{formatMaybeArray(result.report.disclaimer)}</div>
                        </div>
                      </div>
                    </Card>
                  </motion.div>
                )}
                {/* Details panel removed per request to avoid raw JSON exposure */}
              </div>
            ) : (
              <div className="text-sm text-gray-500">No results yet.</div>
            )}
          </div>
        </div>
      </main>

      <footer className="px-4 py-3 text-xs text-center text-gray-500 border-t backdrop-blur bg-white/60 dark:bg-gray-900/50">Research demo. Not medical advice.</footer>
    </div>
  )
}


// Helpers
function getColor(i: number) {
  const colors = ['#2563eb', '#16a34a', '#f59e0b']
  return colors[i % colors.length]
}

function formatMaybeArray(v: any) {
  if (!v) return ''
  if (Array.isArray(v)) return v.join('\n')
  if (typeof v === 'object') return JSON.stringify(v, null, 2)
  return String(v)
}

function PieChart({ data }: { data: { label: string, prob: number }[] }) {
  const size = 160, r = 70, c = 2 * Math.PI * r
  const total = Math.max(1e-8, data.reduce((s, d) => s + Math.max(0, d.prob || 0), 0))
  let acc = 0
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="mx-auto">
      <g transform={`translate(${size / 2},${size / 2})`}>
        {/* Track */}
        <circle r={r} fill="none" stroke="#e5e7eb" strokeWidth={18} />
        {/* Segments */}
        {data.map((d, i) => {
          const val = Math.max(0, d.prob || 0) / total
          const len = val * c
          const dash = `${len} ${c - len}`
          const rot = (acc / c) * 360 - 90
          acc += len
          return (
            <circle key={i} r={r} fill="none" stroke={getColor(i)} strokeWidth={18} strokeLinecap="butt" strokeDasharray={dash} transform={`rotate(${rot})`} />
          )
        })}
        {(() => {
          const maxProb = Math.max(0, ...data.map(d => d.prob || 0))
          const pct = (maxProb * 100).toFixed(1) + '%'
          return (
            <>
              <text textAnchor="middle" dominantBaseline="middle" fontSize="20" y={0} fill="currentColor" className="text-gray-900 dark:text-white font-semibold">{pct}</text>
            </>
          )
        })()}
      </g>
    </svg>
  )
}

function toArray(v: any): string[] {
  if (!v) return []
  if (Array.isArray(v)) return v
  if (typeof v === 'string') return [v]
  return [JSON.stringify(v)]
}
