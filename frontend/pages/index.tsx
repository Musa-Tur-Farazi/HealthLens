import { useCallback, useEffect, useRef, useState } from 'react'
import clsx from 'classnames'
import { diag, getDetailedReport, analyzeReport, combinedAnalysis } from '../lib/api'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { UploadCloud, Sun, Moon, FileText } from 'lucide-react'
import { Logo } from '@/components/Logo'
import { motion } from 'framer-motion'

type Modality = 'disease' | 'xray' | 'report' | 'combined'

export default function Home() {
  const [imageB64, setImageB64] = useState<string | null>(null)
  const [fileMeta, setFileMeta] = useState<string>('')
  const [reportB64, setReportB64] = useState<string | null>(null)
  const [reportFileMeta, setReportFileMeta] = useState<string>('')
  const [reportFileType, setReportFileType] = useState<'image' | 'pdf'>('pdf')
  const [symptoms, setSymptoms] = useState('')
  const [modality, setModality] = useState<Modality | null>(null)
  const [includeCam] = useState(true)
  const [loading, setLoading] = useState<'idle' | 'uploading' | 'classifying' | 'generating' | 'generating_detailed' | 'combined'>('idle')
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<any>(null)
  const [detailedReport, setDetailedReport] = useState<any>(null)
  const [showDetailedReport, setShowDetailedReport] = useState(false)
  const [reportAnalysis, setReportAnalysis] = useState<any>(null)
  const [combinedResult, setCombinedResult] = useState<any>(null)
  const [dragOver, setDragOver] = useState(false)
  const [fileType, setFileType] = useState<'image' | 'pdf'>('image')
  const inputRef = useRef<HTMLInputElement>(null)
  const reportInputRef = useRef<HTMLInputElement>(null)
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
  const onPickReport = useCallback(() => reportInputRef.current?.click(), [])

  const onFile = useCallback(async (f: File) => {
    if (!f) return

    // Check file type
    const isPDF = f.type === 'application/pdf'
    const isImage = f.type.startsWith('image/')

    if (!isPDF && !isImage) {
      setError('Please upload a valid image (JPEG/PNG) or PDF')
      return
    }
    if (f.size > 10 * 1024 * 1024) {
      setError('File too large (max 10MB).')
      return
    }

    setFileType(isPDF ? 'pdf' : 'image')
    setError(null)
    setResult(null)
    setReportAnalysis(null)
    setLoading('uploading')
    setFileMeta(`${f.name} — ${(f.size / 1024).toFixed(0)} KB ${isPDF ? '📄 PDF' : ''}`)

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

  const onReportFile = useCallback(async (f: File) => {
    if (!f) return

    // Check file type
    const isPDF = f.type === 'application/pdf'
    const isImage = f.type.startsWith('image/')

    if (!isPDF && !isImage) {
      setError('Please upload a valid PDF or image for the report')
      return
    }
    if (f.size > 10 * 1024 * 1024) {
      setError('Report file too large (max 10MB).')
      return
    }

    setReportFileType(isPDF ? 'pdf' : 'image')
    setError(null)
    setReportFileMeta(`${f.name} — ${(f.size / 1024).toFixed(0)} KB ${isPDF ? '📄 PDF' : ''}`)

    const reader = new FileReader()
    reader.onload = () => {
      setReportB64(reader.result as string)
    }
    reader.onerror = () => {
      setError('Failed to read report file')
    }
    reader.readAsDataURL(f)
  }, [])

  const send = useCallback(async () => {
    if (!imageB64) { setError('Please upload an image or PDF.'); return }
    if (!modality) { setCategoryTouched(true); setError('Please select an image type.'); return }
    setError(null)

    // Handle combined analysis (image + report)
    if (modality === 'combined') {
      if (!reportB64) { setError('Please upload both a medical image AND a lab report for combined analysis.'); return }
      setLoading('combined')
      try {
        const data = await combinedAnalysis({
          image_b64: imageB64,
          report_data: reportB64,
          report_type: reportFileType,
          modality: 'disease', // Default to disease, can be changed to xray if needed
          symptoms: symptoms || '',
          query: ''
        })
        setCombinedResult(data)
        setResult(null) // Clear diagnosis results
        setReportAnalysis(null) // Clear report analysis
      } catch (e: any) {
        setError(e?.message || 'Combined analysis failed')
      } finally {
        setLoading('idle')
      }
      return
    }

    // Handle medical report analysis
    if (modality === 'report') {
      setLoading('classifying')
      try {
        const data = await analyzeReport({
          file_data: imageB64,
          file_type: fileType,
          query: symptoms || ''
        })
        setReportAnalysis(data)
        setResult(null) // Clear diagnosis results
        setCombinedResult(null) // Clear combined results
      } catch (e: any) {
        setError(e?.message || 'Report analysis failed')
      } finally {
        setLoading('idle')
      }
      return
    }

    // Handle regular diagnosis (disease/xray)
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
      setReportAnalysis(null) // Clear report analysis
      setCombinedResult(null) // Clear combined results
      setShowDetailedReport(false) // Reset detailed report view
      setDetailedReport(null) // Clear previous detailed report
    } catch (e: any) {
      setError(e?.message || 'Request failed')
    } finally {
      setLoading('idle')
    }
  }, [imageB64, modality, symptoms, includeCam, fileType, reportB64, reportFileType])

  const fetchDetailedReport = useCallback(async () => {
    if (!result?.payload_used) { setError('No diagnosis available for detailed report'); return }
    setLoading('generating_detailed')
    setError(null)
    try {
      const data = await getDetailedReport(result.payload_used)
      setDetailedReport(data)
      setShowDetailedReport(true)
    } catch (e: any) {
      setError(e?.message || 'Failed to generate detailed report')
    } finally {
      setLoading('idle')
    }
  }, [result])

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

      <main className="flex-1 max-w-7xl w-full mx-auto p-6 space-y-8">
        {/* Image + Prompt Section - Bigger and at the top */}
        <div className="grid lg:grid-cols-2 gap-8 items-start">
          {/* Image Upload - Bigger */}
          <div className="space-y-4">
            <div
              className={clsx(
                'rounded-2xl border border-dashed p-8 text-center min-h-[550px] flex flex-col items-center justify-center backdrop-blur-md shadow-lg transition-all',
                'bg-white/60 dark:bg-gray-900/40 hover:shadow-xl',
                dragOver ? 'ring-4 ring-blue-500' : 'ring-0'
              )}
              onClick={locked ? undefined : onPick}
              onDragOver={(e) => { if (locked) return; e.preventDefault(); setDragOver(true) }}
              onDragLeave={() => { if (locked) return; setDragOver(false) }}
              onDrop={(e) => { if (locked) return; e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files?.[0]; if (f) onFile(f) }}
            >
              {imageB64 ? (
                fileType === 'pdf' ? (
                  <div className="flex flex-col items-center justify-center space-y-4">
                    <FileText className="h-24 w-24 text-blue-600" />
                    <div className="text-center">
                      <p className="font-semibold text-lg">PDF Uploaded</p>
                      <p className="text-sm text-gray-500 mt-1">{fileMeta}</p>
                    </div>
                    {locked && (
                      <p className="text-xs text-gray-500">Press Reset to change file</p>
                    )}
                  </div>
                ) : (
                  <div className="relative w-full h-full flex items-center justify-center">
                    <img src={imageB64} alt="preview" className="max-h-[480px] max-w-full object-contain rounded-lg shadow-md" />
                    {locked && (
                      <div className="absolute bottom-3 left-1/2 -translate-x-1/2 text-xs px-3 py-2 rounded-full border bg-white/80 dark:bg-gray-900/70 backdrop-blur">
                        Press Reset to change image
                      </div>
                    )}
                  </div>
                )
              ) : (
                <>
                  <UploadCloud className="h-12 w-12 mb-4 text-blue-600" />
                  <p className="font-semibold text-lg">Drag & drop or click to upload</p>
                  <p className="text-sm text-gray-500 mt-2">JPEG/PNG/PDF, up to 10MB</p>
                </>
              )}
              <input ref={inputRef} type="file" accept="image/*,application/pdf" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f) }} />
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-300 h-5 text-center">{fileMeta}</div>
          </div>

          {/* Prompt Section - Bigger */}
          <div className="space-y-5">
            {/* Category Selection */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <label className="text-base font-semibold">Select Image Type</label>
                {!modality && categoryTouched && (
                  <span className="text-sm text-red-600 font-medium">⚠️ Please select one</span>
                )}
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                <button
                  onClick={() => { setModality('disease'); setCategoryTouched(true); setError(null) }}
                  disabled={locked}
                  className={clsx(
                    'px-4 py-4 rounded-xl border-2 transition-all font-medium text-sm',
                    'bg-white/60 dark:bg-gray-900/40 backdrop-blur-sm shadow-sm hover:shadow-md',
                    modality === 'disease'
                      ? 'border-blue-500 ring-2 ring-blue-500/50 text-blue-700 dark:text-blue-300 shadow-lg'
                      : (!modality && categoryTouched ? 'border-red-500' : 'border-gray-300 dark:border-gray-700 hover:border-blue-400'),
                    locked ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
                  )}
                >
                  Skin / Histopathology
                </button>
                <button
                  onClick={() => { setModality('xray'); setCategoryTouched(true); setError(null) }}
                  disabled={locked}
                  className={clsx(
                    'px-4 py-4 rounded-xl border-2 transition-all font-medium text-sm',
                    'bg-white/60 dark:bg-gray-900/40 backdrop-blur-sm shadow-sm hover:shadow-md',
                    modality === 'xray'
                      ? 'border-blue-500 ring-2 ring-blue-500/50 text-blue-700 dark:text-blue-300 shadow-lg'
                      : (!modality && categoryTouched ? 'border-red-500' : 'border-gray-300 dark:border-gray-700 hover:border-blue-400'),
                    locked ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
                  )}
                >
                  X-Ray Imaging
                </button>
                <button
                  onClick={() => { setModality('report'); setCategoryTouched(true); setError(null) }}
                  disabled={locked}
                  className={clsx(
                    'px-4 py-4 rounded-xl border-2 transition-all font-medium text-sm',
                    'bg-white/60 dark:bg-gray-900/40 backdrop-blur-sm shadow-sm hover:shadow-md',
                    modality === 'report'
                      ? 'border-green-500 ring-2 ring-green-500/50 text-green-700 dark:text-green-300 shadow-lg'
                      : (!modality && categoryTouched ? 'border-red-500' : 'border-gray-300 dark:border-gray-700 hover:border-green-400'),
                    locked ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
                  )}
                >
                  Medical Report Analysis
                </button>
                <button
                  onClick={() => { setModality('combined'); setCategoryTouched(true); setError(null) }}
                  disabled={locked}
                  className={clsx(
                    'px-4 py-4 rounded-xl border-2 transition-all font-medium text-sm',
                    'bg-white/60 dark:bg-gray-900/40 backdrop-blur-sm shadow-sm hover:shadow-md',
                    modality === 'combined'
                      ? 'border-purple-500 ring-2 ring-purple-500/50 text-purple-700 dark:text-purple-300 shadow-lg'
                      : (!modality && categoryTouched ? 'border-red-500' : 'border-gray-300 dark:border-gray-700 hover:border-purple-400'),
                    locked ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
                  )}
                >
                  Combined Analysis
                </button>
              </div>
            </div>

            {/* Symptoms Textarea - Bigger */}
            <div className="space-y-2">
              <label className="text-base font-semibold">Symptoms & Clinical History</label>
              <Textarea
                value={symptoms}
                onChange={e => setSymptoms(e.target.value)}
                rows={12}
                placeholder="Describe patient symptoms, medical history, and any relevant clinical information... (optional but recommended for better analysis)"
                className="rounded-2xl border-2 bg-white/60 dark:bg-gray-900/50 backdrop-blur-sm shadow-sm focus:ring-2 focus:ring-blue-500 text-base p-4"
              />
            </div>

            {/* Report Upload Section (for Combined Analysis) */}
            {modality === 'combined' && (
              <div className="space-y-3 p-4 rounded-xl border-2 border-purple-500 bg-purple-50 dark:bg-purple-900/20">
                <label className="text-base font-semibold text-purple-700 dark:text-purple-300">
                  Upload Lab Report (PDF or Image)
                </label>
                <div
                  className={clsx(
                    'rounded-xl border-2 border-dashed p-6 text-center cursor-pointer transition-all',
                    'bg-white/80 dark:bg-gray-900/60 hover:bg-purple-100 dark:hover:bg-purple-900/30',
                    reportB64 ? 'border-purple-500' : 'border-purple-300'
                  )}
                  onClick={onPickReport}
                >
                  {reportB64 ? (
                    <div className="flex flex-col items-center space-y-2">
                      <FileText className="h-10 w-10 text-purple-600" />
                      <p className="text-sm font-medium text-purple-700 dark:text-purple-300">Report Uploaded ✓</p>
                      <p className="text-xs text-gray-600 dark:text-gray-400">{reportFileMeta}</p>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center space-y-2">
                      <UploadCloud className="h-8 w-8 text-purple-600" />
                      <p className="text-sm font-medium text-purple-700 dark:text-purple-300">Click to upload lab report</p>
                      <p className="text-xs text-gray-500">PDF or Image, up to 10MB</p>
                    </div>
                  )}
                  <input
                    ref={reportInputRef}
                    type="file"
                    accept="image/*,application/pdf"
                    className="hidden"
                    onChange={(e) => { const f = e.target.files?.[0]; if (f) onReportFile(f) }}
                  />
                </div>
                <p className="text-xs text-purple-700 dark:text-purple-300">
                  💡 Combined analysis correlates imaging findings with lab results for comprehensive diagnosis
                </p>
              </div>
            )}

            {/* Action Buttons - Bigger */}
            <div className="flex gap-3">
              <Button
                onClick={send}
                disabled={loading !== 'idle'}
                className="flex-1 h-12 text-base font-semibold"
              >
                {loading === 'idle' && (result || reportAnalysis || combinedResult ? 'Retry Analysis' : (modality === 'combined' ? 'Analyze Combined' : 'Analyze Image'))}
                {loading === 'uploading' && 'Uploading...'}
                {loading === 'classifying' && 'Analyzing...'}
                {loading === 'generating' && 'Generating Report...'}
                {loading === 'generating_detailed' && 'Generating Detailed Report...'}
                {loading === 'combined' && 'Running Combined Analysis...'}
              </Button>
              <Button
                variant="outline"
                className="h-12 px-6 text-base"
                onClick={() => {
                  try { localStorage.removeItem('hl:last') } catch { }
                  setImageB64(null)
                  setReportB64(null)
                  setResult(null)
                  setDetailedReport(null)
                  setShowDetailedReport(false)
                  setReportAnalysis(null)
                  setCombinedResult(null)
                  setSymptoms('')
                  setFileMeta('')
                  setReportFileMeta('')
                  setDragOver(false)
                  setError(null)
                  setModality(null)
                  setCategoryTouched(false)
                  setFileType('image')
                  setReportFileType('pdf')
                  if (inputRef.current) inputRef.current.value = ''
                  if (reportInputRef.current) reportInputRef.current.value = ''
                }}
              >
                Reset
              </Button>
            </div>
            {error && (
              <div className="text-red-600 font-medium bg-red-50 dark:bg-red-900/20 p-4 rounded-lg border-2 border-red-500">
                ⚠️ {error}
              </div>
            )}
          </div>
        </div>

        {/* Reports Section - Below Image + Prompt */}
        {/* Combined Analysis Results */}
        {combinedResult && (
          <div className="space-y-6">
            <div className="border-t-2 border-gray-300 dark:border-gray-700 pt-6">
              <h2 className="text-2xl font-bold mb-6 text-center">Combined Medical Analysis</h2>
              <div className="space-y-6">
                <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }}>
                  <Card className="p-6 border-2 border-purple-500">
                    {/* Disclaimer */}
                    <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border-2 border-red-500 rounded-lg">
                      <p className="text-sm font-bold text-red-700 dark:text-red-300 mb-2">⚠️ RESEARCH & EDUCATIONAL USE ONLY</p>
                      <p className="text-xs text-red-600 dark:text-red-400">This combined analysis is AI-generated for research purposes only. NOT for clinical use. All medical decisions require professional evaluation.</p>
                    </div>

                    {/* Image Analysis Summary */}
                    {combinedResult.image_analysis && (
                      <div className="mb-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-300 dark:border-blue-700">
                        <h3 className="font-semibold text-lg mb-3 text-blue-700 dark:text-blue-300">Imaging Findings</h3>
                        <div className="space-y-2">
                          <div className="flex justify-between items-center">
                            <span className="font-medium">Primary Diagnosis:</span>
                            <span className="text-blue-700 dark:text-blue-300 font-bold">
                              {combinedResult.image_analysis.top_diagnoses[0].label}
                            </span>
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="font-medium">Confidence:</span>
                            <span className="text-blue-700 dark:text-blue-300 font-bold">
                              {(combinedResult.image_analysis.top_diagnoses[0].prob * 100).toFixed(1)}%
                            </span>
                          </div>
                          {combinedResult.image_analysis.top_diagnoses.length > 1 && (
                            <div className="mt-3 pt-3 border-t border-blue-200 dark:border-blue-800">
                              <p className="text-sm font-medium mb-2">Differential Diagnoses:</p>
                              <div className="space-y-1">
                                {combinedResult.image_analysis.top_diagnoses.slice(1).map((dx: any, i: number) => (
                                  <div key={i} className="text-sm flex justify-between">
                                    <span>{dx.label}</span>
                                    <span className="text-blue-600 dark:text-blue-400">{(dx.prob * 100).toFixed(1)}%</span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    )}

                    {/* Lab Report Summary */}
                    {combinedResult.report_analysis && (
                      <div className="mb-6 p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-300 dark:border-green-700">
                        <h3 className="font-semibold text-lg mb-3 text-green-700 dark:text-green-300">Laboratory Findings</h3>
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span>Text Extracted:</span>
                            <span className="font-medium">{combinedResult.report_analysis.num_lines} lines</span>
                          </div>
                          <div className="flex justify-between">
                            <span>OCR Confidence:</span>
                            <span className="font-medium">{(combinedResult.report_analysis.confidence * 100).toFixed(1)}%</span>
                          </div>
                          {combinedResult.report_analysis.extracted_text && (
                            <div className="mt-3 pt-3 border-t border-green-200 dark:border-green-800">
                              <p className="font-medium mb-2">Extracted Text Preview:</p>
                              <div className="bg-white dark:bg-gray-800 p-3 rounded text-xs font-mono max-h-32 overflow-y-auto">
                                {combinedResult.report_analysis.extracted_text}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    )}

                    {/* Comprehensive Combined Analysis */}
                    {combinedResult.combined_analysis && (
                      <div>
                        <h3 className="font-semibold text-xl mb-4 text-purple-700 dark:text-purple-300">
                          Integrated Medical Analysis
                        </h3>
                        <div className="bg-gradient-to-br from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20 p-6 rounded-lg border-2 border-purple-300 dark:border-purple-700">
                          <div className="prose prose-sm max-w-none dark:prose-invert whitespace-pre-wrap">
                            {combinedResult.combined_analysis}
                          </div>
                        </div>
                      </div>
                    )}
                  </Card>
                </motion.div>
              </div>
            </div>
          </div>
        )}

        {/* Medical Report Analysis Results */}
        {reportAnalysis && (
          <div className="space-y-6">
            <div className="border-t-2 border-gray-300 dark:border-gray-700 pt-6">
              <h2 className="text-2xl font-bold mb-6 text-center">Medical Report Analysis</h2>
              <div className="space-y-6">
                <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }}>
                  <Card className="p-6 border-2 border-green-500">
                    {/* Disclaimer */}
                    <div className="mb-6 p-4 bg-yellow-50 dark:bg-yellow-900/20 border-2 border-yellow-500 rounded-lg">
                      <p className="text-sm font-bold text-yellow-700 dark:text-yellow-300 mb-2">⚠️ AI-GENERATED ANALYSIS</p>
                      <p className="text-xs text-yellow-600 dark:text-yellow-400">This analysis is for research and educational purposes only. Always consult qualified healthcare professionals for medical decisions.</p>
                    </div>

                    {/* OCR Info */}
                    {reportAnalysis.ocr_result && (
                      <div className="mb-6">
                        <div className="flex items-center justify-between mb-2">
                          <h3 className="font-semibold text-lg">Extracted Text</h3>
                          <div className="text-xs text-gray-500">
                            {reportAnalysis.ocr_result.num_lines} lines • {(reportAnalysis.ocr_result.confidence * 100).toFixed(1)}% confidence
                          </div>
                        </div>
                        <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg text-sm font-mono max-h-64 overflow-y-auto whitespace-pre-wrap">
                          {reportAnalysis.ocr_result.extracted_text}
                        </div>
                      </div>
                    )}

                    {/* LLM Comprehensive Analysis */}
                    {reportAnalysis.llm_analysis && (
                      <div>
                        <h3 className="font-semibold text-xl mb-4">
                          AI Medical Analysis
                        </h3>
                        <div className="bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 p-6 rounded-lg border-2 border-blue-200 dark:border-blue-700">
                          <div className="prose prose-sm max-w-none dark:prose-invert whitespace-pre-wrap">
                            {reportAnalysis.llm_analysis}
                          </div>
                        </div>
                      </div>
                    )}
                  </Card>
                </motion.div>
              </div>
            </div>
          </div>
        )}

        {/* Disease/XRay Diagnosis Results */}
        {result && (
          <div className="space-y-6">
            <div className="border-t-2 border-gray-300 dark:border-gray-700 pt-6">
              <h2 className="text-2xl font-bold mb-6 text-center">Analysis Results</h2>
              <div className="space-y-6">
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
                {result.report && !showDetailedReport && (
                  <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35, delay: 0.05 }}>
                    <Card className="p-4">
                      <div className="flex items-center justify-between mb-4">
                        <h3 className="font-semibold text-lg">Patient Report</h3>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={fetchDetailedReport}
                          disabled={loading === 'generating_detailed'}
                          className="gap-2"
                        >
                          <FileText className="h-4 w-4" />
                          {loading === 'generating_detailed' ? 'Generating...' : 'Show Detailed Report (For Doctors)'}
                        </Button>
                      </div>
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

                {/* Detailed Medical Report for Healthcare Professionals */}
                {showDetailedReport && detailedReport?.detailed_report && (
                  <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }}>
                    <Card className="p-6 border-2 border-blue-500">
                      <div className="flex items-center justify-between mb-4">
                        <div>
                          <h3 className="font-bold text-xl text-blue-700 dark:text-blue-300">Detailed Medical Report</h3>
                          <p className="text-xs text-blue-600 dark:text-blue-400">For Healthcare Professionals Only</p>
                        </div>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => setShowDetailedReport(false)}
                        >
                          Show Patient Report
                        </Button>
                      </div>

                      {/* Prominent Disclaimer */}
                      <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border-2 border-red-500 rounded-lg">
                        <p className="text-sm font-bold text-red-700 dark:text-red-300 mb-2">⚠️ RESEARCH & EDUCATIONAL USE ONLY</p>
                        <p className="text-xs text-red-600 dark:text-red-400">
                          {detailedReport.detailed_report.disclaimer}
                        </p>
                      </div>

                      <div className="space-y-5 text-sm">
                        <div>
                          <div className="font-semibold text-lg text-blue-700 dark:text-blue-300 mb-2">Clinical Impression</div>
                          <div className="whitespace-pre-wrap bg-blue-50 dark:bg-blue-900/10 p-3 rounded-lg">
                            {formatMaybeArray(detailedReport.detailed_report.clinical_impression)}
                          </div>
                        </div>

                        <div>
                          <div className="font-semibold text-lg mb-2">Detailed Findings</div>
                          <ul className="list-disc pl-5 space-y-1">
                            {toArray(detailedReport.detailed_report.detailed_findings).map((f: string, i: number) => (
                              <li key={i}>{f}</li>
                            ))}
                          </ul>
                        </div>

                        <div>
                          <div className="font-semibold text-lg mb-2">Pathophysiology</div>
                          <div className="whitespace-pre-wrap bg-gray-50 dark:bg-gray-800 p-3 rounded-lg">
                            {formatMaybeArray(detailedReport.detailed_report.pathophysiology)}
                          </div>
                        </div>

                        <div>
                          <div className="font-semibold text-lg mb-2">Differential Diagnosis</div>
                          <ol className="list-decimal pl-5 space-y-1 font-mono text-xs">
                            {toArray(detailedReport.detailed_report.differential_diagnosis).map((d: string, i: number) => (
                              <li key={i}>{d}</li>
                            ))}
                          </ol>
                        </div>

                        <div>
                          <div className="font-semibold text-lg mb-2">Recommended Diagnostic Tests</div>
                          <ul className="list-disc pl-5 space-y-1">
                            {toArray(detailedReport.detailed_report.recommended_tests).map((t: string, i: number) => (
                              <li key={i}>{t}</li>
                            ))}
                          </ul>
                        </div>

                        <div>
                          <div className="font-semibold text-lg mb-2">Treatment Considerations</div>
                          <ul className="list-disc pl-5 space-y-1">
                            {toArray(detailedReport.detailed_report.treatment_considerations).map((t: string, i: number) => (
                              <li key={i}>{t}</li>
                            ))}
                          </ul>
                        </div>

                        <div>
                          <div className="font-semibold text-lg mb-2">Prognosis</div>
                          <div className="whitespace-pre-wrap bg-green-50 dark:bg-green-900/10 p-3 rounded-lg">
                            {formatMaybeArray(detailedReport.detailed_report.prognosis)}
                          </div>
                        </div>

                        <div>
                          <div className="font-semibold text-lg text-red-700 dark:text-red-300 mb-2">🚨 Red Flags</div>
                          <ul className="list-disc pl-5 space-y-1 text-red-700 dark:text-red-400 bg-red-50 dark:bg-red-900/10 p-3 rounded-lg">
                            {toArray(detailedReport.detailed_report.red_flags).map((f: string, i: number) => (
                              <li key={i}>{f}</li>
                            ))}
                          </ul>
                        </div>

                        <div>
                          <div className="font-semibold text-lg mb-2">Follow-up Plan</div>
                          <div className="whitespace-pre-wrap bg-yellow-50 dark:bg-yellow-900/10 p-3 rounded-lg">
                            {formatMaybeArray(detailedReport.detailed_report.follow_up)}
                          </div>
                        </div>

                        <div>
                          <div className="font-semibold text-lg mb-2">Clinical Guidelines Reference</div>
                          <div className="whitespace-pre-wrap bg-gray-50 dark:bg-gray-800 p-3 rounded-lg text-xs">
                            {formatMaybeArray(detailedReport.detailed_report.references)}
                          </div>
                        </div>

                        <div className="mt-4 p-3 bg-gray-100 dark:bg-gray-800 rounded-lg text-xs text-gray-600 dark:text-gray-400">
                          <p><strong>Generation Model:</strong> {detailedReport.generation_model}</p>
                          <p className="mt-1"><strong>Note:</strong> This report was AI-generated and requires validation by qualified healthcare professionals.</p>
                        </div>
                      </div>
                    </Card>
                  </motion.div>
                )}
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="px-4 py-3 text-center border-t backdrop-blur bg-white/60 dark:bg-gray-900/50">
        <p className="text-xs text-gray-500 mb-1">⚠️ <strong>RESEARCH DEMONSTRATION ONLY</strong></p>
        <p className="text-xs text-gray-500">This is an AI research tool and NOT a medical device. Not for clinical diagnosis or treatment decisions. Always consult qualified healthcare professionals.</p>
      </footer>
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
