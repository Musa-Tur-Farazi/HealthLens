export type DiagRequest = {
  modality: 'disease' | 'xray'
  symptoms?: string
  topk?: number
  include_cam?: boolean
  image_b64: string
}

export async function diag(req: DiagRequest, signal?: AbortSignal) {
  const base = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'
  const url = `${base}/v1/diag`

  const doFetch = async () => {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ topk: 3, include_cam: true, ...req }),
      signal,
    })
    if (!res.ok) {
      const text = await res.text().catch(() => '')
      throw new Error(`API ${res.status}: ${text || res.statusText}`)
    }
    return res.json()
  }

  try {
    return await doFetch()
  } catch (e: any) {
    if (/^(502|504)/.test(e?.message || '')) {
      return await doFetch() // simple one retry
    }
    throw e
  }
}

export async function getDetailedReport(payload: any, signal?: AbortSignal) {
  const base = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'
  const url = `${base}/v1/detailed_report`

  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ payload }),
    signal,
  })

  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`API ${res.status}: ${text || res.statusText}`)
  }

  return res.json()
}

export type ReportAnalysisRequest = {
  file_data: string  // base64
  file_type: 'image' | 'pdf'
  query?: string
}

export async function analyzeReport(req: ReportAnalysisRequest, signal?: AbortSignal) {
  const base = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'
  const url = `${base}/v1/analyze_report`

  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
    signal,
  })

  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`API ${res.status}: ${text || res.statusText}`)
  }

  return res.json()
}

export type CombinedAnalysisRequest = {
  image_b64: string  // Medical image
  report_data: string  // base64 PDF or image
  report_type: 'image' | 'pdf'
  modality: 'disease' | 'xray'
  symptoms?: string
  query?: string
}

export async function combinedAnalysis(req: CombinedAnalysisRequest, signal?: AbortSignal) {
  const base = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'
  const url = `${base}/v1/combined_analysis`

  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
    signal,
  })

  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`API ${res.status}: ${text || res.statusText}`)
  }

  return res.json()
}



