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



