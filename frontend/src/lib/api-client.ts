/**
 * Typed HTTP + WebSocket client for the Harmony backend API.
 */

import type { AnalysisResponse, ProgressUpdate } from '@/types'

const DEFAULT_API_URL = 'http://localhost:8000'

export class HarmonyClient {
    private apiUrl: string

    constructor(apiUrl: string = DEFAULT_API_URL) {
        this.apiUrl = apiUrl
    }

    /** Health check. */
    async health(): Promise<{ status: string }> {
        const res = await fetch(`${this.apiUrl}/health`)
        return res.json()
    }

    /** Upload an audio file. */
    async uploadFile(file: File): Promise<{ file_id: string; file_path: string }> {
        const formData = new FormData()
        formData.append('audio', file)

        const res = await fetch(`${this.apiUrl}/api/v1/upload`, {
            method: 'POST',
            body: formData,
        })

        if (!res.ok) throw new Error(`Upload failed: ${res.statusText}`)
        return res.json()
    }

    /** Run analysis on a file. */
    async analyze(
        filePath: string,
        options: { enableSeparation?: boolean; enableExtended?: boolean } = {}
    ): Promise<AnalysisResponse> {
        const res = await fetch(`${this.apiUrl}/api/v1/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                file_path: filePath,
                enable_separation: options.enableSeparation ?? true,
                enable_extended: options.enableExtended ?? true,
            }),
        })

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }))
            throw new Error(err.detail || 'Analysis failed')
        }

        return res.json()
    }

    /** Connect WebSocket for real-time progress updates. */
    connectProgress(
        jobId: string,
        onProgress: (update: ProgressUpdate) => void,
        onComplete: (result: AnalysisResponse) => void,
        onError: (error: string) => void
    ): WebSocket {
        const wsUrl = this.apiUrl.replace(/^http/, 'ws')
        const ws = new WebSocket(`${wsUrl}/ws/${jobId}`)

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data)

            switch (data.type) {
                case 'progress':
                    onProgress(data as ProgressUpdate)
                    break
                case 'complete':
                    onComplete(data.result as AnalysisResponse)
                    ws.close()
                    break
                case 'error':
                    onError(data.error)
                    ws.close()
                    break
            }
        }

        ws.onerror = () => onError('WebSocket connection error')

        return ws
    }
}

/** Singleton client instance. */
export const harmonyClient = new HarmonyClient()
