/* ── Analysis Result Types ──────────────────────────────────── */

export interface AlgorithmResult {
    algorithm: string
    value: number | string
    confidence: number
    method: string
}

export interface BPMResult {
    bpm: number
    confidence: number
    tempo_stable: boolean
    algorithm_results: AlgorithmResult[]
    tempo_curve?: number[]
}

export interface SecondaryKey {
    key: string
    mode: string
    camelot: string
    proportion: number
}

export interface KeyResult {
    key: string
    mode: string
    camelot: string
    confidence: number
    secondary_keys?: SecondaryKey[]
    algorithm_results: AlgorithmResult[]
}

export interface LoudnessResult {
    integrated_lufs: number
    short_term_lufs: number
    momentary_lufs: number
    loudness_range: number
    true_peak_dbtp: number
}

export interface InstrumentResult {
    bpm?: number
    key?: string
    mode?: string
    confidence: number
}

export interface AnalysisResponse {
    id: string
    status: string
    file_name: string
    duration_seconds: number
    processing_time_seconds: number
    bpm?: BPMResult
    key?: KeyResult
    loudness?: LoudnessResult
    time_signature?: string
    instruments?: Record<string, InstrumentResult>
}

/* ── Audio File Types ──────────────────────────────────────── */

export interface AudioFile {
    id: string
    name: string
    path: string
    size: number
    format: string
    duration?: number
    status: 'pending' | 'analyzing' | 'complete' | 'error'
    progress: number
    stage: string
    result?: AnalysisResponse
    error?: string
}

/* ── Progress Types ────────────────────────────────────────── */

export interface ProgressUpdate {
    job_id: string
    progress: number
    stage: string
    message: string
}
