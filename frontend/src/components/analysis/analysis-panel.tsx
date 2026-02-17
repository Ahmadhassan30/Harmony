import { motion } from 'framer-motion'
import type { AnalysisResponse } from '@/types'
import { camelotColor } from '@/lib/camelot'

interface AnalysisPanelProps {
    result: AnalysisResponse
    fileName: string
}

export function AnalysisPanel({ result, fileName }: AnalysisPanelProps) {
    const { bpm, key, loudness, time_signature } = result

    return (
        <div className="space-y-6 max-w-4xl mx-auto">
            {/* File Header */}
            <div>
                <h2 className="text-xl font-semibold text-[var(--color-text)]">{fileName}</h2>
                <p className="text-sm text-[var(--color-text-dim)] mt-1">
                    {result.duration_seconds.toFixed(1)}s • Analyzed in {result.processing_time_seconds.toFixed(1)}s
                </p>
            </div>

            {/* Main Results Grid */}
            <div className="grid grid-cols-2 gap-4">
                {/* BPM Card */}
                {bpm && (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="bg-[var(--color-bg-card)] rounded-xl p-6 border border-[var(--color-border)]"
                    >
                        <p className="text-xs font-medium text-[var(--color-text-dim)] uppercase tracking-wider mb-3">
                            Tempo
                        </p>
                        <div className="flex items-baseline gap-2">
                            <span className="text-5xl font-bold tracking-tight bg-gradient-to-r from-[var(--color-primary)] to-[var(--color-primary-light)] bg-clip-text text-transparent">
                                {bpm.bpm.toFixed(1)}
                            </span>
                            <span className="text-lg text-[var(--color-text-dim)]">BPM</span>
                        </div>

                        {/* Confidence Bar */}
                        <div className="mt-4">
                            <div className="flex justify-between text-xs mb-1.5">
                                <span className="text-[var(--color-text-dim)]">Confidence</span>
                                <span className={`font-mono font-semibold ${bpm.confidence >= 0.95 ? 'text-[var(--color-success)]' :
                                        bpm.confidence >= 0.85 ? 'text-[var(--color-warning)]' :
                                            'text-[var(--color-danger)]'
                                    }`}>
                                    {(bpm.confidence * 100).toFixed(1)}%
                                </span>
                            </div>
                            <div className="w-full h-1.5 bg-[var(--color-bg-elevated)] rounded-full overflow-hidden">
                                <motion.div
                                    initial={{ width: 0 }}
                                    animate={{ width: `${bpm.confidence * 100}%` }}
                                    transition={{ duration: 0.8, ease: 'easeOut' }}
                                    className={`h-full rounded-full ${bpm.confidence >= 0.95 ? 'bg-[var(--color-success)]' :
                                            bpm.confidence >= 0.85 ? 'bg-[var(--color-warning)]' :
                                                'bg-[var(--color-danger)]'
                                        }`}
                                />
                            </div>
                        </div>

                        {!bpm.tempo_stable && (
                            <p className="mt-3 text-xs text-[var(--color-warning)] flex items-center gap-1.5">
                                ⚠ Variable tempo detected
                            </p>
                        )}
                    </motion.div>
                )}

                {/* Key Card */}
                {key && (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.1 }}
                        className="bg-[var(--color-bg-card)] rounded-xl p-6 border border-[var(--color-border)]"
                    >
                        <p className="text-xs font-medium text-[var(--color-text-dim)] uppercase tracking-wider mb-3">
                            Musical Key
                        </p>
                        <div className="flex items-baseline gap-3">
                            <span className="text-5xl font-bold tracking-tight text-[var(--color-text)]">
                                {key.key}
                            </span>
                            <span className="text-xl text-[var(--color-text-muted)]">{key.mode}</span>
                        </div>

                        {/* Camelot */}
                        <div className="mt-3 flex items-center gap-2">
                            <span
                                className="inline-flex items-center justify-center w-8 h-8 rounded-full text-xs font-bold text-white"
                                style={{ backgroundColor: camelotColor(key.camelot) }}
                            >
                                {key.camelot}
                            </span>
                            <span className="text-xs text-[var(--color-text-dim)]">Camelot</span>
                        </div>

                        {/* Confidence */}
                        <div className="mt-4">
                            <div className="flex justify-between text-xs mb-1.5">
                                <span className="text-[var(--color-text-dim)]">Confidence</span>
                                <span className="font-mono font-semibold text-[var(--color-text-muted)]">
                                    {(key.confidence * 100).toFixed(1)}%
                                </span>
                            </div>
                            <div className="w-full h-1.5 bg-[var(--color-bg-elevated)] rounded-full overflow-hidden">
                                <motion.div
                                    initial={{ width: 0 }}
                                    animate={{ width: `${key.confidence * 100}%` }}
                                    transition={{ duration: 0.8, ease: 'easeOut', delay: 0.1 }}
                                    className="h-full rounded-full bg-[var(--color-accent)]"
                                />
                            </div>
                        </div>
                    </motion.div>
                )}
            </div>

            {/* Extended Info Row */}
            <div className="grid grid-cols-3 gap-4">
                {time_signature && (
                    <div className="bg-[var(--color-bg-card)] rounded-lg p-4 border border-[var(--color-border)]">
                        <p className="text-xs text-[var(--color-text-dim)] uppercase tracking-wider mb-1">Time Sig</p>
                        <p className="text-2xl font-bold font-mono text-[var(--color-text)]">{time_signature}</p>
                    </div>
                )}
                {loudness && (
                    <>
                        <div className="bg-[var(--color-bg-card)] rounded-lg p-4 border border-[var(--color-border)]">
                            <p className="text-xs text-[var(--color-text-dim)] uppercase tracking-wider mb-1">Loudness</p>
                            <p className="text-2xl font-bold font-mono text-[var(--color-text)]">
                                {loudness.integrated_lufs.toFixed(1)}
                                <span className="text-sm text-[var(--color-text-dim)] ml-1">LUFS</span>
                            </p>
                        </div>
                        <div className="bg-[var(--color-bg-card)] rounded-lg p-4 border border-[var(--color-border)]">
                            <p className="text-xs text-[var(--color-text-dim)] uppercase tracking-wider mb-1">Dynamic Range</p>
                            <p className="text-2xl font-bold font-mono text-[var(--color-text)]">
                                {loudness.loudness_range.toFixed(1)}
                                <span className="text-sm text-[var(--color-text-dim)] ml-1">LU</span>
                            </p>
                        </div>
                    </>
                )}
            </div>

            {/* Algorithm Breakdown */}
            {(bpm?.algorithm_results?.length ?? 0) > 0 && (
                <details className="bg-[var(--color-bg-card)] rounded-xl border border-[var(--color-border)]">
                    <summary className="px-6 py-4 cursor-pointer text-sm font-medium text-[var(--color-text-muted)] hover:text-[var(--color-text)] transition-colors">
                        Algorithm Breakdown
                    </summary>
                    <div className="px-6 pb-4 space-y-2">
                        {bpm?.algorithm_results.map((algo, i) => (
                            <div
                                key={i}
                                className="flex items-center justify-between py-2 px-3 rounded-lg bg-[var(--color-bg-elevated)]"
                            >
                                <div>
                                    <span className="text-sm font-medium text-[var(--color-text)]">{algo.algorithm}</span>
                                    <span className="text-xs text-[var(--color-text-dim)] ml-2">{algo.method}</span>
                                </div>
                                <div className="flex items-center gap-4">
                                    <span className="font-mono text-sm text-[var(--color-text)]">
                                        {typeof algo.value === 'number' ? algo.value.toFixed(1) : algo.value}
                                    </span>
                                    <div className="w-20 h-1 bg-[var(--color-bg)] rounded-full overflow-hidden">
                                        <div
                                            className="h-full bg-[var(--color-primary)] rounded-full"
                                            style={{ width: `${algo.confidence * 100}%` }}
                                        />
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </details>
            )}
        </div>
    )
}
