import { motion } from 'framer-motion'
import type { AnalysisResponse } from '@/types'
import { camelotColor } from '@/lib/camelot'

interface AnalysisPanelProps {
    result: AnalysisResponse
    fileName: string
}

export function AnalysisPanel({ result, fileName }: AnalysisPanelProps) {
    const { bpm, key, loudness } = result

    return (
        <div className="max-w-4xl mx-auto space-y-4 font-sans select-none">
            {/* Plugin Header Style */}
            <div className="bg-[var(--color-bg-elevated)] border-t border-l border-[var(--color-border-highlight)] border-b border-r border-[var(--color-border-subtle)] px-4 py-2 rounded-sm flex items-center justify-between shadow-md">
                <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-[var(--color-primary)] shadow-[0_0_8px_var(--color-primary)]"></div>
                    <h2 className="text-sm font-bold tracking-widest text-[var(--color-text)] uppercase drop-shadow-md">
                        {fileName}
                    </h2>
                </div>
                <div className="flex items-center gap-4 text-[10px] font-mono text-[var(--color-text-dim)]">
                    <span>{(result.duration_seconds / 60).toFixed(0)}:{(result.duration_seconds % 60).toFixed(0).padStart(2, '0')}</span>
                    <span>16-BIT</span>
                    <span>44.1KHZ</span>
                </div>
            </div>

            {/* Main Rack Grid */}
            <div className="grid grid-cols-12 gap-1 bg-[var(--color-bg-card)] p-1 rounded-sm border border-[var(--color-border-subtle)] shadow-inner">

                {/* BPM Module */}
                <div className="col-span-12 md:col-span-4 bg-[var(--color-bg-dark)] border border-[var(--color-border-subtle)] rounded-sm p-4 relative group hover:border-[var(--color-border-highlight)] transition-colors">
                    <p className="text-[10px] text-[var(--color-primary)] font-bold tracking-widest mb-1 opacity-70">TEMPO</p>
                    {bpm ? (
                        <div className="flex flex-col items-center justify-center py-2 relative">
                            {/* Digital Display */}
                            <div className="bg-[#1a1a1a] border border-[#000] px-4 py-2 rounded-sm shadow-inner min-w-[120px] text-center mb-2">
                                <span className="text-4xl font-mono font-bold text-[var(--color-primary)] drop-shadow-[0_0_4px_rgba(250,142,62,0.4)]">
                                    {Math.round(bpm.bpm)}
                                </span>
                                <span className="text-[10px] text-[var(--color-primary)] block opacity-50 -mt-1 font-mono">
                                    .{bpm.bpm.toFixed(2).split('.')[1]} BPM
                                </span>
                            </div>

                            {/* Visual Confidence Meter */}
                            <div className="w-full h-1 bg-[#111] mt-2 rounded-full overflow-hidden">
                                <motion.div
                                    initial={{ width: 0 }}
                                    animate={{ width: `${bpm.confidence * 100}%` }}
                                    className="h-full bg-[var(--color-primary)] shadow-[0_0_8px_var(--color-primary)]"
                                />
                            </div>
                        </div>
                    ) : <div className="h-20 flex items-center justify-center text-[var(--color-text-dim)] text-xs">--</div>}
                </div>

                {/* Key Module */}
                <div className="col-span-12 md:col-span-4 bg-[var(--color-bg-dark)] border border-[var(--color-border-subtle)] rounded-sm p-4 relative hover:border-[var(--color-border-highlight)] transition-colors">
                    <p className="text-[10px] text-[var(--color-accent)] font-bold tracking-widest mb-1 opacity-70">KEY & SCALE</p>
                    {key ? (
                        <div className="flex flex-col items-center justify-center py-2">
                            {/* Digital Display */}
                            <div className="bg-[#1a1a1a] border border-[#000] px-4 py-2 rounded-sm shadow-inner min-w-[120px] text-center mb-2">
                                <span className="text-4xl font-mono font-bold text-[var(--color-accent)] drop-shadow-[0_0_4px_rgba(153,229,80,0.4)]">
                                    {key.key}
                                </span>
                                <span className="text-[10px] text-[var(--color-accent)] block opacity-50 -mt-1 font-mono uppercase">
                                    {key.mode}
                                </span>
                            </div>

                            {/* Camelot Badge */}
                            <div className="flex items-center gap-2 mt-2 bg-[#222] px-2 py-1 rounded-sm border border-[#333]">
                                <div
                                    className="w-2 h-2 rounded-full"
                                    style={{ backgroundColor: camelotColor(key.camelot), boxShadow: `0 0 6px ${camelotColor(key.camelot)}` }}
                                />
                                <span className="text-[10px] font-mono text-[var(--color-text-muted)] tracking-wider">
                                    {key.camelot}
                                </span>
                            </div>
                        </div>
                    ) : <div className="h-20 flex items-center justify-center text-[var(--color-text-dim)] text-xs">--</div>}
                </div>

                {/* Loudness Module */}
                <div className="col-span-12 md:col-span-4 bg-[var(--color-bg-dark)] border border-[var(--color-border-subtle)] rounded-sm p-4 hover:border-[var(--color-border-highlight)] transition-colors">
                    <p className="text-[10px] text-[var(--color-info)] font-bold tracking-widest mb-1 opacity-70">LEVELS</p>
                    {loudness ? (
                        <div className="flex flex-col gap-3 py-2">
                            <div className="flex items-center justify-between">
                                <span className="text-[10px] text-[var(--color-text-dim)] font-mono">INTEGRATED</span>
                                <span className="text-lg font-mono font-bold text-[var(--color-info)]">
                                    {loudness.integrated_lufs.toFixed(1)} <span className="text-[10px] opacity-50">LUFS</span>
                                </span>
                            </div>
                            <div className="flex items-center justify-between">
                                <span className="text-[10px] text-[var(--color-text-dim)] font-mono">RANGE</span>
                                <span className="text-lg font-mono font-bold text-[var(--color-info)]">
                                    {loudness.loudness_range.toFixed(1)} <span className="text-[10px] opacity-50">LU</span>
                                </span>
                            </div>
                        </div>
                    ) : <div className="h-20 flex items-center justify-center text-[var(--color-text-dim)] text-xs">--</div>}
                </div>
            </div>

            {/* Algorithm Details (Collapsible Rack) */}
            {(bpm?.algorithm_results?.length ?? 0) > 0 && (
                <div className="bg-[var(--color-bg-card)] border border-[var(--color-border-subtle)] rounded-sm overflow-hidden">
                    <div className="bg-[var(--color-bg-elevated)] px-4 py-1.5 border-b border-[var(--color-border-subtle)] flex items-center justify-between">
                        <span className="text-[10px] font-bold text-[var(--color-text-muted)] uppercase tracking-wider">Detection Engine Details</span>
                    </div>
                    <div className="p-1 grid grid-cols-1 gap-px bg-[var(--color-border-subtle)]">
                        {bpm?.algorithm_results.map((algo, i) => (
                            <div key={i} className="bg-[var(--color-bg-dark)] flex items-center justify-between px-3 py-2">
                                <span className="text-[11px] font-medium text-[var(--color-text)] font-sans">{algo.algorithm}</span>
                                <div className="flex items-center gap-3">
                                    <div className="w-16 h-1 bg-[#111] rounded-full overflow-hidden">
                                        <div
                                            className="h-full bg-[var(--color-primary-dim)]"
                                            style={{ width: `${algo.confidence * 100}%` }}
                                        />
                                    </div>
                                    <span className="text-[10px] font-mono text-[var(--color-text-dim)] w-8 text-right">
                                        {(algo.confidence * 100).toFixed(0)}%
                                    </span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    )
}
