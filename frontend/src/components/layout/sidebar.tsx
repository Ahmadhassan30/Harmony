import { useAnalysisStore } from '@/stores'

export function Sidebar() {
    const { files, activeFileId, setActiveFile } = useAnalysisStore()

    // Helper functions removed as they are no longer used in the FL Studio design


    return (
        <aside className="w-64 border-r border-[var(--color-border)] bg-[var(--color-bg-card)] flex flex-col h-full">
            {/* Branding */}
            <div className="h-14 flex items-center gap-3 px-4 border-b border-[var(--color-border)] bg-[var(--color-bg-elevated)]/50 backdrop-blur-sm">
                <img src="/logo.png" alt="Harmony" className="w-6 h-6 object-contain" />
                <span className="font-bold text-sm tracking-wide text-[var(--color-text)]">Harmony</span>
            </div>

            {/* Header */}
            <div className="h-10 flex items-center px-4 border-b border-[var(--color-border)] bg-[var(--color-bg-subtle)]">
                <span className="text-xs font-medium text-[var(--color-text-muted)] uppercase tracking-wider">
                    Files
                </span>
                <span className="ml-auto text-[10px] text-[var(--color-text-dim)] font-mono bg-[var(--color-bg-surface)] px-1.5 py-0.5 rounded">
                    {files.length}
                </span>
            </div>

            {/* File List - FL Studio Browser Style */}
            <div className="flex-1 overflow-y-auto bg-[var(--color-bg-dark)]">
                {files.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-40 text-[var(--color-text-dim)] gap-2">
                        <span className="text-3xl opacity-20">📂</span>
                        <p className="text-[11px] font-mono">BROWSER EMPTY</p>
                    </div>
                ) : (
                    <div className="flex flex-col">
                        {files.map((file) => (
                            <button
                                key={file.id}
                                onClick={() => setActiveFile(file.id)}
                                className={`
                                    w-full pl-3 pr-2 py-[2px] flex items-center gap-2 text-left group
                                    border-b border-[var(--color-border-subtle)]/30 hover:bg-[var(--color-bg-hover)]
                                    ${activeFileId === file.id
                                        ? 'bg-[var(--color-primary-dim)]/20 text-[var(--color-primary)] border-l-2 border-l-[var(--color-primary)]'
                                        : 'text-[var(--color-text)] border-l-2 border-l-transparent'}
                                `}
                            >
                                {/* FL-style folder/file icon */}
                                <span className={`text-[10px] ${activeFileId === file.id ? 'opacity-100' : 'opacity-40 group-hover:opacity-70'}`}>
                                    {file.status === 'complete' ? '🎵' : '📄'}
                                </span>

                                <span className="truncate flex-1 text-[11px] font-medium tracking-tight font-sans">
                                    {file.name}
                                </span>

                                {/* Status indicators (FL LED style) */}
                                <div className="flex items-center gap-1.5">
                                    {file.status === 'analyzing' && (
                                        <div className="w-1.5 h-1.5 rounded-full bg-[var(--color-warning)] animate-pulse shadow-[0_0_4px_var(--color-warning)]" />
                                    )}
                                    {file.result?.bpm && (
                                        <span className={`text-[10px] font-mono px-1 rounded flex items-center justify-center min-w-[32px]
                                            ${activeFileId === file.id ? 'bg-[var(--color-primary)] text-[var(--color-text-on-primary)]' : 'bg-[var(--color-bg-elevated)] text-[var(--color-text-muted)]'}
                                        `}>
                                            {Math.round(file.result.bpm.bpm)}
                                        </span>
                                    )}
                                </div>
                            </button>
                        ))}
                    </div>
                )}
            </div>

            {/* Footer */}
            <div className="border-t border-[var(--color-border)] px-4 py-3">
                <p className="text-[10px] text-[var(--color-text-dim)] text-center">
                    Harmony Audio Engine v0.1.0
                </p>
            </div>
        </aside>
    )
}
