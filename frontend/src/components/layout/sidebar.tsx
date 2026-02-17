import { useAnalysisStore } from '@/stores'

export function Sidebar() {
    const { files, activeFileId, setActiveFile } = useAnalysisStore()

    const statusIcon = (status: string) => {
        switch (status) {
            case 'complete': return '✓'
            case 'analyzing': return '◌'
            case 'error': return '✗'
            default: return '○'
        }
    }

    const statusColor = (status: string) => {
        switch (status) {
            case 'complete': return 'text-[var(--color-success)]'
            case 'analyzing': return 'text-[var(--color-primary)] animate-pulse'
            case 'error': return 'text-[var(--color-danger)]'
            default: return 'text-[var(--color-text-dim)]'
        }
    }

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

            {/* File List */}
            <div className="flex-1 overflow-y-auto py-1">
                {files.length === 0 ? (
                    <p className="text-xs text-[var(--color-text-dim)] text-center py-8 px-4">
                        No files loaded yet
                    </p>
                ) : (
                    files.map((file) => (
                        <button
                            key={file.id}
                            onClick={() => setActiveFile(file.id)}
                            className={`
                w-full px-4 py-2.5 flex items-center gap-3 text-left
                hover:bg-[var(--color-bg-hover)] transition-colors text-sm
                ${activeFileId === file.id ? 'bg-[var(--color-bg-elevated)] border-r-2 border-[var(--color-primary)]' : ''}
              `}
                        >
                            <span className={`text-xs ${statusColor(file.status)}`}>
                                {statusIcon(file.status)}
                            </span>
                            <span className="truncate flex-1 text-[var(--color-text)]">
                                {file.name}
                            </span>
                            {file.result?.bpm && (
                                <span className="text-xs font-mono text-[var(--color-text-dim)]">
                                    {file.result.bpm.bpm}
                                </span>
                            )}
                        </button>
                    ))
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
