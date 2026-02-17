import { FileDropzone } from '@/components/audio/file-dropzone'
import { AnalysisPanel } from '@/components/analysis/analysis-panel'
import { Sidebar } from '@/components/layout/sidebar'
import { useAnalysisStore } from '@/stores'

function App() {
  const { files, activeFileId } = useAnalysisStore()
  const activeFile = files.find((f) => f.id === activeFileId)

  return (
    <div className="flex h-screen w-screen bg-[var(--color-bg)]">
      {/* Sidebar */}
      <Sidebar />

      {/* Main Content */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Title Bar */}
        <header className="h-12 flex items-center justify-between px-6 border-b border-[var(--color-border)]">
          <div className="flex items-center gap-3">
            <h1 className="text-lg font-semibold tracking-tight bg-gradient-to-r from-[var(--color-primary)] to-[var(--color-accent)] bg-clip-text text-transparent">
              Harmony
            </h1>
            <span className="text-xs text-[var(--color-text-dim)] font-mono">v0.1.0</span>
          </div>
        </header>

        {/* Content Area */}
        <div className="flex-1 overflow-y-auto p-6">
          {activeFile?.result ? (
            <AnalysisPanel result={activeFile.result} fileName={activeFile.name} />
          ) : activeFile?.status === 'analyzing' ? (
            <div className="flex flex-col items-center justify-center h-full gap-4">
              <div className="w-12 h-12 border-3 border-[var(--color-primary)] border-t-transparent rounded-full animate-spin" />
              <p className="text-[var(--color-text-muted)]">
                {activeFile.stage || 'Analyzing...'}
              </p>
              <div className="w-64 h-1.5 bg-[var(--color-bg-elevated)] rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-[var(--color-primary)] to-[var(--color-accent)] rounded-full transition-all duration-300"
                  style={{ width: `${(activeFile.progress ?? 0) * 100}%` }}
                />
              </div>
            </div>
          ) : (
            <FileDropzone />
          )}
        </div>
      </main>
    </div>
  )
}

export default App
