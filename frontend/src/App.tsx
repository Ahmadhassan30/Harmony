import { FileDropzone } from '@/components/audio/file-dropzone'
import { AnalysisPanel } from '@/components/analysis/analysis-panel'
import { Sidebar } from '@/components/layout/sidebar'
import { useAnalysisStore } from '@/stores'

function App() {
  const { files, activeFileId } = useAnalysisStore()
  // const files = useAnalysisStore(state => state.files) // Alternative selector if needed
  const activeFile = files.find((f) => f.id === activeFileId)

  return (
    <div className="flex h-screen w-screen bg-[var(--color-bg)] text-[var(--color-text)] overflow-hidden font-sans select-none">

      {/* Sidebar (Browser) */}
      <Sidebar />

      {/* Main Workspace */}
      <main className="flex-1 flex flex-col min-w-0 bg-[#1e1e1e]">

        {/* FL-Style Toolbar */}
        <header className="h-10 bg-[var(--color-bg-elevated)] border-b border-[var(--color-border)] flex items-center px-4 justify-between shadow-lg z-10">

          {/* Left: Branding & status */}
          <div className="flex items-center gap-4">
            {/* Simple Transport Controls (Visual only) */}
            <div className="flex items-center gap-1">
              <div className="w-8 h-5 bg-[#333] border border-[#111] rounded-[2px] flex items-center justify-center hover:bg-[#444] cursor-pointer">
                <div className="w-0 h-0 border-l-[6px] border-l-[var(--color-accent)] border-y-[4px] border-y-transparent ml-1"></div>
              </div>
              <div className="w-8 h-5 bg-[#333] border border-[#111] rounded-[2px] flex items-center justify-center hover:bg-[#444] cursor-pointer">
                <div className="w-3 h-3 bg-[var(--color-danger)] rounded-[1px]"></div>
              </div>
            </div>

            <div className="h-6 w-px bg-[var(--color-border-subtle)] mx-2"></div>

            {/* Digital Clock / Status */}
            <div className="bg-[#111] px-3 py-0.5 border border-[#333] rounded-[2px] flex items-center gap-2 shadow-inner">
              <span className="text-[var(--color-accent)] font-mono text-xs tracking-wider font-bold">
                {activeFile ? "001:01:00" : "--:--:--"}
              </span>
            </div>
          </div>

          {/* Right: Window Controls / Info */}
          <div className="flex items-center gap-3">
            <div className="bg-[#2a2a2a] px-3 py-1 rounded-full border border-[var(--color-border-subtle)] flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-[var(--color-primary)] animate-pulse"></div>
              <span className="text-[10px] text-[var(--color-text-dim)] font-mono uppercase">
                CPU: 2% MEM: 140MB
              </span>
            </div>
          </div>
        </header>

        {/* Content Area (Playlist / Plugin View) */}
        <div className="flex-1 overflow-y-auto p-1 bg-[#1e1e1e] relative">
          {/* Background Grid Lines (visual flair) */}
          <div className="absolute inset-0 pointer-events-none opacity-5"
            style={{ backgroundImage: 'linear-gradient(#333 1px, transparent 1px), linear-gradient(90deg, #333 1px, transparent 1px)', backgroundSize: '20px 20px' }}>
          </div>

          <div className="relative z-0 h-full p-4">
            {activeFile?.result ? (
              <AnalysisPanel result={activeFile.result} fileName={activeFile.name} />
            ) : (
              <div className="h-full flex flex-col">
                {activeFile?.status === 'analyzing' ? (
                  <div className="flex flex-col items-center justify-center h-full gap-4">
                    <div className="w-12 h-12 border-3 border-[var(--color-primary)] border-t-transparent rounded-full animate-spin" />
                    <p className="text-[var(--color-text-muted)] font-mono text-xs uppercase tracking-widest">
                      {activeFile.stage || 'PROCESSING...'}
                    </p>
                    <div className="w-64 h-2 bg-[#111] rounded-full overflow-hidden border border-[#333]">
                      <div
                        className="h-full bg-[var(--color-primary)] shadow-[0_0_10px_var(--color-primary)] transition-all duration-300"
                        style={{ width: `${(activeFile.progress ?? 0) * 100}%` }}
                      />
                    </div>
                  </div>
                ) : (
                  <div className="flex-1 flex items-center justify-center border-2 border-dashed border-[var(--color-border)] rounded-lg m-4 bg-[var(--color-bg-card)]/50">
                    <FileDropzone />
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
