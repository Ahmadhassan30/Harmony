import { create } from 'zustand'
import { devtools } from 'zustand/middleware'
import type { AudioFile, AnalysisResponse } from '@/types'

interface AnalysisStore {
    // State
    files: AudioFile[]
    activeFileId: string | null
    isAnalyzing: boolean

    // Actions
    addFiles: (files: AudioFile[]) => void
    removeFile: (id: string) => void
    setActiveFile: (id: string | null) => void
    updateFileProgress: (id: string, progress: number, stage: string) => void
    setFileResult: (id: string, result: AnalysisResponse) => void
    setFileError: (id: string, error: string) => void
    clearAll: () => void
}

export const useAnalysisStore = create<AnalysisStore>()(
    devtools(
        (set) => ({
            files: [],
            activeFileId: null,
            isAnalyzing: false,

            addFiles: (newFiles) =>
                set((state) => ({
                    files: [...state.files, ...newFiles],
                    activeFileId: state.activeFileId ?? newFiles[0]?.id ?? null,
                })),

            removeFile: (id) =>
                set((state) => ({
                    files: state.files.filter((f) => f.id !== id),
                    activeFileId: state.activeFileId === id ? null : state.activeFileId,
                })),

            setActiveFile: (id) => set({ activeFileId: id }),

            updateFileProgress: (id, progress, stage) =>
                set((state) => ({
                    isAnalyzing: true,
                    files: state.files.map((f) =>
                        f.id === id
                            ? { ...f, status: 'analyzing' as const, progress, stage }
                            : f
                    ),
                })),

            setFileResult: (id, result) =>
                set((state) => ({
                    isAnalyzing: state.files.some(
                        (f) => f.id !== id && f.status === 'analyzing'
                    ),
                    files: state.files.map((f) =>
                        f.id === id
                            ? { ...f, status: 'complete' as const, progress: 1, result }
                            : f
                    ),
                })),

            setFileError: (id, error) =>
                set((state) => ({
                    isAnalyzing: state.files.some(
                        (f) => f.id !== id && f.status === 'analyzing'
                    ),
                    files: state.files.map((f) =>
                        f.id === id
                            ? { ...f, status: 'error' as const, error }
                            : f
                    ),
                })),

            clearAll: () => set({ files: [], activeFileId: null, isAnalyzing: false }),
        }),
        { name: 'harmony-analysis' }
    )
)
