import { useCallback } from 'react'
import { useAnalysisStore } from '@/stores'
import { SUPPORTED_FORMATS, MAX_FILE_SIZE } from '@/lib/constants'
import type { AudioFile } from '@/types'

export function FileDropzone() {
    const { addFiles } = useAnalysisStore()

    const handleFiles = useCallback(
        (fileList: FileList | null) => {
            if (!fileList) return

            const audioFiles: AudioFile[] = Array.from(fileList)
                .filter((f) => {
                    const ext = f.name.split('.').pop()?.toLowerCase() ?? ''
                    return SUPPORTED_FORMATS.includes(ext) && f.size <= MAX_FILE_SIZE
                })
                .map((f) => ({
                    id: crypto.randomUUID(),
                    name: f.name,
                    path: '', // Will be set after upload
                    size: f.size,
                    format: f.name.split('.').pop()?.toLowerCase() ?? '',
                    status: 'pending' as const,
                    progress: 0,
                    stage: '',
                }))

            if (audioFiles.length > 0) addFiles(audioFiles)
        },
        [addFiles]
    )

    return (
        <div className="flex items-center justify-center h-full">
            <label
                className="
          w-full max-w-xl aspect-[16/9]
          flex flex-col items-center justify-center gap-4
          rounded-2xl border-2 border-dashed border-[var(--color-border)]
          bg-[var(--color-bg-card)]
          hover:border-[var(--color-primary)] hover:bg-[var(--color-bg-elevated)]
          transition-all duration-200 cursor-pointer group
        "
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                    e.preventDefault()
                    handleFiles(e.dataTransfer.files)
                }}
            >
                {/* Icon */}
                <div className="w-16 h-16 rounded-full bg-[var(--color-bg-elevated)] group-hover:bg-[var(--color-primary)]/10 flex items-center justify-center transition-colors">
                    <svg className="w-8 h-8 text-[var(--color-text-dim)] group-hover:text-[var(--color-primary)] transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 8.25H7.5a2.25 2.25 0 0 0-2.25 2.25v9a2.25 2.25 0 0 0 2.25 2.25h9a2.25 2.25 0 0 0 2.25-2.25v-9a2.25 2.25 0 0 0-2.25-2.25H15m0-3-3-3m0 0-3 3m3-3V15" />
                    </svg>
                </div>

                <div className="text-center">
                    <p className="text-lg font-medium text-[var(--color-text)]">
                        Drop audio files here
                    </p>
                    <p className="text-sm text-[var(--color-text-dim)] mt-1">
                        or click to browse — MP3, WAV, FLAC, OGG, M4A, AAC
                    </p>
                </div>

                <input
                    type="file"
                    accept="audio/*"
                    multiple
                    className="hidden"
                    onChange={(e) => handleFiles(e.target.files)}
                />
            </label>
        </div>
    )
}
