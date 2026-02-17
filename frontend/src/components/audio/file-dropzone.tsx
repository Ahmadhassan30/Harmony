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
          w-full max-w-2xl aspect-[3/1]
          flex flex-col items-center justify-center gap-6
          rounded-sm border border-dashed border-[var(--color-border-highlight)]
          bg-[var(--color-bg-dark)]
          hover:bg-[var(--color-bg-hover)]
          transition-all duration-200 cursor-pointer group
          select-none
        "
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                    e.preventDefault()
                    handleFiles(e.dataTransfer.files)
                }}
            >
                {/* Icon */}
                <div className="w-12 h-12 bg-[var(--color-bg-elevated)] border border-[var(--color-border-subtle)] flex items-center justify-center group-hover:border-[var(--color-primary)] transition-colors">
                    <span className="text-2xl text-[var(--color-text-dim)] group-hover:text-[var(--color-primary)] transition-colors">⬇</span>
                </div>

                <div className="text-center font-mono">
                    <p className="text-sm font-bold tracking-widest text-[var(--color-text-muted)] uppercase group-hover:text-[var(--color-primary)] transition-colors">
                        Drop Sample Here to Analyze
                    </p>
                    <p className="text-[10px] text-[var(--color-text-dim)] mt-2 uppercase tracking-wide opacity-60">
                        Supports MP3 • WAV • FLAC • OGG • M4A
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
