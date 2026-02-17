import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface SettingsStore {
    // Analysis settings
    enableSeparation: boolean
    enableExtended: boolean

    // UI settings
    showAlgorithmBreakdown: boolean
    showSpectrogram: boolean
    theme: 'dark' | 'light'

    // API
    apiUrl: string

    // Actions
    setEnableSeparation: (v: boolean) => void
    setEnableExtended: (v: boolean) => void
    setShowAlgorithmBreakdown: (v: boolean) => void
    setShowSpectrogram: (v: boolean) => void
    setApiUrl: (url: string) => void
}

export const useSettingsStore = create<SettingsStore>()(
    persist(
        (set) => ({
            enableSeparation: true,
            enableExtended: true,
            showAlgorithmBreakdown: false,
            showSpectrogram: true,
            theme: 'dark',
            apiUrl: 'http://localhost:8000',

            setEnableSeparation: (v) => set({ enableSeparation: v }),
            setEnableExtended: (v) => set({ enableExtended: v }),
            setShowAlgorithmBreakdown: (v) => set({ showAlgorithmBreakdown: v }),
            setShowSpectrogram: (v) => set({ showSpectrogram: v }),
            setApiUrl: (url) => set({ apiUrl: url }),
        }),
        { name: 'harmony-settings' }
    )
)
