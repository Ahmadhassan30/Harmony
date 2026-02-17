/**
 * Camelot wheel utilities for harmonic mixing.
 */

const CAMELOT_MAP: Record<string, string> = {
    'A_minor': '8A', 'E_minor': '9A', 'B_minor': '10A',
    'F#_minor': '11A', 'C#_minor': '12A', 'G#_minor': '1A',
    'D#_minor': '2A', 'A#_minor': '3A', 'F_minor': '4A',
    'C_minor': '5A', 'G_minor': '6A', 'D_minor': '7A',
    'C_major': '8B', 'G_major': '9B', 'D_major': '10B',
    'A_major': '11B', 'E_major': '12B', 'B_major': '1B',
    'F#_major': '2B', 'C#_major': '3B', 'G#_major': '4B',
    'D#_major': '5B', 'A#_major': '6B', 'F_major': '7B',
}

export function keyToCamelot(key: string, mode: string): string {
    return CAMELOT_MAP[`${key}_${mode}`] ?? '?'
}

export function getCompatibleKeys(camelot: string): string[] {
    if (camelot.length < 2) return []

    const num = parseInt(camelot.slice(0, -1))
    const letter = camelot.slice(-1)

    return [
        camelot,
        `${num}${letter === 'A' ? 'B' : 'A'}`,
        `${(num % 12) + 1}${letter}`,
        `${((num - 2 + 12) % 12) + 1}${letter}`,
    ]
}

/** Color for Camelot position (for visualization). */
export function camelotColor(camelot: string): string {
    const num = parseInt(camelot.slice(0, -1))
    const hue = ((num - 1) * 30) % 360
    return `hsl(${hue}, 70%, 60%)`
}
