export const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const SUPPORTED_FORMATS = [
    'mp3', 'wav', 'flac', 'ogg', 'm4a', 'aac', 'wma', 'aiff',
]

export const SUPPORTED_MIME_TYPES = [
    'audio/mpeg', 'audio/wav', 'audio/flac', 'audio/ogg',
    'audio/mp4', 'audio/aac', 'audio/x-ms-wma', 'audio/aiff',
]

/** Max file size in bytes (200 MB). */
export const MAX_FILE_SIZE = 200 * 1024 * 1024
