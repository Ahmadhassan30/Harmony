"""Export endpoints — download results in various formats."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/export/{job_id}/{format}")
async def export_results(job_id: str, format: str) -> dict[str, str]:
    """Export analysis results in the specified format.

    Supported formats: json, csv, rekordbox_xml, traktor_nml
    """
    supported = {"json", "csv", "rekordbox_xml", "traktor_nml"}
    if format not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{format}'. Supported: {supported}",
        )

    # TODO: Retrieve results from cache/db and serialize
    return {"status": "not_implemented", "format": format, "job_id": job_id}
