"""Universal dataset models for GEPA optimization."""

from typing import Any, Dict, Optional

from pydantic import BaseModel


class DatasetEntry(BaseModel):
    """Universal dataset entry for prompt optimization."""

    input: Dict[str, Any]
    expected: str
    metadata: Optional[Dict[str, Any]] = None
