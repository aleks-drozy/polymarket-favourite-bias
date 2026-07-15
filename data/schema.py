"""Canonical market/candle/exclusion models. Timestamps are unix seconds UTC."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, model_validator


class MarketRecord(BaseModel):
    market_id: str
    category: str
    created_ts: int
    resolved_ts: int
    resolved_outcome: Literal["YES", "NO"]
    volume: float = Field(ge=0)

    @model_validator(mode="after")
    def _resolved_after_created(self):
        if self.resolved_ts <= self.created_ts:
            raise ValueError("resolved_ts must be after created_ts")
        return self


class Candle(BaseModel):
    t: int
    price_yes: float = Field(gt=0, lt=1)


class Exclusion(BaseModel):
    market_id: str
    reason: str
