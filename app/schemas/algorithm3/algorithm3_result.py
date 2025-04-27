from pydantic import BaseModel
from datetime import datetime


class Algorithm3ResultBase(BaseModel):
    timestamp: datetime
    point_id: str
    temperature: float
    pressure: float
    flow_rate: float
    level: float
    gas_type: str
    gas_concentration: float
    risk_level: str
    risk_level_name: str
    message: str | None


class Algorithm3ResultResponse(Algorithm3ResultBase):
    config_id: int
