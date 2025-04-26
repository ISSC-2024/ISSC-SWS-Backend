from pydantic import BaseModel


class Algorithm2ResultBase(BaseModel):
    point_id: str
    area_code: str
    pred_risk: str
    weight: float


class Algorithm2ResultResponse(Algorithm2ResultBase):

    class Config:
        orm_mode = True
