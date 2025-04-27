from pydantic import BaseModel


class Algorithm2ConfigBase(BaseModel):
    tree_count: int
    max_depth: int
    sensitivity: float


class Algorithm2ConfigCreate(Algorithm2ConfigBase):
    pass


class Algorithm2ConfigResponse(Algorithm2ConfigBase):
    pass
