from pydantic import BaseModel, field_validator
from typing import Optional


class Algorithm3ConfigBase(BaseModel):
    algorithm: str
    learning_rate: float
    max_depth: Optional[int] = None
    max_epochs: Optional[int] = None

    @field_validator('algorithm')
    @classmethod
    def algorithm_must_be_valid(cls, v):
        valid_algorithms = ['xgboost', 'lightGBM', 'TabNet']
        if v not in valid_algorithms:
            raise ValueError(f'算法必须是以下之一: {", ".join(valid_algorithms)}')
        return v

    @field_validator('max_depth')
    @classmethod
    def validate_max_depth(cls, v, info):
        if 'algorithm' not in info.data:
            return v

        algorithm = info.data['algorithm']

        # 处理xgboost和lightGBM的max_depth
        if algorithm in ['xgboost', 'lightGBM'] and v is None:
            raise ValueError(f"{algorithm}算法需要设置max_depth参数")

        return v

    @field_validator('max_epochs')
    @classmethod
    def validate_max_epochs(cls, v, info):
        if 'algorithm' not in info.data:
            return v

        algorithm = info.data['algorithm']

        # 处理TabNet的max_epochs
        if algorithm == 'TabNet' and v is None:
            raise ValueError(f"{algorithm}算法需要设置max_epochs参数")

        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "algorithm": "xgboost",
                    "learning_rate": 0.1,
                    "max_depth": 5,
                    "max_epochs": None
                }
            ]
        }
    }


class Algorithm3ConfigCreate(Algorithm3ConfigBase):
    pass


class Algorithm3ConfigResponse(Algorithm3ConfigBase):
    pass
