from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)

    @field_validator("text")
    @classmethod
    def not_just_spaces(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text cannot be empty or just spaces")
        return value.strip()


class PredictResponse(BaseModel):
    category: str
    priority: str
    confidence: float
