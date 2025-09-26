from pydantic import BaseModel, Field
from typing import Literal, Optional

class StrokeInput(BaseModel):
    gender: Literal["Male","Female","Other"]
    age: float = Field(..., ge=0)
    hypertension: Literal[0,1]
    heart_disease: Literal[0,1]
    ever_married: Literal["Yes","No"]
    work_type: Literal["children","Govt_job","Never_worked","Private","Self-employed"]
    Residence_type: Literal["Urban","Rural"]
    avg_glucose_level: float = Field(..., ge=0)
    bmi: Optional[float] = Field(None, ge=0)
    smoking_status: Literal["formerly smoked","never smoked","smokes","Unknown"]
