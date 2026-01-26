from pydantic import BaseModel, Field
from typing import Optional
from datetime import date
from enum import Enum


class Gender(Enum):
    """Enum for gender values in passport."""
    MALE = "M"
    FEMALE = "F"


class MRZ(BaseModel):
    """Machine Readable Zone containing both lines."""
    line_1: str = Field(..., description="First line of the MRZ")
    line_2: str = Field(..., description="Second line of the MRZ")


class Passeport(BaseModel):
    """Pydantic model for extracted French passport information."""
    type: str = Field(..., description="Document type, e.g., 'P' for passport")
    country_code: str = Field(..., description="Issuing country code, e.g., 'FRA' for France")
    passport_number: str = Field(..., description="Passport number")
    surname: str = Field(..., description="Holder's surname")
    given_names: str = Field(..., description="Holder's given names")
    nationality: str = Field(..., description="Holder's nationality")
    date_of_birth: date = Field(..., description="Date of birth")
    sex: Gender = Field(..., description="Gender of the holder")
    place_of_birth: str = Field(..., description="Place of birth")
    date_of_issue: date = Field(..., description="Date of issue")
    date_of_expiry: date = Field(..., description="Date of expiry")
    authority: str = Field(..., description="Issuing authority")
    mrz: Optional[MRZ] = Field(None, description="Machine Readable Zone data")