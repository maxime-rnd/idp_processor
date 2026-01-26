import pytest
from datetime import date
from idp_extractor.models import FrenchPassportInfo


def test_french_passport_info():
    data = {
        "type": "P",
        "country_code": "FRA",
        "passport_number": "123456789",
        "surname": "Doe",
        "given_names": "John",
        "nationality": "French",
        "date_of_birth": date(1990, 1, 1),
        "sex": "M",
        "place_of_birth": "Paris",
        "date_of_issue": date(2020, 1, 1),
        "date_of_expiry": date(2030, 1, 1),
        "authority": "OFII"
    }
    passport = FrenchPassportInfo(**data)
    assert passport.passport_number == "123456789"
    assert passport.date_of_birth == date(1990, 1, 1)