import os
import json
import re
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    SqliteDatabase, Model, IntegrityError,
    IntegerField, FloatField, TextField
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect

########################################
# Input validation functions
def check_request(request):
    """
        Validates that our request is well formatted
        Returns:
        - assertion value: True if request is ok, False otherwise
        - error message: empty if request is ok, False otherwise
    """
    if "id" not in request:
        error = "Field `id` missing from request: {}".format(request)
        return False, error
    if "observation" not in request:
        error = "Field `observation` missing from request: {}".format(request)
        return False, error
    return True, ""
def check_valid_column(observation):
    """
        Validates that our observation only has valid columns
        Returns:
        - assertion value: True if all provided columns are valid, False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    """
    valid_columns = {
        "observation_id",
        "Health Service Area",
        "Hospital County",
        "Operating Certificate Number",
        "Facility Id",
        "Facility Name",
        "Age Group",
        "Zip Code - 3 digits",
        "Gender",
        "Race",
        "Ethnicity",
        "Type of Admission",
        "CCS Diagnosis Code",
        "CCS Diagnosis Description",
        "CCS Procedure Code",
        "CCS Procedure Description",
        "APR DRG Code",
        "APR DRG Description",
        "APR MDC Code",
        "APR MDC Description",
        "APR Severity of Illness Code",
        "APR Severity of Illness Description",
        "APR Risk of Mortality",
        "APR Medical Surgical Description",
        "Payment Typology 1",
        "Payment Typology 2",
        "Payment Typology 3",
        "Attending Provider License Number",
        "Operating Provider License Number",
        "Other Provider License Number",
        "Birth Weight",
        "Abortion Edit Indicator",
        "Emergency Department Indicator",
    }
    keys = set(observation.keys())
    if len(valid_columns - keys) > 0:
        missing = valid_columns - keys
        error = "Missing columns: {}".format(missing)
        return False, error
    if len(keys - valid_columns) > 0:
        extra = keys - valid_columns
        error = "Unrecognized columns provided: {}".format(extra)
        return False, error
    return True, ""
def check_categorical_values(observation):
    """
        Validates that all categorical fields are in the observation and values are valid
        Returns:
        - assertion value: True if all provided categorical columns contain valid values,
                           False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    """
    valid_category_map = {
        "Health Service Area": [
            'Western NY', 'Finger Lakes', 'Southern Tier', 'Central NY',
            'Capital/Adiron', 'Hudson Valley', 'New York City', 'Long Island'],
        #"Hospital County": [],
        #"Facility Name": [],
        "Age Group": ['0 to 17', '18 to 29', '30 to 49', '50 to 69', '70 or Older'],
        "Gender": ['F', 'M'],
        "Race": ['White', 'Other Race', 'Black/African American', 'Multi-racial'],
        "Ethnicity": ['Not Span/Hispanic', 'Unknown', 'Spanish/Hispanic', 'Multi-ethnic'],
        "Type of Admission": ['Emergency', 'Elective', 'Urgent', 'Newborn', 'Not Available','Trauma'],
        #"CCS Diagnosis Description": [],
        #"CCS Procedure Description": [],
        #"APR DRG Description": [],
        #"APR MDC Description": [],
        "APR Severity of Illness Description": ['Moderate', 'Minor', 'Major', 'Extreme'],
        "APR Severity of Illness Code": [1, 2, 3, 4],
        "APR Risk of Mortality": ['Minor', 'Moderate', 'Major', 'Extreme'],
        "APR Medical Surgical Description": ['Medical', 'Surgical', 'Not Applicable'],
        "Payment Typology 1": [
            'Medicaid', 'Medicare', 'Blue Cross/Blue Shield', 'Private Health Insurance',
            'Self-Pay', 'Managed Care, Unspecified', 'Federal/State/Local/VA',
            'Miscellaneous/Other', 'Department of Corrections', 'Unknown'],
        #"Payment Typology 2": [
        #    'Self-Pay', 'Medicaid', nan, 'Medicare', 'Private Health Insurance', 'Unknown',
        #    'Blue Cross/Blue Shield', 'Federal/State/Local/VA', 'Miscellaneous/Other',
        #    'Managed Care, Unspecified', 'Department of Corrections'],
        #"Payment Typology 3": [],
        "Abortion Edit Indicator": ['Y', 'N'],
        "Emergency Department Indicator": ['Y', 'N'],
    }
    for key, valid_categories in valid_category_map.items():
        if key in observation:
            value = observation[key]
            if value not in valid_categories:
                error = "Invalid value provided for {}: {}. Allowed values are: {}".format(
                    key, value, ",".join(["'{}'".format(v) for v in valid_categories]))
                return False, error
        else:
            error = "Categorical field {} missing"
            return False, error
    return True, ""
def check_facility_id(observation):
    """
        Validates that observation contains valid facility id value
        Returns:
        - assertion value: True if facility id is valid, False otherwise
        - error message: empty if facility id is valid, False otherwise
    """
    facility_id = observation.get("Facility Id")
    if not facility_id:
        error = "Field `Facility Id` missing"
        return False, error
    if not isinstance(facility_id, int):
        error = "Field `Facility Id` is not an integer"
        return False, error
    if facility_id < 0:
        error = "Field `Facility Id` is negative"
        return False, error
    return True, ""
def check_zip_code(observation):
        # Validates that observation has 3 digits or 'OOS'
    zip_code = observation.get("Zip Code - 3 digits")
    if not zip_code:
        error = "Field `Zip Code - 3 digits` missing"
        return False, error
    # Check if zip_code is 'OOS'
    if zip_code == 'OOS':
        return True, ""
    # Check if zip_code is a 3-digit integer or a string representing a 3-digit number
    if (isinstance(zip_code, int) and 100 <= zip_code <= 200) or \
       (isinstance(zip_code, str) and zip_code.isdigit() and len(zip_code) == 3):
        return True, ""
    error = "Field `Zip Code - 3 digits` must be a 3-digit integer or 'OOS'"
    return False, error
def check_ccs_diagnosis_code(observation):
        # Validates that observation contains valid CCS diagnosis code value
    ccs_diagnosis_code = observation.get("CCS Diagnosis Code")
    if not ccs_diagnosis_code:
        error = "Field `CCS Diagnosis Code` missing"
        return False, error
    if not isinstance(ccs_diagnosis_code, int):
        error = "Field `CCS Diagnosis Code` is not an integer"
        return False, error
    if ccs_diagnosis_code < 1 or ccs_diagnosis_code > 917:
        error = "Field `CCS Diagnosis Code` is invalid"
        return False, error
    return True, ""
def check_ccs_procedure_code(observation):
        # Validates that observation contains valid CCS procedure code value
    ccs_procedure_code = observation.get("CCS Procedure Code")
    if not ccs_procedure_code:
        error = "Field `CCS Procedure Code` missing"
        return False, error
    if not isinstance(ccs_procedure_code, int):
        error = "Field `CCS Procedure Code` is not an integer"
        return False, error
    if ccs_procedure_code < 0 or ccs_procedure_code > 999:
        error = "Field `CCS Procedure Code` is invalid"
        return False, error
    return True, ""
def check_apr_drg_code(observation):
        # Validates that observation contains valid APR DRG code value
    apr_drg_code = observation.get("APR DRG Code")
    if not apr_drg_code:
        error = "Field `APR DRG Code` missing"
        return False, error
    if not isinstance(apr_drg_code, int):
        error = "Field `APR DRG Code` is not an integer"
        return False, error
    if apr_drg_code < 1 or apr_drg_code > 956:
        error = "Field `APR DRG Code` is invalid"
        return False, error
    return True, ""
def check_apr_mdc_code(observation):
        # Validates that observation contains valid APR MDC code value
    apr_mdc_code = observation.get("APR MDC Code")
    if not apr_mdc_code:
        error = "Field `APR MDC Code` missing"
        return False, error
    if not isinstance(apr_mdc_code, int):
        error = "Field `APR MDC Code` is not an integer"
        return False, error
    if apr_mdc_code < 0 or apr_mdc_code > 25:
        error = "Field `APR MDC Code` is invalid"
        return False, error
    return True, ""
def check_birth_weight(observation):
        # Validates that observation contains valid birth weight value
    birth_weight = observation.get("Birth Weight")
    if not birth_weight:
        error = "Field `Birth Weight` missing"
        return False, error
    if not isinstance(birth_weight, int):
        error = "Field `Birth Weight` is not an integer"
        return False, error
    if birth_weight < 0:
        error = "Field `Birth Weight` is negative"
        return False, error
    return True, ""
# NOT REQUIRED IN OUR MODEL ############################
def check_attending_provider_number(observation):
        # Validates that observation contains valid Attending Provider License Number
    attending_provider_number = observation.get("Attending Provider License Number")
    if not attending_provider_number:
        error = "Field `Attending Provider License Number` missing"
        return False, error
    if isinstance(attending_provider_number, int):
        return True, ""
    if isinstance(attending_provider_number, float) and attending_provider_number.is_integer():
        return True, ""
    if attending_provider_number < 0:
        error = "Field `Attending Provider License Number` is negative"
        return False, error
    error = "Field `Attending Provider License Number` must be an integer or a float with no decimal places"
    return False, error