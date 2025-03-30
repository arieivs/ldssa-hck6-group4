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

# 1. DESERIALIZE
# with open('columns.json') as c_fd:
#     columns = json.load(c_fd)

# with open('dtypes.pickle', 'rb') as t_fd:
#     dtypes = pickle.load(t_fd)

# pipeline = joblib.load('pipeline.pickle')

# 2. DATABASE
# DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

# class Prediction(Model):
#     observation_id = TextField(unique=True)
#     #observation = TextField() # THINK: what to put here
#     prediction = IntegerField() # THINK: if a float is received, should convert it?
#     true_value = IntegerField(null=True)

#     class Meta:
#         database = DB

# DB.create_tables([Prediction], safe=True)

# 3. APP
app = Flask(__name__)

# 3.1 Input validation auxiliary functions
# Note: we are checking well the features which are important for us
# Room for improvement: checking well all features
def respond_error(observation_id, error_msg):
    print(error_msg)
    response = {'observation_id': observation_id, 'error': error_msg}
    return response

def check_valid_columns(request):
    """
        Validates that our observation only has valid columns
        Returns:
        - assertion value: True if all provided columns are valid, False otherwise
        - error message: empty if all provided columns are valid, False otherwise
    """
    mandatory_columns = ["Health Service Area",
                         "Hospital County",
                         "Facility Id",
                         "Age Group",
                         "Zip Code - 3 digits",
                         "Gender",
                         "Race",
                         "Ethnicity",
                         "Type of Admission",
                         "CCS Diagnosis Code",
                         "CCS Procedure Code",
                         "APR DRG Code",
                         "APR MDC Code",
                         "APR Severity of Illness Code",
                         "APR Severity of Illness Description", # not in the model
                         "APR Risk of Mortality",
                         "APR Medical Surgical Description",
                         "Payment Typology 1",
                         "Emergency Department Indicator",
                         "Abortion Edit Indicator" # not in the model
                         ]
    valid_columns = ["Health Service Area",
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
                     "Emergency Department Indicator"]
    
    try:
        observation_id = request.pop('observation_id')
    except KeyError:
        return respond_error(None, "Observation has no ID.")
    
    for feature in request:
        if feature not in valid_columns:
            return respond_error(observation_id, f"{feature} is not a valid feature.")
    for feature in mandatory_columns:
        if feature not in request:
            return respond_error(observation_id, f"{feature} is missing.")

    return {'observation_id': observation_id, 'observation': request}

def check_categorical_values(response):
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
        "Age Group": ['0 to 17', '18 to 29', '30 to 49', '50 to 69', '70 or Older'],
        "Gender": ['F', 'M'],
        "Race": ['White', 'Other Race', 'Black/African American', 'Multi-racial'],
        "Ethnicity": ['Not Span/Hispanic', 'Unknown', 'Spanish/Hispanic', 'Multi-ethnic'],
        "Type of Admission": ['Emergency', 'Elective', 'Urgent', 'Newborn', 'Not Available','Trauma'],
        "APR Severity of Illness Description": ['Moderate', 'Minor', 'Major', 'Extreme'],
        "APR Severity of Illness Code": ['1', '2', '3', '4'],
        "APR Risk of Mortality": ['Minor', 'Moderate', 'Major', 'Extreme'],
        "APR Medical Surgical Description": ['Medical', 'Surgical', 'Not Applicable'],
        "Payment Typology 1": [
            'Medicaid', 'Medicare', 'Blue Cross/Blue Shield', 'Private Health Insurance',
            'Self-Pay', 'Managed Care, Unspecified', 'Federal/State/Local/VA',
            'Miscellaneous/Other', 'Department of Corrections', 'Unknown'],
        "Abortion Edit Indicator": ['Y', 'N'],
        "Emergency Department Indicator": ['Y', 'N'],
    }
    for category in valid_category_map:
        value = response['observation'][category]
        if value not in valid_category_map[category]:
            error_msg = "Invalid value provided for {}: {}. Allowed values are: {}".format(
                category, value, ",".join(["'{}'".format(v) for v in valid_category_map[category]]))
            return respond_error(response['observation_id'], error_msg)
    zip_code_regex = r"\b(\d{3}|OOS)\b"
    if not re.match(zip_code_regex, response['observation']["Zip Code - 3 digits"]):
        return respond_error(response['observation_id'], "Zip Code does not obey to standard format.")
    return response

def check_numerical_values(response):
    float_categories = ["Facility Id"]
    for category in float_categories:
        try:
            response['observation'][category] = float(response['observation'][category])
        except ValueError:
            respond_error(response['observation_id'], f"{category} should be a number.")
        if response['observation'][category] < 0:
            respond_error(response['observation_id'], f"{category} cannot be negative.")

    int_categories = ["CCS Diagnosis Code", "CCS Procedure Code", "APR DRG Code", "APR MDC Code"]
    for category in int_categories:
        try:
            response['observation'][category] = int(response['observation'][category])
        except ValueError:
            respond_error(response['observation_id'], f"{category} should be an integer.")
    ccs_codes = ["CCS Diagnosis Code", "CCS Procedure Code", "APR DRG Code"]
    for code in ccs_codes:
        if response['observation'][code] < 0 or response['observation'][code] > 999:
            respond_error(response['observation_id'], f"Invalid {code}.")
    if response['observation']["APR MDC Code"] < 0 or response['observation']["APR MDC Code"] > 25:
        respond_error(response['observation_id'], "Invalid APR MDC Code.")
    # Birth Weight is not mandatory, there's a lot of missing values as zero
    if "Birth Weight" in response['observation']:
        try:
            response['observation']["Birth Weight"] = int(response['observation']["Birth Weight"])
        except ValueError:
            respond_error(response['observation_id'], "Birth Weight should be an integer.")
        if response['observation']["Birth Weight"] < 0
            respond_error(response['observation_id'], "Birth Weight cannot be negative.")
    else:
        response['observation']["Birth Weight"] = 0
    return response

def check_input(request):
    response = check_valid_columns(request)
    if 'error' in response:
        return response
    response = check_categorical_values(response)
    if 'error' in response:
        return response
    response = check_numerical_values(response)
    if 'error' in response:
        return response
    return response

# 3.2 Predict Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # get input
    payload = request.get_json()

    # validate input
    response = check_input(payload)
    if 'error' in response:
        return jsonify(response)
    observation_id = response['observation_id']

    # predict
    # try:
    #     obs_df = pd.DataFrame([observation], columns=columns).astype(dtypes)
    # except ValueError:
    #     error_msg = f"Observation is invalid!"
    #     response = {"error": error_msg}
    #     print(error_msg)
    #     return jsonify(response)
    # proba = pipeline.predict_proba(obs_df)[0, 1]
    # response = {'proba': proba}

    # store in database
    # prediction = Prediction(pred_id=input_id, observation=observation, proba=proba)
    # try:
    #     prediction.save()
    # this is not ideal - we are doing the work of getting the prediction for cases when it might not be necessary
    # except IntegrityError:
    #     error_msg = f"Observation ID: {input_id} already exists"
    #     response["error"] = error_msg
    #     print(error_msg)
    #     DB.rollback()
    print(f"Received prediction for observation {observation_id}!")
    return jsonify({'observation_id': observation_id})

# 3.3 Validate updates input
def check_update(request):
    try:
        observation_id = payload['observation_id']
    except KeyError:
        return respond_error(None, "Observation has no ID.")
    try:
        true_value = payload['true_value']
    except KeyError:
        return respond_error(observation_id, "Observation has no true value.")
    for key in request:
        if key not in ['observation_id', 'true_value']:
            return respond_error(observation_id, f"{key} is not a correct parameter.")
    try:
        true_value = int(request['true_value'])
    except ValueError:
        return respond_error(observation_id, "True value should be an integer")
    return request

# 3.4 Update Endpoint
@app.route('/update', methods=['POST']) # shouldn't it be PATCH or PUT?
def update():
    # get input
    payload = request.get_json()

    # validate input
    response = check_update(payload)
    if 'error' in response:
        return jsonify(response)
    observation_id = response['observation_id']

    # true_class = payload['true_class']
    # try:
    #     prediction = Prediction.get(Prediction.pred_id == input_id)
    #     prediction.true_class = true_class
    #     prediction.save()
    #     return jsonify(model_to_dict(prediction))
    # except Prediction.DoesNotExist:
    #     error_msg = f"Observation ID: {input_id} does not exist"
    #     response = {"error": error_msg}
    #     print(error_msg)
    #     return jsonify(response)
    print(f"Received update for observation {observation_id}!")
    return jsonify({'observation_id': observation_id})


# 3.X Run the App
# Important to add host='0.0.0.0' (or a specific IP if the app is running on a fixed IP)
# else one cannot reach it from the outside
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
