import os
import json
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
# Input validation functions
def respond_error(observation_id, error_msg):
    print(error_msg)
    response = {'observation_id': observation_id, 'error': error_msg}
    return jsonify(response)

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
                         "Patient Disposition",
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
        return respond_error(None, "Error: observation has no ID.")
    
    for feature in request:
        if feature not in valid_columns:
            return respond_error(observation_id, f"Error: {feature} is not a valid feature.")
    
    for feature in mandatory_columns:
        if feature not in request:
            return respond_error(observation_id, f"Error: {feature} is missing.")

    return jsonify({'observation_id': observation_id, 'observation': request})

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
            error_msg = "Error: Invalid value provided for {}: {}. Allowed values are: {}".format(
                category, value, ",".join(["'{}'".format(v) for v in valid_category_map[category]]))
            return respond_error(response['observation_id'], error_msg)
    zip_code_regex = r"\b(\d{3}|OOS)\b"
    if not re.match(zip_code_regex, response['observation']["Zip Code - 3 digits"]):
        return respond_error(response['observation_id'], "Error: Zip Code does not obey to standard format.")
    return response

#def check_numerical_values(response):


def check_input(request):
    response = check_valid_columns(request)
    if 'error' in response:
        return response
    response = check_categorical_values(response)
    if 'error' in response:
        return response

# 3.2 Predict
@app.route('/predict', methods=['POST'])
def predict():
    # get input
    payload = request.get_json()

    # validate input
    response = check_input(payload)
    if 'error' in response:
        return response

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

# 3.2 Update
@app.route('/update', methods=['POST']) # shouldn't it be PATCH or PUT?

def update():
    payload = request.get_json()
    try:
        observation_id = payload['observation_id']
    except KeyError:
        error_msg = "Error: observation has no ID."
        print(error_msg)
        return jsonify({'error': error_msg})

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
