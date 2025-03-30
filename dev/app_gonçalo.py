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
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    observation_id = TextField(unique=True)
    #observation = TextField() # THINK: what to put here
    prediction = IntegerField() # THINK: if a float is received, should convert it?
    true_value = IntegerField(null=True)

    class Meta:
        database = DB

# DB.create_tables([Prediction], safe=True)

# 3. APP
app = Flask(__name__)

with open('columns.json') as fh:
    columns = json.load(fh)

pipeline = joblib.load('pipeline.pickle')

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)

# 3.1 Predict
@app.route('/predict', methods=['POST'])

def predict():
    # get input
    payload = request.get_json()
    try:
        observation_id = payload['observation_id']
    except KeyError:
        error_msg = "Error: observation has no ID."
        print(error_msg)
        return jsonify({'error': error_msg})

    observation = payload
    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    
    result = pipeline.predict(obs)[0]
    response = {'proba': int(result)}
    # p = Prediction(
    #     observation_id=observation_id,
    #     proba=result,
    #     observation=request.data
    # )

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
    return jsonify(response)

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
