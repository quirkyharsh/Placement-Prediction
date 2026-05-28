from flask import Flask, request, render_template

import pandas as pd
import numpy as np

from src.pipeline.predict_pipeline import (
    CustomData,
    PredictPipeline
)

application = Flask(__name__)

app = application


# ---------------- HOME ROUTE ----------------

@app.route('/')

def index():

    return render_template('home.html')


# ---------------- PREDICTION ROUTE ----------------

@app.route('/predictdata', methods=['GET', 'POST'])

def predict_datapoint():

    if request.method == 'GET':

        return render_template('home.html')

    else:

        try:

            data = CustomData(

                CGPA=float(request.form.get('CGPA')),

                Internships=int(request.form.get('Internships')),

                Projects=int(request.form.get('Projects')),

                Workshops_Certifications=int(
                    request.form.get('Workshops_Certifications')
                ),

                AptitudeTestScore=int(
                    request.form.get('AptitudeTestScore')
                ),

                SoftSkillsRating=float(
                    request.form.get('SoftSkillsRating')
                ),

                ExtracurricularActivities=int(
                    request.form.get('ExtracurricularActivities')
                ),

                PlacementTraining=int(
                    request.form.get('PlacementTraining')
                ),

                SSC_Marks=int(
                    request.form.get('SSC_Marks')
                ),

                HSC_Marks=int(
                    request.form.get('HSC_Marks')
                )
            )

            pred_df = data.get_data_as_data_frame()

            print(pred_df)

            predict_pipeline = PredictPipeline()

            results = predict_pipeline.predict(pred_df)

            prediction = results[0]

            if prediction == 1:

                result_text = "Student Will Be Placed"

            else:

                result_text = "Student Will Not Be Placed"

            return render_template(

                'home.html',

                results=result_text
            )

        except Exception as e:

            return str(e)


if __name__ == "__main__":
    
    app.run(host="0.0.0.0", port=5000)