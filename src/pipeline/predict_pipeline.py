import sys
import os

import pandas as pd

from src.exception import CustomException

from src.utils import load_object


class PredictPipeline:

    def __init__(self):
        pass

    def predict(self, features):

        try:

            model_path = os.path.join(
                "artifacts",
                "model.pkl"
            )

            preprocessor_path = os.path.join(
                "artifacts",
                "preprocessor.pkl"
            )

            print("Before Loading")

            model = load_object(file_path=model_path)

            preprocessor = load_object(file_path=preprocessor_path)

            print("After Loading")

            # Transform input data
            data_scaled = preprocessor.transform(features)

            # Prediction
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:

            raise CustomException(e, sys)


class CustomData:

    def __init__(

        self,

        CGPA: float,

        Internships: int,

        Projects: int,

        Workshops_Certifications: int,

        AptitudeTestScore: int,

        SoftSkillsRating: float,

        ExtracurricularActivities: int,

        PlacementTraining: int,

        SSC_Marks: int,

        HSC_Marks: int

    ):

        self.CGPA = CGPA

        self.Internships = Internships

        self.Projects = Projects

        self.Workshops_Certifications = Workshops_Certifications

        self.AptitudeTestScore = AptitudeTestScore

        self.SoftSkillsRating = SoftSkillsRating

        self.ExtracurricularActivities = ExtracurricularActivities

        self.PlacementTraining = PlacementTraining

        self.SSC_Marks = SSC_Marks

        self.HSC_Marks = HSC_Marks


    def get_data_as_data_frame(self):

        try:

            custom_data_input_dict = {

                "CGPA": [self.CGPA],

                "Internships": [self.Internships],

                "Projects": [self.Projects],

                "Workshops/Certifications": [
                    self.Workshops_Certifications
                ],

                "AptitudeTestScore": [
                    self.AptitudeTestScore
                ],

                "SoftSkillsRating": [
                    self.SoftSkillsRating
                ],

                "ExtracurricularActivities": [
                    self.ExtracurricularActivities
                ],

                "PlacementTraining": [
                    self.PlacementTraining
                ],

                "SSC_Marks": [
                    self.SSC_Marks
                ],

                "HSC_Marks": [
                    self.HSC_Marks
                ]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:

            raise CustomException(e, sys)