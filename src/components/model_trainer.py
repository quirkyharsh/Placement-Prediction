import os
import sys

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

from catboost import CatBoostClassifier

from src.exception import CustomException

from src.logger import logging

from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:

    trained_model_file_path = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:

    def __init__(self):

        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(
        self,
        train_array,
        test_array
    ):

        try:

            logging.info(
                "Splitting training and testing input data"
            )

            # ---------------- TRAIN TEST SPLIT ----------------

            X_train = train_array[:, :-1]

            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]

            y_test = test_array[:, -1]

            logging.info("Data split completed")


            # ---------------- MODELS ----------------

            models = {

                "Logistic Regression": LogisticRegression(),

                "KNeighbors Classifier": KNeighborsClassifier(),

                "Decision Tree": DecisionTreeClassifier(
                    random_state=42
                ),

                "Random Forest": RandomForestClassifier(
                    random_state=42
                ),

                "Gradient Boosting": GradientBoostingClassifier(
                    random_state=42
                ),

                "AdaBoost Classifier": AdaBoostClassifier(
                    random_state=42
                ),

                "SVC": SVC(
                    probability=True
                ),

                "XGBoost Classifier": XGBClassifier(
                    eval_metric='logloss',
                    random_state=42
                ),

                "CatBoost Classifier": CatBoostClassifier(
                    verbose=False,
                    random_state=42
                )
            }


            # ---------------- HYPERPARAMETERS ----------------

            params = {

                "Logistic Regression": {},

                "KNeighbors Classifier": {

                    'n_neighbors': [3, 5, 7, 9]
                },

                "Decision Tree": {

                    'criterion': [
                        'gini',
                        'entropy'
                    ]
                },

                "Random Forest": {

                    'n_estimators': [
                        50,
                        100,
                        200
                    ]
                },

                "Gradient Boosting": {

                    'learning_rate': [
                        0.1,
                        0.01
                    ],

                    'n_estimators': [
                        50,
                        100
                    ]
                },

                "AdaBoost Classifier": {

                    'learning_rate': [
                        0.1,
                        0.01
                    ],

                    'n_estimators': [
                        50,
                        100
                    ]
                },

                "SVC": {

                    'C': [
                        1,
                        10
                    ],

                    'kernel': [
                        'linear',
                        'rbf'
                    ]
                },

                "XGBoost Classifier": {

                    'learning_rate': [
                        0.1,
                        0.01
                    ],

                    'n_estimators': [
                        50,
                        100
                    ]
                },

                "CatBoost Classifier": {

                    'depth': [
                        6,
                        8
                    ],

                    'learning_rate': [
                        0.01,
                        0.05
                    ],

                    'iterations': [
                        50,
                        100
                    ]
                }
            }


            # ---------------- MODEL EVALUATION ----------------

            model_report = evaluate_models(

                X_train=X_train,

                y_train=y_train,

                X_test=X_test,

                y_test=y_test,

                models=models,

                param=params
            )

            logging.info(
                f"Model Report : {model_report}"
            )


            # ---------------- BEST MODEL ----------------

            best_model_score = max(
                sorted(model_report.values())
            )

            best_model_name = list(
                model_report.keys()
            )[
                list(model_report.values()).index(
                    best_model_score
                )
            ]

            best_model = models[best_model_name]

            logging.info(
                f"Best Model Found : {best_model_name}"
            )

            logging.info(
                f"Best Model Accuracy : {best_model_score}"
            )


            # ---------------- VALIDATION ----------------

            if best_model_score < 0.6:

                raise CustomException(
                    "No best model found"
                )


            # ---------------- TRAIN BEST MODEL ----------------

            best_model.fit(
                X_train,
                y_train
            )


            # ---------------- SAVE MODEL ----------------

            save_object(

                file_path=self.model_trainer_config.trained_model_file_path,

                obj=best_model
            )

            logging.info(
                "Best model saved successfully"
            )


            # ---------------- PREDICTION ----------------

            predicted = best_model.predict(
                X_test
            )

            accuracy = accuracy_score(
                y_test,
                predicted
            )

            logging.info(
                f"Final Accuracy Score : {accuracy}"
            )

            return accuracy


        except Exception as e:

            raise CustomException(e, sys)