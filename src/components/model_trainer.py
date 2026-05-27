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

            logging.info("Splitting training and test input data")

            X_train, y_train, X_test, y_test = (

                train_array[:, :-1],

                train_array[:, -1],

                test_array[:, :-1],

                test_array[:, -1]
            )

            models = {

                "Logistic Regression": LogisticRegression(),

                "KNeighbors Classifier": KNeighborsClassifier(),

                "Decision Tree": DecisionTreeClassifier(),

                "Random Forest": RandomForestClassifier(),

                "Gradient Boosting": GradientBoostingClassifier(),

                "AdaBoost Classifier": AdaBoostClassifier(),

                "SVC": SVC(),

                "XGBoost Classifier": XGBClassifier(
                    use_label_encoder=False,
                    eval_metric='logloss'
                ),

                "CatBoost Classifier": CatBoostClassifier(
                    verbose=False
                )
            }

            params = {

                "Logistic Regression": {},

                "KNeighbors Classifier": {

                    'n_neighbors': [3,5,7,9]
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
                        .1,
                        .01
                    ],

                    'n_estimators': [
                        50,
                        100
                    ]
                },

                "AdaBoost Classifier": {

                    'learning_rate': [
                        .1,
                        .01
                    ],

                    'n_estimators': [
                        50,
                        100
                    ]
                },

                "SVC": {

                    'C': [1,10],

                    'kernel': [
                        'linear',
                        'rbf'
                    ]
                },

                "XGBoost Classifier": {

                    'learning_rate': [
                        .1,
                        .01
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

            model_report = evaluate_models(

                X_train=X_train,

                y_train=y_train,

                X_test=X_test,

                y_test=y_test,

                models=models,

                param=params
            )

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
                f"Best Model Found: {best_model_name}"
            )

            # Train best model again

            best_model.fit(X_train, y_train)

            if best_model_score < 0.6:

                raise CustomException(
                    "No best model found"
                )

            save_object(

                file_path=self.model_trainer_config.trained_model_file_path,

                obj=best_model
            )

            predicted = best_model.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                predicted
            )

            return accuracy

        except Exception as e:

            raise CustomException(e, sys)