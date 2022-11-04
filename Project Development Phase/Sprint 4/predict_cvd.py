from ibm_watson_machine_learning import APIClient
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model


class PredictCVD:
    def __init__(self):
        self.__image_to_predict = None
        self.__CVDs = [
            'Left Bundle Branch Block',
            'Normal',
            'Premature Atrial Contraction',
            'Premature Ventricular Contractions',
            'Right Bundle Branch Block',
            'Ventricular Fibrillation'
        ]
        wml_credential = {
            'url': 'https://eu-de.ml.cloud.ibm.com',
            'apikey': 'kx-42wZuXhAyU21dDBA3RL9tcSwqk-GHPd3SUMEjwbeO'
        }
        self.__deployment_id = "2ef1e397-a035-4559-a99f-83a5231498b1"
        try:
            self.__client = APIClient(wml_credential)
            self.__run_locally = True
            self.__local_model = load_model("model_30_oct_22.h5")
            self.__client.set.default_space("b9b9359c-227c-461e-a909-667a401ef747")
        except Exception:
            print("Unable to communicate with IBM cloud")
            print("Preparing to run model locally")
            self.__run_locally = True
            self.__local_model = load_model("model_30_oct_22.h5")

    def predict(self, image_path) -> str:
        self.__image_to_predict = np.expand_dims(image.img_to_array(
            image.load_img(image_path, target_size=(64, 64), color_mode="grayscale")), axis=0)
        if self.__run_locally:
            return self.__predict_cvd_locally()
        else:
            try:
                return self.__predict_cvd_cloud()
            except Exception:
                self.__run_locally = True
                self.__local_model = load_model("model_30_oct_22.h5")
                return self.__predict_cvd_locally()

    def __predict_cvd_locally(self):
        prediction = self.__local_model.predict(self.__image_to_predict)
        return self.__CVDs[list(prediction[0]).index(1)]

    def __predict_cvd_cloud(self):
        scoring_payload = {
            self.__client.deployments.ScoringMetaNames.INPUT_DATA: [{
                "values": self.__image_to_predict
            }]
        }
        class_value = self.__client.deployments.score(deployment_id=self.__deployment_id,
                                                      meta_props=scoring_payload)['values'][1]
        return self.__CVDs[class_value]

