from ibm_watson_machine_learning import APIClient
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model


class PredictCVD:
    def __init__(self):
        self.__image_to_predict = None
        self.__CVDs = [
            'left_bundle_branch_block.html',
            'normal.html',
            'premature_atrial_contraction.html',
            'premature_ventricular_contractions.html',
            'right_bundle_branch_block.html',
            'ventricular_fibrillation.html'
        ]
        wml_credential = {
            'url': 'https://eu-de.ml.cloud.ibm.com',
            'apikey': 'Ndj3UnNat-ZW_rOzehbxfzaMRt5DfuLeISJfNW_vuebX'
        }
        self.__deployment_id = "b48cd709-a43a-45af-8f80-8535affa8849"
        try:
            self.__client = APIClient(wml_credential)
            self.__run_locally = False
            self.__local_model = load_model("model_30_oct_22.h5")
            self.__client.set.default_space("4ecc4a52-91f7-4d31-a885-6851624e76ec")
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
            except Exception as e:
                self.__run_locally = True
                self.__local_model = load_model("model_30_oct_22.h5")
                return self.__predict_cvd_locally()

    def __predict_cvd_locally(self):
        print("Prediction starts in Local Machine")
        prediction = self.__local_model.predict(self.__image_to_predict)
        return self.__CVDs[list(prediction[0]).index(1)]

    def __predict_cvd_cloud(self):
        print("Prediction starts in Cloud")
        scoring_payload = {
            self.__client.deployments.ScoringMetaNames.INPUT_DATA: [{
                "values": self.__image_to_predict
            }]
        }
        class_value = self.__client.deployments.score(deployment_id=self.__deployment_id,
                                                      meta_props=scoring_payload)['values'][1]
        return self.__CVDs[class_value]

