import ludwig
import pandas as pd
from ludwig.api import LudwigModel


#Load CSV
train = pd.read_csv("pneumoniaTrain.csv")
test = pd.read_csv("pneumoniaTest.csv")
validation = pd.read_csv("pneumoniaValidate.csv")

# train a model

model_definition = {
    "input_features":
        [
            {"name": "image_path", "type": "image", "encoder":"stacked_cnn","preprocessing":
             {"resize_method": "crop_or_pad", "width":128, "height":128, "num_channels":1}
            }
        ],
    "output_features":
        [
            {"name": "label", "type": "category"}
        ],
    "training":

            {"batch_size": 8, "epochs": 7}
}

model = LudwigModel(model_definition)

train_stats = model.train(data_train_df=train,data_test_df=validation)

#load a model
# model_path = "results/api_experiment_run_0/model"
# model = LudwigModel.load(model_path)

#predictions
predictions = model.predict(data_df=test)
save_path = '/Users/kalyan/PythonPrograms/ludwig/predictions.csv'
predictions.to_csv(save_path, index=None)

#
model.close()
