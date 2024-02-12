import pandas as pd
import pickle
import joblib
import os

def load_and_predict_models(start_date, end_date):
    def load_and_predict_models_internal(models_directory, rf_model_path):

        all_files = os.listdir(models_directory)

        model_paths = [
            os.path.join(models_directory, filename) for filename in [
                "temperature_2m_max.pkl",
                "temperature_2m_min.pkl",
                "temperature_2m_mean.pkl",
                "apparent_temperature_max.pkl",
                "apparent_temperature_min.pkl",
                "apparent_temperature_mean.pkl",
                "shortwave_radiation_sum.pkl",
                "precipitation_sum.pkl",
                "precipitation_hours.pkl",
                "windspeed_10m_max.pkl",
                "windgusts_10m_max.pkl",
                "winddirection_10m_dominant.pkl",
                "et0_fao_evapotranspiration.pkl",
            ] if filename in all_files
        ]


        prediction_dates = pd.date_range(start=start_date, end=end_date)

        all_predictions = pd.DataFrame({'ds': prediction_dates})

        for model_path in model_paths:
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)

            forecast = loaded_model.predict(all_predictions)


            model_name = os.path.splitext(os.path.basename(model_path))[0] 
            all_predictions[model_name] = forecast['yhat'].astype('float64')

        loaded_rf = joblib.load(rf_model_path)


        prediction_df = all_predictions.drop(["ds"], axis=1)
        y_pred = loaded_rf.predict(prediction_df)

        return y_pred


    model_directories = [
        ("Hatton", "E:/myProjects/weather/Hatton/"),
        ("Kandy", "E:/myProjects/weather/kandy/"),
        ("Jaffna", "E:/myProjects/weather/Jaffna/"),
        ("Badulla", "E:/myProjects/weather/Badulla/")
    ]

    rf_model_path = "E:/myProjects/weather/final/rf.pkl"

    weather_predictions = {}

    for folder_name, models_directory in model_directories:
        y_pred = load_and_predict_models_internal(models_directory, rf_model_path)
        weather_predictions[folder_name] = y_pred

    return weather_predictions


start_date = '2023-10-01'
end_date = '2024-10-01'

predictions = load_and_predict_models(start_date, end_date)
print(predictions)