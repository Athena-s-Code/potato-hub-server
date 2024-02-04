import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

__locations = None
__varietys = None
__conditions = None
__model = None
__mle = None

class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns  # array of column names to encode
        self.label_encoders = {}

    def fit(self, X, y=None):
        if self.columns is not None:
            for col in self.columns:
                self.label_encoders[col] = LabelEncoder().fit(X[col])
        else:
            for colname, col in X.iteritems():
                self.label_encoders[colname] = LabelEncoder().fit(col)
        return self  # not relevant here

    def transform(self, X):
        """
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        """
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = self.label_encoders[col].transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = self.label_encoders[colname].transform(col)
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def transform_single(self, record):
        """Transforms a single record using the stored LabelEncoders."""
        transformed_record = record.copy()
        if self.columns is not None:
            for col in self.columns:
                transformed_record[col] = self.label_encoders[col].transform([transformed_record[col]])[0]
        else:
            for colname, col_value in record.iteritems():
                transformed_record[colname] = self.label_encoders[colname].transform([col_value])[0]
        return transformed_record

__mle = MultiColumnLabelEncoder()

def get_predicted_price( disaster, condition, variety, origin, organic, location, temp=27.4956, rainfall=148.1091):
    record = {
    "Temperature (°C)": temp,
    "Disaster Happened (Last 3 Months)": disaster,
    "Vegetable Condition": condition,
    "Variety": variety,
    "Rainfall (mm)": rainfall,
    "Origin": origin,
    "Is Organic": organic,
    "Location": location
    }
    en_record = __mle.transform_single(record)

    ar = np.ones(8)
    ar[0] = en_record['Temperature (°C)']
    ar[1] = en_record['Disaster Happened (Last 3 Months)']
    ar[2] = en_record['Vegetable Condition']
    ar[3] = en_record['Variety']
    ar[4] = en_record['Rainfall (mm)']
    ar[5] = en_record['Origin']
    ar[6] = en_record['Is Organic']
    ar[7] = en_record['Location']

    return __model.predict([ar])[0]
def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __locations
    global __varietys
    global __conditions
    global __mle

    # Load the label encoder
    with open("./models/Price Prediction/label_encoder.pickle", "rb") as f:
        __mle = pickle.load(f)
        __locations = list(__mle.label_encoders['Location'].classes_)
        __varietys = list(__mle.label_encoders['Variety'].classes_)
        __conditions = list(__mle.label_encoders['Vegetable Condition'].classes_)


    global __model
    if __model is None:
        with open("./models/Price Prediction/finalize_model.pickle", "rb") as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done")


def get_location_names():
    return __locations

def get_vegetable_conditions():
    return __conditions

def get_potato_varietys():
    return __varietys

if __name__ == "__main__":
    load_saved_artifacts()
    print(get_location_names())
    print(get_potato_varietys())
    print(get_vegetable_conditions())

    print(get_predicted_price(temp=35, disaster='Yes', condition='Dry', variety='BABY POTATOES', origin='Imported', organic='No', location=' Seeduwa,  Gampaha'))