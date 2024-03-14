from flask import Flask, request, jsonify
import numpy
import util_price_pred
import util_weather_pred 


app = Flask(__name__)

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

@app.route("/get_location_names", methods=["GET"])
def get_location_names():
    response = jsonify({"locations": util_price_pred.get_location_names()})
    response.headers.add("Access-Control-Allow-Origin", "*")

    return response

@app.route("/get_vegetable_conditions", methods=["GET"])
def get_vegetable_conditions():
    response = jsonify({"vegetable_conditions": util_price_pred.get_vegetable_conditions()})
    response.headers.add("Access-Control-Allow-Origin", "*")

    return response

@app.route("/get_potato_varietys", methods=["GET"])
def get_potato_varietys():
    response = jsonify({"potato_variety": util_price_pred.get_potato_varietys()})
    response.headers.add("Access-Control-Allow-Origin", "*")

    return response

@app.route("/predict_potato_price", methods=["GET", "POST"])
def predict_potato_price():
    temp = float(request.form["temp"])
    disaster = request.form["disaster"]
    condition = request.form["condition"]
    variety = request.form["variety"]
    rainfall = float(request.form["rainfall"])
    origin = request.form["origin"]
    organic = request.form["organic"]
    location = request.form["location"]

    response = jsonify(
        {
            "predicted_price": numpy.float64(
                util_price_pred.get_predicted_price(
                    temp=temp, disaster=disaster, condition=condition, variety=variety, rainfall=rainfall, origin=origin, organic=organic, location=location
                )
            )
        }
    )
    response.headers.add("Access-Control-Allow-Origin", "*")

    return response


@app.route("/predict_weather", methods=["GET"])
def predict_weather():

    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    if start_date is None or end_date is None:
        return jsonify({"error": "Missing start_date or end_date parameter"}), 400

    try:
        predictions = util_weather_pred.load_and_predict_models(start_date, end_date)
        
        predictions_dict = {}
        for location, values in predictions.items():
            predictions_dict[location] = values.tolist()

        response = jsonify(predictions_dict)
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    print("Starting Python Flask Server...")
    util_price_pred.load_saved_artifacts()
    app.run()