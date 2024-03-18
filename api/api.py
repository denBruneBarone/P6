from flask import Flask, jsonify, request
import json

app = Flask(__name__)


@app.route('/', methods=["GET"])
def index():
    return jsonify('hello world!')


@app.route('/predict', methods=["POST"])
def predict():
    print("Received POST request...")
    print("Predicting power consumption for given route")

    try:
        post_data = request.get_data().decode('utf-8')
        post_json = json.loads(post_data)

        print(post_json)

        power_consump_wh = 23.23
        response_data = {
            "wh": power_consump_wh
        }
        return jsonify(response_data), 200

    except Exception as e:
        return jsonify(error=f"Error occured: {str(e)}"), 422


@app.errorhandler(404)
def page_not_found(error):
    message = "Invalid endpoint"
    return jsonify(error=message), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
