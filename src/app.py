from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Example Python function to generate data
def generate_data():
    return {
        "ROK":50,
        "JAP":70,
        "PRC":90,
        "ROC":45,
        "RUS":80
    }

# API endpoint to send data to the frontend
@app.route('/get-data', methods=['GET'])
def get_data():
    data = generate_data()  # Call the Python function
    return jsonify(data)    # Send data as JSON

if __name__ == '__main__':
    app.run(debug=True)