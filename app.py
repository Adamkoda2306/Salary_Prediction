import joblib
from dotenv import load_dotenv
from flask import Flask, render_template, request
import os
from flask_cors import CORS


load_dotenv()

app = Flask(__name__)
CORS(app)


#Get Environment Variables
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG')

# Load the model using joblib
with open("salary_predict_model.pkl", "rb") as file:
    model = joblib.load(file)
print(f"Loaded model type: {type(model)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input from the user
        experience = float(request.form['experience'])
        
        # Prepare the input data in 2D list format
        exp_data = [[experience]]
        
        # Make a prediction
        prediction = model.predict(exp_data)
        
        return render_template('index.html', prediction_text=f"Predicted Salary: ₹{prediction[0]:.2f}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
