from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
filename = 'file_StudentsPerformance.pkl'
model = pickle.load(open(filename, 'rb'))

# Initialize an empty list to store predictions
predictions = []

@app.route('/')
def index():
    return render_template('index.html', predictions=predictions)

@app.route('/predict', methods=['POST'])
def predict():
    math_score = request.form['math_score']
    reading_score = request.form['reading_score']
    writing_score = request.form['writing_score']
    
    pred = model.predict(np.array([[math_score, reading_score, writing_score,]]))
    result = f"Math Score: {math_score}, Reading Score: {reading_score}, Writing Score: {writing_score} => Prediction: {pred[0]}"
    
    # Append the result to the list of predictions
    predictions.append(result)
    
    # Display the results and clear the form
    return render_template('index.html', predict=pred[0], predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
