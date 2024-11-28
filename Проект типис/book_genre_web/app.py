from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)


model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:

        user_rating = float(request.form['user_rating'])
        reviews = int(request.form['reviews'])
        price = float(request.form['price'])
        year = int(request.form['year'])


        input_data = np.array([[user_rating, reviews, price, year]])


        if hasattr(model, 'predict'):
            prediction = model.predict(input_data)
            genre = 'Fiction' if prediction[0] == 'Fiction' else 'Non-Fiction'
            return render_template('result.html', prediction=genre)
        else:
            raise Exception("Модель не поддерживает метод 'predict'.")
    except Exception as e:
        return render_template('index.html', prediction=f'Ошибка: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
