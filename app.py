from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
import io

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            df = pd.read_csv(file)
            result = analyze_data(df)
            return render_template('result.html', result=result)
        elif 'data' in request.form:
            data = request.form['data']
            df = pd.read_csv(io.StringIO(data))
            result = analyze_data(df)
            return render_template('result.html', result=result)
    return render_template('index.html')

def analyze_data(df):
    model = LinearRegression()
    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]   # Target
    model.fit(X, y)
    return f"Model coefficients: {model.coef_}, intercept: {model.intercept_}"

if __name__ == '__main__':
    app.run(debug=True)
