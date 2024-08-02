from flask import Flask, request, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            df = pd.read_csv(file)
        elif 'data' in request.form:
            data = request.form['data']
            df = pd.DataFrame([x.split(',') for x in data.split('\n')])
        
        # Perform some analysis
        # For simplicity, we'll standardize the data
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)
        return render_template('index.html', tables=[df_scaled.to_html(classes='data')])
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
