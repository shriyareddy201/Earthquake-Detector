import base64
import io
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flaskext.mysql import MySQL
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from flask import Flask, render_template, request
import requests
import numpy as np
#import os
app = Flask(__name__)
#secret_key=os.urandom(24)
app.secret_key = 'your_secret_key'  # Change to a strong, secret key
mysql = MySQL(app)

# MySQL configuration
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = 'Ras@mysql9'
app.config['MYSQL_DATABASE_DB'] = 'earth1'
app.config['MYSQL_DATABASE_HOST'] = '127.0.0.1'  # Modify as needed

mysql.init_app(app)


       
# Set the title and logo
app.config['APP_TITLE'] = "EARTHQUAKE WARNING USING MACHINE LEARNING ALGORITHMS"
app.config['APP_LOGO'] = './static/newlogo.png'

@app.route('/')
def home():
    if 'username' in session:
        return render_template('home.html', title=app.config['APP_TITLE'], logo=app.config['APP_LOGO'])
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cursor = mysql.get_db().cursor()
        cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
        account = cursor.fetchone()
        if account:
            session['username'] = username
            flash('Logged in successfully', 'success')
            return redirect(url_for('home'))
        else:
            flash('Login failed. Please check your credentials and try again.', 'danger')
    return render_template('login.html', title=app.config['APP_TITLE'], logo=app.config['APP_LOGO'])

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        cursor = mysql.get_db().cursor()
        cursor.execute("INSERT INTO users (username, password, email) VALUES (%s, %s, %s)", (username, password,email))
        mysql.get_db().commit()
        flash('Registration successful. Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title=app.config['APP_TITLE'], logo=app.config['APP_LOGO'])

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('Logged out', 'success')
    return redirect(url_for('login'))

USGS_API_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson"


@app.route('/livedata', methods=['POST'])
def predict():
    latitude = float(request.form['latitude'])
    longitude = float(request.form['longitude'])

    # Make a request to the USGS API for earthquake data
    response = requests.get(USGS_API_URL)
    earthquake_data = response.json()

    # Process the earthquake data to determine alert and affected area
    earthquakes_in_region = 0
    total_magnitude = 0
    for earthquake in earthquake_data['features']:
        earthquake_lat = earthquake['geometry']['coordinates'][1]
        earthquake_lng = earthquake['geometry']['coordinates'][0]
        earthquake_magnitude = earthquake['properties']['mag']

        if (latitude - 1 <= earthquake_lat <= latitude + 1) and (longitude - 1 <= earthquake_lng <= longitude + 1):
            earthquakes_in_region += 1
            total_magnitude += earthquake_magnitude

    if earthquakes_in_region > 0:
        alert_message = "Alert: Yes, There have been earthquakes in the region."

        magnitude_to_area_factor = 10

        affected_area = f"{total_magnitude * magnitude_to_area_factor} km"
    else:
        alert_message = "No immediate earthquake threat detected in the region."
        affected_area = "No area affected."       

    return render_template('area.html', alert_message=alert_message, affected_area=affected_area)

# USGS_API_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson"

def fetch_and_process_data(USGS_API_URL):
    response = requests.get(USGS_API_URL)
    data = response.json()

    features = np.random.rand(100, 4)  
    labels = np.random.randint(0, 2, 100)  

    return features, labels

@app.route('/Graph')
def Graph():

    features, labels = fetch_and_process_data(USGS_API_URL)  
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
    svm_model = svm.SVC()
    rf_model = ensemble.RandomForestClassifier()

    svm_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    svm_predictions = svm_model.predict(X_test)
    rf_predictions = rf_model.predict(X_test)

    svm_accuracy = accuracy_score(y_test, svm_predictions)
    rf_accuracy = accuracy_score(y_test, rf_predictions)

    # Create a bar graph comparing accuracies
    plt.switch_backend('Agg')  
    fig, ax = plt.subplots()
    ax.bar(['SVM', 'Random Forest'], [svm_accuracy, rf_accuracy], color=['blue', 'green'])
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison')

    # Save the plot to a PNG image in memory
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    # Render the plot in the index.html template
    return render_template('area.html', plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
