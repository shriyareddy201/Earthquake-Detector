import base64
import io
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flaskext.mysql import MySQL
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import requests
import numpy as np
import time
import os
import glob

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change to a strong, secret key

mysql = MySQL(app)
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = 'Ras@mysql9'
app.config['MYSQL_DATABASE_DB'] = 'earth1'
app.config['MYSQL_DATABASE_HOST'] = '127.0.0.1'

mysql.init_app(app)

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
        cursor.execute("INSERT INTO users (username, password, email) VALUES (%s, %s, %s)", (username, password, email))
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

    response = requests.get(USGS_API_URL)
    earthquake_data = response.json()

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

def fetch_and_process_data(USGS_API_URL):
    response = requests.get(USGS_API_URL)
    data = response.json()

    features = []
    labels = []
    for feature in data['features']:
        coords = feature['geometry']['coordinates']
        mag = feature['properties']['mag']
        features.append(coords[:2] + [coords[2], mag])
        labels.append(1 if mag >= 4.0 else 0)

    features = np.array(features)
    labels = np.array(labels)

    return features, labels

@app.route('/Graph')
def Graph():
    features, labels = fetch_and_process_data(USGS_API_URL)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    svm_param_grid = {
        'C': [0.1, 1, 10],
        'gamma': [1, 0.1, 0.01],
        'kernel': ['rbf']
    }
    rf_param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    svm_grid_search = GridSearchCV(SVC(), svm_param_grid, refit=True, verbose=1, cv=3)
    svm_grid_search.fit(X_train, y_train)
    best_svm = svm_grid_search.best_estimator_

    rf_grid_search = GridSearchCV(RandomForestClassifier(), rf_param_grid, refit=True, verbose=1, cv=3)
    rf_grid_search.fit(X_train, y_train)
    best_rf = rf_grid_search.best_estimator_

    svm_predictions = best_svm.predict(X_test)
    rf_predictions = best_rf.predict(X_test)

    svm_accuracy = accuracy_score(y_test, svm_predictions)
    rf_accuracy = accuracy_score(y_test, rf_predictions)

    plt.switch_backend('Agg')
    fig, ax = plt.subplots()
    bars = ax.bar(['SVM', 'Random Forest'], [svm_accuracy, rf_accuracy], color=['blue', 'green'])
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison')

    # for bar in bars:
    #     yval = bar.get_height()
    #     ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('area.html', plot_url=plot_url, svm_accuracy=svm_accuracy, rf_accuracy=rf_accuracy)

if __name__ == '__main__':
    app.run(debug=True)
