import base64
import io
from flask import Flask, render_template, request, redirect, url_for, session, flash
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

# Training function for Random Forest
def train_rf(X_train, y_train, n_estimators=100, epochs=10):
    accuracies = []
    for epoch in range(epochs):
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        accuracy = model.score(X_train, y_train)
        accuracies.append(accuracy)
        print(f'RF Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.2f}')
        time.sleep(0.1)
    return accuracies

# Training function for SVM
def train_svm(X_train, y_train, C=1, gamma=0.1, epochs=10):
    accuracies = []
    for epoch in range(epochs):
        model = SVC(kernel='rbf', C=C, gamma=gamma)
        model.fit(X_train, y_train)
        accuracy = model.score(X_train, y_train)
        accuracies.append(accuracy)
        print(f'SVM Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.2f}')
        time.sleep(0.1)
    return accuracies

@app.route('/Graph')
def Graph():
    features, labels = fetch_and_process_data(USGS_API_URL)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Training with epochs for Random Forest
    rf_accuracies = train_rf(X_train, y_train, n_estimators=100, epochs=10)
    
    # Training with epochs for SVM
    svm_accuracies = train_svm(X_train, y_train, C=1, gamma=0.1, epochs=10)

    # Plot the training accuracy over epochs
    plt.figure()
    plt.plot(rf_accuracies, marker='o', linestyle='-', color='g', label='Random Forest')
    plt.plot(svm_accuracies, marker='o', linestyle='-', color='b', label='SVM')
    plt.title('Model Training Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Save the plot to a bytes object and encode it as a base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    # Predicting and computing test accuracy for comparison
    best_rf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    rf_test_accuracy = accuracy_score(y_test, best_rf_model.predict(X_test))

    best_svm_model = SVC(kernel='rbf', C=1, gamma=0.1).fit(X_train, y_train)
    svm_test_accuracy = accuracy_score(y_test, best_svm_model.predict(X_test))

    return render_template('area.html', plot_url=plot_url, svm_accuracy=svm_test_accuracy, rf_accuracy=rf_test_accuracy)

if __name__ == '__main__':
    app.run(debug=True)
