from flask import Flask, render_template
import requests
import numpy as np
from sklearn import svm, ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)


USGS_API_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson"

def fetch_and_process_data(USGS_API_URL):
    response = requests.get(USGS_API_URL)
    data = response.json()

    features = np.random.rand(100, 4)  # 100 samples, 4 features each
    labels = np.random.randint(0, 2, 100)  # Binary labels

    return features, labels

@app.route('/')
def index():
    # Fetch and process the data
    features, labels = fetch_and_process_data(USGS_API_URL)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

    # Initialize and train the SVM and Random Forest models
    svm_model = svm.SVC()
    rf_model = ensemble.RandomForestClassifier()

    svm_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    # Make predictions and calculate accuracies
    svm_predictions = svm_model.predict(X_test)
    rf_predictions = rf_model.predict(X_test)

    svm_accuracy = accuracy_score(y_test, svm_predictions)
    rf_accuracy = accuracy_score(y_test, rf_predictions)

    # Create a bar graph comparing accuracies
    plt.switch_backend('Agg')  # Use this backend to avoid the need for a display server
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
    return render_template('index.html', plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
