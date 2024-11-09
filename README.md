
<h1><b>Earthquake Detection Project</b></h1>
<h2>Overview</h2>
 <p>This project utilizes Python to analyze and detect seismic activity. It leverages real-time earthquake data from the United States Geological Survey (USGS) to provide insights and visualizations about earthquake occurrences around the globe.
</p>

<h2>Dataset</h2>
<p>The data for this project is sourced from the USGS Earthquake Hazards Program, which provides a real-time feed of earthquakes with a magnitude of 2.5 or higher from the past day. The dataset can be accessed directly via the following GeoJSON feed: USGS Earthquake Data.</p>

<h2>Project Structure</h2>

```
EARTHQUAKE DETECTION
│
├── app.py                  - Main Python script to launch the web application.
├── db.sql                  - SQL script to create the database schema.
├── Earthquake_data_processed.xlsx  - Processed earthquake data stored in Excel format.
├── indian_Earthquake_Live.txt      - Live data examples focused on Indian seismic activity.
├── map.py                  - Script for generating earthquake data maps.
├── newapp.py               - Auxiliary Python script.
├── newnew.py               - Additional auxiliary Python script.
├── newnewnew.py            - Further auxiliary Python script.
├── register                - Contains registration logic for a web application.
├── requirements.txt        - Required Python libraries to run the project.
├── train                   - Machine learning module or script for training models on earthquake data.
│
├── static/                 - Folder containing static files like CSS/JS.
└── templates/              - HTML templates for the web interface.
```

<h2>Installation</h2>
<p>To run this project, you will need Python 3.6 or higher. First, clone the repository and navigate to the project directory. Then install the required dependencies:</p>
``
pip install -r requirements.txt 
``

<h2>Usage</h2>
<p>To start the application, run:</p>
``` python app.py```
<p>This will start a local web server. Access the web application by navigating to http://localhost:5000 in your web browser.</p>

<h2>Features</h2>
<b>Real-time Data Analysis:</b> Analyze real-time earthquake data.</br>
<b>Data Visualization:</b> Generate maps and other visualizations to represent earthquake data.</br>
<b>Historical Data Tracking:</b> Utilize processed historical earthquake data for trend analysis and pattern recognition.</br>

<h2>Contributing</h2>
Contributions to this project are welcome! Please fork the repository and submit a pull request with your proposed changes.

<h2>License</h2>
Distributed under the MIT License. See LICENSE for more information.

