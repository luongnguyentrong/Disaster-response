# Udacity Data Science Nanodegree
## Disaster Reponse Project

### Table of Contents

2. [Project description](#description)
3. [File Descriptions](#files)
4. [How to run this project ?](#tutorial)
5. [Acknowledgements](#ack)

## Project Motivation<a name="description"></a>

The Udacity Disaster Response Project aims to improve the efficiency of managing natural disasters and emergencies. In critical situations, quick and accurate information is essential for effective relief efforts. By using data science and machine learning, this project categorizes and prioritizes disaster-related messages from social media and other channels. The goal is to enhance disaster response speed and accuracy, ensuring timely resource allocation to those in need, demonstrating the impactful role of data science in real-world challenges.

Finally, this project contains a web app where you can input a message and get classification results.

![Screenshot of Web App](web_screenshot.PNG)

## File Description<a name="files"></a>
~~~~~~~
        disaster_response_pipeline
          |-- app
                |-- templates
                        |-- go.html
                        |-- master.html
                |-- run.py
          |-- data
                |-- disaster_messages.csv       # message contained the message about the disasters from various source
                |-- disaster_categories.csv     # the categories of messages
                |-- DisasterResponse.db
                |-- process_data.py
          |-- models
                |-- classifier.pkl
                |-- train_classifier.py
          |-- Preparation
                |-- categories.csv
                |-- ETL Pipeline Preparation.ipynb
                |-- ETL_Preparation.db
                |-- messages.csv
                |-- ML Pipeline Preparation.ipynb
                |-- README
          |-- README
~~~~~~~

### How to run this project ?<a name="tutorial"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage


## Acknowledgements<a name="ack"></a>
* [Udacity](https://www.udacity.com/) for providing an excellent Data Scientist training program.