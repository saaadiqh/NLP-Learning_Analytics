# NLP & Learning Analytics
2023/24 CS350 Data Science Project - The University of Warwick

## Title
Developing Natural Language Processing tools to enhance Learning Analytics in Higher Education.
Creating an automated dashboard that diagnoses strengths and weaknesses from educational data using NLP concepts in the following order:

1. Sentiment Analysis: seperate strengths from areas for improvements.

2. Topic Modelling: identify overarching topics in each sentiment.

3. Extractive Text Summarisation: identify themes and key talking points in each topic.

4. Abstractive Text Summarisaton (optional): create summary diagnosis for each topic.


https://github.com/saaadiqh/NLP_Learning_Analytics/assets/119862810/1de690ce-a317-48b1-9a06-c961f1766d1a


## Requirements
Python 3.11

All libraries and packages from 'requirements.txt' file

It is recommended to create a virtual environment in which to install the above dependencies.

## Installation and Running
To generate the dashboard UI that produces the model results using the command line:

1. From command line, go to the directory of the project folder (called code):

<code> cd file_path/code</code>

2. Create a virtual environment in this directory (must have python3 already installed on the computer):

 <code> python3 -m venv environment_name</code>

3. Activate the virtual environment:

<code>  source environment_name/bin/activate</code>

4. Install the required libraries and packages from requirements.txt file using PIP (must install if you do not have):

<code>  pip install -r requirements.txt </code>

5. Run the code: 

<code>  environment_name/bin/python3.11 "src/main.py" </code>


## Contributors
Main contributor: Sadiq Habib