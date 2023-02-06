# Description
This is the project for CS 6190 Probabilistic Learning Spring 2022 at the University of Utah.
This project consists of predicting certain covid-19 data using hidden markov models and markov chains.  

# Dataset
The dataset is taken from the Utah Coronavirus dashboard The data was downloaded from the Utah coronavirus dashboard on February 25, 2022 
and contains data from March 18, 2020 through February 24, 2002.  The data consisted of 722 samples and contained daily reported Covid-19 
case counts, total active case counts and daily death counts in the state of Utah. The data has been gathered into the covid_data.csv file.

The data is processed and the labels are created using the following command:

`python get_covid_data.py`

# Experiments
The experiments are described in [this paper](https://github.com/JanaanL/Covid_19_Project/blob/main/Final_Project_Report.pdf "report"). To run the hidden markov models for the numeric metrics (change in daily case counts, change in active
case count, change in daily deaths) use the following command:

`python hmm_1.py`

To run the hidden markov model for the labelled metrics run the following command:

`python hmm_2.py`

To run the markov model and baseline calculations for the numeric metrics run the following command:

`ptyhon markov_model_1.py`

To run the markov model and baseline calculations for the numeric metrics run the following command:

`python markov_model_2.py`
