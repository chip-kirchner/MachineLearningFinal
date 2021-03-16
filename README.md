# MachineLearningFinal
Final Project code for CSS581 Machine Learning Winter Quarter 2021

Raw data can be retrieved from:

https://www.ncsbe.gov/results-data/voter-history-data
https://www.ncsbe.gov/results-data/voter-registration-data

Data is first parsed using the parse_history.py and the parse_registration.py files.
Run each script individually (not dependent on order).

Data is then cleaned and formatted via the data_clean.py script file.

Machine learning model training and evaluation takes place in the gridsearch_MLP.py file.


Steps:
Download datasets from ncsbe.gov
Save to sub-directory "/data"
Run parse_history.py
Run parse_registration.py
Run data_clean.py
  Clean data is linked in pdf submission
Run gridsearch_MLP.py
  If file takes too long two parameters can lower the amount of data used to train the file
    cv controls the number of cross-validations (set to 10)
    nrows controls the number of rows of data read into the file (at 100,000 rows file takes 5+ hours to compute)
  
