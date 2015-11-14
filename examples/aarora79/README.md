# Generic Machine Learning Example

Code for a generic machine learning example implemented as a datascience pipeline (ingestion->wrangling->analysis->visualization). 
- The code takes a list of top level URLs as an input and does a first level parsing of these web pages looking for CSV files. 
- The downloaded CSV files are then analysed as per three ML models viz.  LogisticRegression, RandomForestClassifier and GaussianNB. 
- Classification scores are stored in a dataframe and written as a csv file at the end of the program. 
- A color coded heatmap is stored for a more visual comparison of the models.
- Visualization of the dataset in terms of scatter matrix, parallel coordinates and radviz is also done and all generated images are stored as png files.
- The estimator i.e. model is stored on disk as a python pickle file.

ToDo List:
- The models use all of the attributes as features i.e. it is assumed that the last column in the CSV file is the target variable and all other 
  columns represent features. This is not entirely desirable, feature analysis should be done to find out a subset of features which have a greater 
  impact on the target rather than using all the features, this would provide a more effective model.