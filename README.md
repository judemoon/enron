# Identify Fraud from Enron Email and Financial datasets using Python

July-August 2017, by Jude Moon

## Overview

In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. 

In this project, I played a detective, and put the new skills to use by building a person of interest (POI) identifier based on financial and email data made public as a result of the Enron scandal. I used [the provided dataset](https://github.com/udacity/ud120-projects/tree/master/final_project) from [Udacity Intro to Machine Learning Course](https://www.udacity.com/course/intro-to-machine-learning--ud120), which was combined with a hand-generated list of POI in the fraud case. POIs are individuals who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity.


## Files
- final_project_dataset.pkl: main data file; data dictionary is stored as a pickle file
- poi_names.txt: supplementary data file
- enron61702insiderpay.pdf: supplementary information file
- enron_project.ipynb: documentation of algorithm analysis and answers to [a series of questions](https://docs.google.com/document/d/1NDgi1PrNJP7WTbfSUuRUnz8yzs5nGVTSzpO7oeNTEWA/pub?embedded=true)
- poi_id.py: python script to create three pickle files (my_dataset.pkl, my_classifier.pkl, my_feature_list.pkl) for the finalized classifier
- tester.py: python script to evaluate the three pickle files
- feature_format.py: python module to convert data dictionary to numpy array for sklearn modules