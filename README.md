# Spam-Ham-Classification-Using-Naive-Bayes-and-Logistic-Regression
The code on compilation, does the following- 
1. Runs Naive Bayes Algorithm on the ham and spam dataset and returns the accuracies both with stop words and without them.
2. Runs Logistic regression with user inputtable i, e and l values to return accuracies both with stop words and without them.

The project contains the following- 
1.naiveBayesAndLogisticRegression.py - the script which performs the aforementioned tasks
2.stop_words_list.txt - The text file containing stop words.
3.The test and training data set folders
4.Other files which essentially are pre-computed and trained models on Naive-Bayes. Delete them to see a fresh model being generated(this has been done to save computational time)

To run the code, write the following command on the command prompt- 

Run the following command on the prompt:
python naiveBayesAndLogisticRegression.py 500 0.001 1
where 500 = i
0.001 = e
1 = l

Using these sample values, the accuracies were the following- 
Logistic Regression Accuracy without Stop Words : 94.76987447698745
Logistic Regression Accuracy with Stop Words : 92.88702928870293
Test Data Accuracy on Ham dataset:  96.26436781609196  %
Test Data Accuracy on Spam dataset:  90.76923076923077  %
Test Data Accuracy on Ham dataset without stop words:  97.98850574712644  %
Test Data Accuracy on Spam dataset without stop words:  87.6923076923077  %

Seeing the accuracies we saw the accuracies increase when the stop words were omitted, except in the case of spam dataset, in which case it decreased. According to my observations, stop words are generally words which donate no valuable information for the task of text classification. As there is not a standard and perfect stop word dictionary, we might be throwing away potentially useful information by using an off the shelf dictionary in the case of spam classification. 
