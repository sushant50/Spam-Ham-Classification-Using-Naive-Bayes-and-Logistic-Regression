#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 00:48:50 2018

@author: Sushant
"""
import os
import math
import pickle
import fnmatch
import nltk
import sys
from os import listdir
from os.path import isfile, join
from nltk.tokenize import wordpunct_tokenize
i = int(sys.argv[1])
e = float(sys.argv[2])
l = int(sys.argv[3])
classes = ['spam','ham'];
path = os.getcwd();
training_ham_path = path + '/train/ham/';
training_spam_path = path + '/train/spam/';
test_ham_path = path + '/test/ham/';
test_spam_path = path + '/test/spam/';

spam_train_size = len(fnmatch.filter(os.listdir(training_spam_path), '*.txt'))
ham_train_size = len(fnmatch.filter(os.listdir(training_ham_path), '*.txt'))
spam_test_size = len(fnmatch.filter(os.listdir(test_spam_path), '*.txt'))
ham_test_size = len(fnmatch.filter(os.listdir(test_ham_path), '*.txt'))
total_size = spam_train_size + ham_train_size

def simple_probablity(num,den):
    return num/den

ham_prior = simple_probablity(ham_train_size,total_size);
spam_prior = simple_probablity(spam_train_size,total_size);

def get_words(message, flag):

    all_words = set(wordpunct_tokenize(message.replace('=\\n', '').lower()))
#    msg_words = [word for word in all_words if word not in stopwords.words() and len(word) > 2]
    if flag=='true':
        
        msg_words = [word for word in all_words if len(word) > 2 and word not in stop_words] 
    else:
        msg_words = [word for word in all_words if len(word) > 2]   
    return msg_words
    
def get_mail_from_file(file_name):

    message = ''
    
    with open(file_name, encoding="latin-1") as mail_file:
        
        for line in mail_file:
            for line in mail_file:
                message += line
                    
    return message
    
stop_words_message = get_mail_from_file(path+'/' +'stop_words_list.txt')
stop_words = get_words(stop_words_message, 'false')   
    
def make_training_set(path, flag):

    training_set = {}
    mails_in_dir = [mail_file for mail_file in listdir(path) if isfile(join(path, mail_file))] 
    for mail_name in mails_in_dir:
        message = get_mail_from_file(path + mail_name)
        terms = get_words(message, flag)
        for term in terms:
            if term in training_set:
                training_set[term] = training_set[term] + 1
            else:
                training_set[term] = 1
    return training_set

print('')    
print('Loading training sets...');
def count_total_docs(training_set):
    total_count = 0;
    for term in training_set.keys():
        total_count = total_count + training_set[term]
    return total_count
def calculate_conditional_probablities(training_set, total_count, smoothing):
    for term in training_set.keys():
        training_set[term] = float(training_set[term]+1) / (total_count + smoothing)
    return training_set
        
exists = os.path.isfile(path + '/ham.file')
if exists:
    with open("spam.file", "rb") as f:
        spam_training_set = pickle.load(f)
    with open("ham.file", "rb") as f:
        ham_training_set = pickle.load(f)
    with open("spam_without_stop_words.file", "rb") as f:
        spam_training_set_without_stop_words = pickle.load(f)
    with open("ham_without_stop_words.file", "rb") as f:
        ham_training_set_without_stop_words = pickle.load(f)
else:
    spam_training_set = make_training_set(training_spam_path, 'false')
    
    ham_training_set = make_training_set(training_ham_path, 'false')
    spam_training_set_without_stop_words = make_training_set(training_spam_path, 'true')
    
    ham_training_set_without_stop_words = make_training_set(training_ham_path, 'true')

    spam_unique_word_count = len(spam_training_set.keys())
    ham_unique_word_count = len(ham_training_set.keys())
    spam_unique_word_count_without_stop_words = len(spam_training_set_without_stop_words.keys())
    ham_unique_word_count_without_stop_words = len(ham_training_set_without_stop_words.keys())
    spam_set = set(spam_training_set)
    ham_set = set(ham_training_set)
    common_count = 0
    for name in spam_set.intersection(ham_set):
        common_count = common_count + 1;
    smoothing_count = spam_unique_word_count + ham_unique_word_count - common_count;
    
    spam_set_w_s_w = set(spam_training_set_without_stop_words)
    ham_set_w_s_w = set(ham_training_set_without_stop_words)
    common_count_w_s_w = 0
    for name in spam_set_w_s_w.intersection(ham_set_w_s_w):
        common_count_w_s_w = common_count_w_s_w + 1;
    smoothing_count_w_s_w = spam_unique_word_count_without_stop_words + ham_unique_word_count_without_stop_words - common_count_w_s_w;
    
    total_spam_count = 0;
    total_ham_count = 0; 
    total_spam_count = count_total_docs(spam_training_set)
    total_ham_count = count_total_docs(ham_training_set)
    spam_training_set = calculate_conditional_probablities(spam_training_set,total_spam_count, smoothing_count)
    ham_training_set = calculate_conditional_probablities(ham_training_set,total_ham_count, smoothing_count)
    total_spam_count_w_s_w = 0;
    total_ham_count_w_s_w = 0; 
    total_spam_count_w_s_w = count_total_docs(spam_training_set_without_stop_words)
    total_ham_count_w_s_w = count_total_docs(ham_training_set_without_stop_words)
    spam_training_set = calculate_conditional_probablities(spam_training_set_without_stop_words,total_spam_count_w_s_w, smoothing_count_w_s_w)
    ham_training_set = calculate_conditional_probablities(ham_training_set_without_stop_words,total_ham_count_w_s_w, smoothing_count_w_s_w)
    with open("spam.file", "wb") as f:
        pickle.dump(spam_training_set, f, pickle.HIGHEST_PROTOCOL)
    with open("ham.file", "wb") as f:
        pickle.dump(ham_training_set, f, pickle.HIGHEST_PROTOCOL)
    with open("spam_without_stop_words.file", "wb") as f:
        pickle.dump(spam_training_set_without_stop_words, f, pickle.HIGHEST_PROTOCOL)
    with open("ham_without_stop_words.file", "wb") as f:
        pickle.dump(ham_training_set_without_stop_words, f, pickle.HIGHEST_PROTOCOL)
print('done.')
print('')

def classify(path, flag):
    mails_in_dir = [mail_file for mail_file in listdir(path) if isfile(join(path, mail_file))]
    classify = {}
    for mail_name in mails_in_dir:
        spam_probablity = math.log(spam_prior);
        ham_probablity = math.log(ham_prior);
        message = get_mail_from_file(path + mail_name)
        terms = get_words(message, flag)
        for term in terms:
            if term in spam_training_set:
                spam_probablity = spam_probablity + math.log(spam_training_set[term])
            else:
                spam_probablity = spam_probablity + math.log(1/(total_spam_count + smoothing_count))
            if term in ham_training_set:
                ham_probablity = ham_probablity + math.log(ham_training_set[term])
            else:
                ham_probablity = ham_probablity + math.log(1/(total_ham_count + smoothing_count))
        
        if spam_probablity < ham_probablity:
            classify[mail_name] = 'ham';
        else:
            classify[mail_name] = 'spam'
    return classify



ham_exists = os.path.isfile(path + '/ham_classify.file')
if ham_exists:
    with open("spam_classify.file", "rb") as f:
        classification_spam = pickle.load(f)
    with open("ham_classify.file", "rb") as f:
        classification_ham = pickle.load(f)
    with open("spam_classify_w_s_w.file", "rb") as f:
        classification_spam_w_s_w = pickle.load(f)
    with open("ham_classify_w_s_w.file", "rb") as f:
        classification_ham_w_s_w = pickle.load(f)
else:
    classification_ham = classify(test_ham_path, 'false')
    classification_spam = classify(test_spam_path, 'false')
    with open("spam_classify.file", "wb") as f:
        pickle.dump(classification_spam, f, pickle.HIGHEST_PROTOCOL)
    with open("ham_classify.file", "wb") as f:
        pickle.dump(classification_ham, f, pickle.HIGHEST_PROTOCOL)
        
    classification_ham_w_s_w = classify(test_ham_path, 'true')
    classification_spam_w_s_w = classify(test_spam_path, 'true')
    with open("spam_classify_w_s_w.file", "wb") as f:
        pickle.dump(classification_spam_w_s_w, f, pickle.HIGHEST_PROTOCOL)
    with open("ham_classify_w_s_w.file", "wb") as f:
        pickle.dump(classification_ham_w_s_w, f, pickle.HIGHEST_PROTOCOL)
def classify_ham_spam(classifications, classes):
    result = 0;
    for term in classifications.keys():
        if classifications[term] == classes:
            result = result + 1
    return result
        
ham_classified_as_ham = 0;
ham_classified_as_spam = 0; 
spam_classified_as_spam = 0;
spam_classified_as_ham = 0;
ham_classified_as_ham = classify_ham_spam(classification_ham, 'ham')
ham_classified_as_spam = len(classification_ham) - ham_classified_as_ham
spam_classified_as_spam = classify_ham_spam(classification_spam, 'spam')
spam_classified_as_ham = len(classification_spam) - spam_classified_as_spam

ham_classified_as_ham_wsw = 0;
ham_classified_as_spam_wsw = 0; 
spam_classified_as_spam_wsw = 0;
spam_classified_as_ham_wsw = 0;
ham_classified_as_ham_wsw = classify_ham_spam(classification_ham_w_s_w, 'ham')
ham_classified_as_spam_wsw = len(classification_ham_w_s_w) - ham_classified_as_ham_wsw
spam_classified_as_spam_wsw = classify_ham_spam(classification_spam_w_s_w, 'spam')
spam_classified_as_ham_wsw = len(classification_spam_w_s_w) - spam_classified_as_spam_wsw

def extract_features_from_mail(fileDir, filename):
    features = {'bias': 1.0}
    message = get_mail_from_file(fileDir + '/' + filename)
    terms = get_words(message,'false')
    for word in terms:
        features[word] = terms.count(word)
    return features

document_count = {'ham': 0.0, 'spam': 0.0}
class_probablity = {'ham': 1.0, 'spam': 0.0}

def separate_data_in_two_classes():
    mails_in_ham_dir = [mail_file for mail_file in listdir(training_ham_path) if isfile(join(training_ham_path, mail_file))]
    mails_in_spam_dir = [mail_file for mail_file in listdir(training_spam_path) if isfile(join(training_spam_path, mail_file))]
    separate_data = {'ham': [], 'spam': []}
    for mail_name in mails_in_ham_dir:
        message = get_mail_from_file(training_ham_path + mail_name)
        terms = get_words(message, 'false')
        separate_data['ham'].append(terms);
        document_count['ham'] += 1.0
    for mail_name in mails_in_spam_dir:
        message = get_mail_from_file(training_spam_path + mail_name)
        terms = get_words(message, 'false')
        separate_data['spam'].append(terms);
        document_count['spam'] += 1.0
    return separate_data



def weighted_sum(features, wts):
    weightedSum = 0.0
    for feature, value in features.items():
        if feature in wts:
            weightedSum += value * wts[feature]
    return weightedSum

def class_probability(features, wts):
    weightedSum = weighted_sum(features, wts)
    try:
        value = math.exp(weightedSum) * 1.0
    except OverflowError as exp:
        return 1
    return round((value) / (1.0 + value), 5)

def build_bag_of_words(data, skipWordsList):
    bag_of_words = []
    for class_name in data.keys():
        for array in data[class_name]:
            for term in array:
                if term not in bag_of_words and term.lower() not in skipWordsList:
                    bag_of_words.append(term)
    return bag_of_words

def build_logistic_regression_model(separated_data, bag_of_words, I, E, L):
    wts = initialize_weights(bag_of_words)
    for i in range(0, I):
        cost_function = {}
        for classType in separated_data:
            for item in separated_data[classType]:
                features = extract_features_from_document(item)
                error = class_probablity[classType] - class_probability(features, wts)
                if error != 0:
                    for feature in features.keys():
                        if feature in cost_function:
                            cost_function[feature] += (features[feature] * error)
                        else:
                            cost_function[feature] = (features[feature] * error)
        for wt in wts.keys():
            if wt in cost_function:
                wts[wt] = wts[wt] + (E * cost_function[wt]) - (E * L * wts[wt])
    return wts

def classify_using_logistic_regression(wts):
    accuracy = {1: 0.0, 0: 0.0}
    for filename in os.listdir(path + '/test/' + 'ham'):
        features = extract_features_from_mail(path+ '/test/' + 'ham', filename)
        classWeightedSum = weighted_sum(features, wts)
        if(classWeightedSum >= 0):
            accuracy[1] += 1.0
        else:
            accuracy[0] += 1.0
    for filename in os.listdir(path + '/test/' + 'spam'):
        features = extract_features_from_mail(path + '/test/' + 'spam', filename)
        classWeightedSum = weighted_sum(features, wts)
        if(classWeightedSum < 0):
            accuracy[1] += 1.0
        else:
            accuracy[0] += 1.0
    return (accuracy[1] * 100) / sum(accuracy.values())

def initialize_weights(bag_of_words):
    weights = {'bias': 0.0}
    for word in bag_of_words:
        weights[word] = 0.0
    return weights

def extract_features_from_document(doc):
    features = {'bias': 1.0}
    for word in doc:
        features[word] = doc.count(word)
    return features

separated_data = separate_data_in_two_classes()
bag_of_words = build_bag_of_words(separated_data, [])
trained_weights = build_logistic_regression_model(separated_data, bag_of_words, i, e, l)
print("Logistic Regression Accuracy without Stop Words : " + str(classify_using_logistic_regression(trained_weights)))
bag_of_words2 = build_bag_of_words(separated_data, stop_words)
trained_weights2 = build_logistic_regression_model(separated_data, bag_of_words2, i, e, l)
print("Logistic Regression Accuracy with Stop Words : " + str(classify_using_logistic_regression(trained_weights2)))

print("Test Data Accuracy on Ham dataset: ", ham_classified_as_ham/( ham_classified_as_spam + ham_classified_as_ham)*100," %")
print("Test Data Accuracy on Spam dataset: ", spam_classified_as_spam/ (spam_classified_as_ham + spam_classified_as_spam)*100," %")

print("Test Data Accuracy on Ham dataset without stop words: ", ham_classified_as_ham_wsw/( ham_classified_as_spam_wsw + ham_classified_as_ham_wsw)*100," %")
print("Test Data Accuracy on Spam dataset without stop words: ", spam_classified_as_spam_wsw/ (spam_classified_as_ham_wsw + spam_classified_as_spam_wsw)*100," %")