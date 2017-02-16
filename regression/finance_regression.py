#!/usr/bin/python

"""
    Starter code for the regression mini-project.
    
    Loads up/formats a modified version of the dataset
    (why modified?  we've removed some trouble points
    that you'll find yourself in the outliers mini-project).

    Draws a little scatterplot of the training/testing data

    You fill in the regression code where indicated:
"""

import sys
import pickle

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

dictionary = pickle.load(open("final_project/final_project_dataset_modified.pkl", "r"))

### list the features you want to look at--first item in the 
### list will be the "target" feature
features_list = ["bonus", "salary"]
data = featureFormat(dictionary, features_list, remove_any_zeroes=True)
target, features = targetFeatureSplit(data)
sort_keys = '../tools/python2_lesson06_keys.pkl'

### training-testing split needed in regression, just like classification
from sklearn.cross_validation import train_test_split

feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5,
                                                                          random_state=42)
train_color = "b"
test_color = "r"

### Your regression goes here!
### Please name it reg, so that the plotting code below picks it up and 
### plots it correctly. Don't forget to change the test_color above from "b" to
### "r" to differentiate training points from test points.
from sklearn import linear_model

reg = linear_model.LinearRegression()
reg.fit(feature_train, target_train)

reg.coef_
reg.intercept_
reg.score(feature_train, target_train)  # evaluate the train data
reg.score(feature_test, target_test)  # evaluate the test data

# regress the bonus against the long term incentive, and see if the regression score is
# significantly higher than regressing the bonus against the salary. Perform the regression
# of bonus against long term incentive--what’s the score on the test data?
features_list = ["long_term_incentive", "bonus"]
data = featureFormat(dictionary, features_list, remove_any_zeroes=True)
target, features = targetFeatureSplit(data)

feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5,
                                                                          random_state=42)
train_color = "b"
test_color = "r"

reg.fit(feature_train, target_train)
reg.score(feature_test, target_test)

# Let’s add a little hack to see what happens if it falls in the training set instead. Add these two lines
# near the bottom of finance_regression.py, right before
# plt.xlabel(features_list[1]):

reg.fit(feature_test, target_test)
plt.plot(feature_train, reg.predict(feature_train), color="b")

### draw the scatterplot, with color-coded training and testing points
import matplotlib.pyplot as plt

for feature, target in zip(feature_test, target_test):
    plt.scatter(feature, target, color=test_color)
for feature, target in zip(feature_train, target_train):
    plt.scatter(feature, target, color=train_color)

### labels for the legend
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")

### draw the regression line, once it's coded
try:
    plt.plot(feature_test, reg.predict(feature_test))
except NameError:
    pass

# reg.fit(feature_test, target_test)
# plt.plot(feature_train, reg.predict(feature_train), color="b")
# reg.fit(feature_test, target_test)

# Now we’ll be drawing two regression lines, one fit on the test data (with outlier) and one fit
# on the training data (no outlier). Look at the plot now--big difference, huh? That single outlier
#  is driving most of the difference. What’s the slope of the new regression line?

# (That’s a big difference, and it’s mostly driven by the outliers. The next lesson will dig into
# outliers in more detail so you have tools to detect and deal with them.)

plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()
