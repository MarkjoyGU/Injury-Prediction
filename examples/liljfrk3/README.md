# Modeling wine preferences by data mining from physicochemical properties.
By [liljfrk3](https://github.com/liljfrk3)

*BLUF*:

- It is possible to train a model
- It is possible to modify the features to increase the F1 score

### QUESTIONS FOR INSTRUCTOR:

- What does it mean when the F1 score = 1? Does this mean I likely did something wrong?

- In this notebook, I've trained a model using `n_folds=12`, but how do I save a 13th slice for testing the model? Or is that happening already with the 12th slice and I don't realize it?

- Is it common to add/delete some of the features analyzed in order to improve a model's prediction?

- Input variables (based on physicochemical tests): 1 - fixed acidity 2 - volatile acidity 3 - citric acid 4 - residual sugar 5 - chlorides 6 - free sulfur dioxide 7 - total sulfur dioxide 8 - density 9 - pH 10 - sulphates 11 - alcohol
Output variable (based on sensory data): 12 - quality (score between 0 and 10)

## Background

Two datasets were created, using red and white wine samples. **I only test the red samples in this notebook.

Inputs include objective tests (e.g. PH values) and the output is based on sensory data (median of at least 3 evaluations made by wine experts). Each expert graded the wine quality between 0 (very bad) and 10 (very excellent).

In the original ML analysis, several data mining methods were applied to model these datasets under a regression approach. The support vector machine model achieved the best results. Several metrics were computed: MAD, confusion matrix for a fixed error tolerance (T), etc.

Also, we plot the relative importances of the input variables (as measured by a sensitivity analysis procedure).

These datasets can be viewed as classification or regression tasks. The classes are ordered and not balanced (e.g. there are munch more normal wines than excellent or poor ones). Outlier detection algorithms could be used to detect the few excellent or poor wines.

Also, we are not sure if all input variables are relevant. So it could be interesting to test feature selection methods.

## Description 

- Number of Instances: red wine - 1599; white wine - 4898.
- Number of Attributes: 11 + output attribute

Note: several of the attributes may be correlated, thus it makes sense to apply some sort of feature selection.
Missing Attribute Values: None
