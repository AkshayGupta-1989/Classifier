# Classifier
Multi-class Document Classification:
I have used LinearSVC classifier (SVM based) from scikit-learn python library to build the classification model. 
The dataset provided was split into training and validation sets in the ratio of 67:33. According to the code, 
the model is dumped into a pickle file once, which can be used by the input docs (to be classified) and the predicted class along with its confidence score is returned. 
The choice of the classifier has been made by evaluating the performance on the validation dataset.
