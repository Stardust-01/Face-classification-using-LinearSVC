# Face-classification-using-LinearSVC
* The webdemo takes an image as an input and returns 5 possible person name's in decreasing precedence.
* The model showing best results on training and validation data of provided lfw dataset is a support vector classifier with linear kernel taking 128 PCA features as input.
* Input must be a single Image having dimensions 250*250.
* Cropping an Image and resizing it to 250 * 250 may lead to loss of information and some variability in results , this is the reason to provide 5 possible predictions with decreasing decision scores respectively.
* Resnet CNN model is used for feature extraction.
* 128 features are selected after PCA.
* Model is deployed online using streamlit sharing.
* You can download the LFW dataset [here](http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz).

