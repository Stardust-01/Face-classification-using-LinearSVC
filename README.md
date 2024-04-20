# Face-classification-using-LinearSVC
* Input must be a single Image having dimensions 250*250
* Cropping an Image and resizing it to 250 * 250 may lead to loss of information and some variability in results , this is the reason to provide 5 possible predictions with decreasing decision scores respectively.
* Resnet CNN model is used for feature extraction
* 128 features are selected after PCA
* Model is deployed online using streamlit sharing
