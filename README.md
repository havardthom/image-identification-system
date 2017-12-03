# README

Capstone project 15.08.2016 - 01.12.2016

Automatic Rodent Identification in Camera Trap Images using Deep Convolutional Neural Networks

### Train and test:

- To train a model the given training directory must contain class folders with images. Example structure: rodentcam_training_10k_11c/Weasel/img.jpg
- To predict test images, the training directory used to train the model must be given for the system to get appropriate class names.

### Run:

Use -h command to see arguments.

```Shell
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python run.py
```

### Dependencies:

Theano, Keras, Matplotlib, Numpy, Scikit-learn, OpenCV
