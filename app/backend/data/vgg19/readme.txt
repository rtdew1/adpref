We need an optimized vector representation of the clothing images in order to map preferences in an accurate and memory-efficient manner. To do so, we can employee the use of transfer learning, which takes a pre-trained model architecture and uses the context/general embeddings to speed up the learning process. Therefore, only the last few layers, which are specific to the new task, are trained using the original model. In our case, we’re using the VGG-19 network. We are then taking the output of the last fully connected layer (dim size of 4096 x 1), and using PCA to reduce the dimensionality.

Therefore, the transfer learning procedure is:

1. Load in dress training data
2. Reshape/transform the RGB values of the images to be in the input shape of VGG-19
3. Build the VGG-19 model using our training data
4. Save the last layer (RELU-7) of each image as a vector

# Instructions for using Tensorflow VGG16/19

We use VGG-19 implemented in Tensorflow to get the transfer-learned embeddings. We use the following repository for the VGG-19 model: https://github.com/machrisaa/tensorflow-vgg 

Thus, step 1 is to clone and set up that repository. Once that is done, move your working directory to `tensorflow-vgg-master` and execute the following shell commands:

python get_fc_layers.py
python pca.py