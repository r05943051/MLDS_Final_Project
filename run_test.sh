
gzip -d ./MNIST_data/train-images-idx3-ubyte.gz
gzip -d ./MNIST_data/train-labels-idx1-ubyte.gz
gzip -d ./MNIST_data/t10k-images-idx3-ubyte.gz
gzip -d ./MNIST_data/t10k-labels-idx1-ubyte.gz
python2 src/test.py 
