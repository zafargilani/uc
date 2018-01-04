# User Classification

Contributors: [Sarim Zafar](https://github.com/sarim-zafar/), [Usman Sarwar](mailto:usman.sarwar23@gmail.com), [Zafar Gilani](https://github.com/zafargilani/)

License: GPL 3.0

A library that classifies social media entities of Pakistan into any of the four main political entities per feature-set (Hashtags, Links, Friends, Tweets). The strength or accuracy of the classification is indicated by confidence levels in percentage.

## Installation

We require the following packages:
* [Miniconda](https://conda.io/miniconda.html) - to setup a base environment for managing dependencies.
* [Keras](https://keras.io/) - a high-level neural networks API running atop TensorFlow and Theano.
* [Theano](http://deeplearning.net/software/theano/) - for mathematical backend support in Python.
* [TensorFlow](https://www.tensorflow.org/) - for machine learning backend (Python).
* [Tweepy](http://www.tweepy.org/) - for accessing the Twitter APIs.
* [Spyre](https://github.com/adamhajari/spyre) - for web application purposes.
* [scikit-learn](http://scikit-learn.org/stable/) - machine learning library for Python.
* NumPy, SciPy, matplotlib - dependencies of scikit-learn.
  
Miniconda is the base environment for managing dependencies for this Python project. Install [Miniconda](https://conda.io/miniconda.html) for Python 3.6 using the guide, making sure the OS and the version. Usually (yes to all questions):
``` bash
sh Miniconda3.sh
```

After installing Miniconda, create the conda environment:
``` bash
source ~/.bashrc
source activate root
conda install python==3.5.2
```

Install dependencies:
``` bash
pip install pandas matplotlib DataSpyre tweepy sklearn scipy keras
pip install scikit-learn==0.19.0
pip install tensorflow
pip install h5py
conda install numpy scipy mkl mkl-service nose
conda install theano
```

Once installation of all dependencies is successful, exit the root environment and make the following change in "/home/user/.keras/keras.json":
``` bash
"backend": "theano",
```

## Allowing port 8080

We need to open listening access to port 8080 (required su access):
``` bash
sudo ufw allow 8080
```

Confirm if the allow action was successful:
``` bash
sudo ufw status
```

## Running the application

Run the application:
``` bash
cd ~/uc/
python app.py
```

Point the browser to http://localhost:8080/ or a public URL (if available) with port 8080, to use the application.

