# Small Data Classification

This project allows binary image classification for small datasets, based on retraining the final layer of a MobileNetV2.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

For training a new network you will need python3 and for the inference server a Dockerfile is provided. Running that on your own is possible, but Docker is recommended.


### Installing

In order to train your own networks and start developing, you will need to install the requirements, preferably in a virtual environment:


```
virtualenv -p python3.6 .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Afterwards you can run the training with:
```
python training.py
```

Show possible flags with:
```
python training.py -h
```

## Deployment

## Built With

* [Docker](https://www.docker.com/) - Container Framework
* [Flask](http://flask.pocoo.org/) - Slim Web Framework
* [Keras](https://keras.io/) - High-level Deep Learning framework
* [Tensorflow](https://www.tensorflow.org/) - Deep Learning backend for Keras

## Authors

* **Alexander Prams** - [aprams](https://github.com/aprams)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Tensorflow authors for their awesome framework and the Dockerfile template


