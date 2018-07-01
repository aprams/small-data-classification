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

In order to train with your own data, the default class paths are './data/class1' and './data/class2'. The name of the class folders doesn't matter, the path to them does though. You can specify your own path in the command line arguments.
Afterwards you can run the training with:
```
python training.py
```

Show possible flags with:
```
python training.py -h
```

## Deployment

Run the docker image with:

```
docker build -t classifier-server . && docker run -p 5000:5000 classifier-server:latest
```

You can then either use the 'upload_predict' endpoint at:
```
http://<DOCKER-IP>:5000/upload_predict
```

where you can upload an image an have it classified or specify an image_url at the 'predict' endpoint like this: 

```
curl http://<DOCKER-IP>:5000/predict?image_url=http://domain.com/image.jpeg
```
The result will be a json with fields 'result' indicating 'good' or 'bad' (adjustable to your classes) and the 'sigmoid_output' field, being a value between 0 and 1, indicating the network's last layer's output.

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


