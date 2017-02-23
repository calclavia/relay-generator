# Relay Level Generator
Automatic level generation trained using reinforcement learning for Relay.
Uses A3C algorithm with OpenAI Gym, Keras and Tensorflow.

See `requirements.txt` for dependencies

Runs with Python 3.5.

## Running using Virtualenv
Install Virtualenv
```
sudo pip install virtualenv
```

Create an isolated Python environment, and install dependencies:
```
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

## Train Model
```
python main.py
```

## Run Model
```
python main.py -r True
```

For Relay, we used Keras.js to run the model in production environment (https://github.com/transcranial/keras-js).

## References
- https://github.com/openai/gym/tree/master/gym/envs
- http://ai.berkeley.edu/project_overview.html
- https://arxiv.org/pdf/1602.01783v2.pdf
- http://www.nature.com/nature/journal/v518/n7540/abs/nature14236.html
