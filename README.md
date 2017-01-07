# Relay Level Generator
Automatic level generation trained using reinforcement learning.
Uses A3C algorithm, OpenAI Gym, Keras and Tensorflow.

See `requirements.txt` for dependencies

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

Run server
```
python server.py
```

You can visit the server via `http://localhost:5000`

## References
- https://github.com/openai/gym/tree/master/gym/envs
- http://ai.berkeley.edu/project_overview.html
- https://github.com/sherjilozair/dqn
- https://github.com/tatsuyaokubo/dqn/
- http://www.nature.com/nature/journal/v518/n7540/abs/nature14236.html
