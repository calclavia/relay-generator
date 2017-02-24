# Relay Level Generator
Relay is an action game that tests your reflex skills! In every level, your mission is to relay the circle to the goal. Relay uses deep learning to create levels that match your skill level, with unique levels every single game adapted to you.

This repository contains the architecture used to train the level generator in Relay. Automatic level generation is trained using reinforcement learning. For more details,
see blog posts https://medium.com/@henrymao/reinforcement-learning-using-asynchronous-advantage-actor-critic-704147f91686#.n1mvh7mpg and https://medium.com/@henrymao/applying-deep-learning-to-puzzle-level-design-15e9372b562#.z1tln9rzr.

### Training Graph
![Training Graph](https://i.gyazo.com/f4b6db2118fd04f091201f8d4f3db0c1.png)

### Generated Level
![Generated Level](https://i.gyazo.com/54eb049d382d5f0236c6878b90487134.png)

Download Relay on the app store!
- App Store https://itunes.apple.com/us/app/relay/id1203089200
- Google Play https://play.google.com/store/apps/details?id=com.calclavia.relay

## Dependencies
Runs with Python 3.5.

We used A3C algorithm with OpenAI Gym, Keras and Tensorflow as our primary tools.

See `requirements.txt` for all dependencies.

To setup Python, run:
```
pip install -r requirements.txt
```

## Train Model
To train your own model, run:
```
python main.py
```

We have a sample model in the archives folder that is used in our production game.

## Run Model
To run the model and generate a level, run:
```
python main.py -r True
```
This will output the level as an array to the console.

For the game Relay, we used Keras.js to run the model in production environment (https://github.com/transcranial/keras-js).

## References
- https://github.com/openai/gym/tree/master/gym/envs
- https://arxiv.org/pdf/1602.01783v2.pdf
