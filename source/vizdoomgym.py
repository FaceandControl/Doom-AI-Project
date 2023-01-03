from vizdoom import DoomGame
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import cv2


class VizDoomGym(Env):
    def __init__(self, show=False):
        self.game = DoomGame()
        self.game.load_config("./scenarious/basic.cfg")
        self.game.init()

        self.game.set_window_visible(show)

        self.observation_space = Box(low=0, high=255, shape=(3,240,320), dtype=np.uint8)
        self.action_space = Discrete(3)

    def step(self, action):
        actions = np.identity(3, dtype=np.uint8)
        reward = self.game.make_action(actions[action], 4)

        if self.game.get_state():
            state = self.game.get_state()
            image = state.screen_buffer
            info = state.game_variables
        else:
            image = np.zeros(self.observation_space.shape)
            info = 0
        
        done = self.game.is_episode_finished()

        return image, reward, done, info 

    def reset(self):
        self.game.new_episode()
        return self.game.get_state().screen_buffer

    def grayscale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (100, 160, 1))
        return state

    def close(self):
        self.game.close()