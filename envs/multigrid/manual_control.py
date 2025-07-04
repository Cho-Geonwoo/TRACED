# Copyright (c) 2019 Maxime Chevalier-Boisvert.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

#!/usr/bin/env python3

import time
import argparse
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from . import window as multigrid_window

from .maze import *
from .mst_maze import *
from .crossing import *
from envs.registration import make as gym_make


def redraw(img):
    if not args.agent_view:
        img = env.render("rgb_array", tile_size=args.tile_size)

    window.show_img(img)


def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()

    if hasattr(env, "mission"):
        print("Mission: %s" % env.mission)
        window.set_caption(env.mission)

    redraw(obs)


def step(action):
    print("taking action", action)
    obs, reward, done, info = env.step(action)
    # print('step=%s, reward=%.2f' % (env.step_count, reward))

    if done or action == env.actions.done:
        print("done!")
        reset()
    else:
        redraw(obs)


def key_handler(event):
    print("pressed", event.key)

    if event.key == "escape":
        window.close()
        return

    if event.key == "backspace":
        reset()
        return

    if event.key == "left":
        step(env.actions.left)
        return
    if event.key == "right":
        step(env.actions.right)
        return
    if event.key == "up":
        step(env.actions.forward)
        return

    # Spacebar
    if event.key == " ":
        step(env.actions.toggle)
        return
    if event.key == "pageup":
        step(env.actions.pickup)
        return
    if event.key == "pagedown":
        step(env.actions.drop)
        return

    if event.key == "enter":
        step(env.actions.done)
        return


parser = argparse.ArgumentParser()
parser.add_argument(
    "--env", help="gym environment to load", default="MultiGrid-MultiRoom-N4-S5-v0"
)
parser.add_argument(
    "--seed", type=int, help="random seed to generate the environment with", default=-1
)
parser.add_argument(
    "--tile_size", type=int, help="size at which to render tiles", default=32
)
parser.add_argument(
    "--agent_view",
    default=False,
    help="draw the agent sees (partially observable view)",
    action="store_true",
)
parser.add_argument(
    "--use_walls",
    default=False,
    action="store_true",
    help="draw the agent sees (partially observable view)",
)

args = parser.parse_args()

env = gym_make(args.env)

window = multigrid_window.Window("gym_minigrid - " + args.env)
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)
