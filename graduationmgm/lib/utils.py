import datetime
import logging
import pickle

import numpy as np

logger = logging.getLogger(__name__)


def gen_mem_end(gen_mem, episode, model, frame_idx):
    gen_mem = False
    frame_idx = 0
    model.learn_start = 0
    logging.info('Start Learning at Episode %s', episode)


def episode_end(total_reward, episode, model):
    # Get the total reward of the episode
    logging.info('Episode %s reward %d', episode, total_reward)
    model.finish_nstep()
    model.reset_hx()


def save_modelmem(episode, test, model, model_path,
                  optim_path, mem_path, frame_idx, replay_size):
    if episode % 100 == 0 and episode > 0 and not test:
        model.save_w(path_model=model_path,
                     path_optim=optim_path)
    if ((frame_idx / 16) % replay_size) == 0 and episode % 1000 == 0:
        model.save_replay(mem_path=mem_path)
        logging.info("Memory Saved")


def save_rewards(rewards):
    day = datetime.datetime.now().today().day()
    hour = datetime.datetime.now().hour()
    minute = datetime.datetime.now().minute()
    final_str = str(day) + "-" + str(hour) + "-" + str(minute)
    with open('saved_agents/rewards_{}.pickle'.format(final_str),
              'w+') as fiile:
        pickle.dump(rewards, fiile)
        fiile.close()
