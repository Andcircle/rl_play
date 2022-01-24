from email.mime import base
import sys
import gym
import numpy as np
from datetime import datetime
import os
import logging
import argparse

import matplotlib.pyplot as plt
from agents.ddpg import DDPGAgent
from utils.helpers import load_config, save_config, init_logging

from memories.replay_memory import ReplayMemory
from environments.scaling.continuous_scaling_env import ContinuousScalingEnv

import wandb

def _parse_args():
    parser = argparse.ArgumentParser(description='This script play with reinforcement learning',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # optional
    parser.add_argument('--config_path',
                        help='Path to config file storing hyper parameters.')
    
    parser.add_argument('--model_dir',
                        help='Dir to pretrained model, must combine with load_epoch')
    
    parser.add_argument('--load_epoch',
                        help='Dir to pretrained model, must combine with load_epoch')
    
    parser.add_argument('--mode',
                        default='train',
                        help='between train, eval, show, show will do the rendering job')

    return parser.parse_args()


def main():
    args = _parse_args()
    cur_dir = os.getcwd()
    
    config_path =  args.config_path if args.config_path else os.path.join(cur_dir, 'configs/config.yml')
    mode = args.mode
    
    base_dir = os.path.join(cur_dir, 'experiments')
    config_path = os.path.join(cur_dir, 'configs/config.yml')

    config = load_config(config_path)
    with wandb.init(project=config['project'], name=config['name'], config=config):
    
        experiment_time = datetime.now().strftime('%y_%m_%d_%H_%M')
        experiment_name = config['name'] + '_' + experiment_time
        experiment_dir = os.path.join(base_dir, experiment_name)

        os.makedirs(experiment_dir, 0o777, True)

        save_config(config, experiment_dir)
        init_logging(experiment_dir)

        # env = gym.make("Pendulum-v0")
        env = gym.make("Continuous-Scaling-v0")
        
        memory = ReplayMemory(config['memory_size'])
        agent = DDPGAgent(env, memory, config, **config)
        
        if args.model_dir and args.load_epoch:
            agent.load(args.model_dir, args.load_epoch)

        rewards = []
        avg_rewards = []

        for epoch in range(config['train_epochs']):
            state = env.reset()
            # noise.reset()
            agent.reset()
            episode_reward = 0
            
            for step in range(config['train_steps']):
                action = agent.get_action(state, step)
                # action = noise.get_action(action, step)
                new_state, reward, done, _ = env.step(action) 
                
                if mode == 'train':
                    memory.push(state, action, reward, new_state, done)
                    agent.update()        
                
                # if mode == 'show' or epoch % config['show_epochs'] == 0:
                #     env.render()
                    
                state = new_state
                episode_reward += reward

                if done:
                    break
                
            if epoch % config['log_epochs'] == 0:
                logging.info("episode: {}, steps: {}, reward: {}, average _reward: {} \n".format(epoch, step, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
                wandb.log({'reward': episode_reward}, step=epoch) 
                
            if epoch % config['save_epochs'] == 0:
                agent.save(experiment_dir, epoch)
                
            if epoch % config['eval_epochs'] == 0:
                pass
            
            rewards.append(episode_reward)
            avg_rewards.append(np.mean(rewards[-10:]))

    # plt.plot(rewards)
    # plt.plot(avg_rewards)
    # plt.plot()
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    # plt.show()


if __name__ == "__main__":
    main()