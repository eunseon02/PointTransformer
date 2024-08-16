# task specification
from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin.rsg_raibo_rough_terrain import RaisimGymRaiboRoughTerrain
from raisimGymTorch.env.bin.rsg_raibo_rough_terrain import NormalSampler
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param
import os
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo_LSTM_v2 as PPO
import torch.nn as nn
import numpy as np
import torch
import argparse
import wandb
import datetime
# from raisimGymTorch.helper.Planner import *
from raisimGymTorch.helper.map_process_helper import *
from raisimGymTorch.algo.PointTransFormer.feature_model import PointTransformerV3ForGlobalFeature
import cProfile

# task specification

# os.environ["WANDB_API_KEY"] = '3bdcf6389f74ce8110d7914041ec50f6771bbee8'
# os.environ["WANDB_MODE"] = "dryrun"

# initialize wandb


# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default='')
parser.add_argument('--wandb', action='store_true', help='wandb on/off')
parser.add_argument('--num_env', type=int, default=1, help='num env')
parser.add_argument('--num_thread', type=int, default=30, help='num thread')
args = parser.parse_args()
mode = args.mode
weight_path = args.weight

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
if mode == 'retrain':
    cfg['environment']['curriculum']['initial_factor'] = 1

cfg['environment']['num_envs'] = args.num_env
cfg['environment']['num_threads'] = args.num_thread
env = VecEnv(RaisimGymRaiboRoughTerrain(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])
print("env create pass")
# wandb name
name = cfg['name']
task_name = cfg['task_name']


# Encoding
historyNum = cfg['environment']['dimension']['historyNum_']
pro_dim = cfg['environment']['dimension']['proprioceptiveDim_']
ext_dim = cfg['environment']['dimension']['exteroceptiveDim_']
inertial_dim = cfg['environment']['dimension']['inertialparamDim_']
ROA_ext_dim = ext_dim - inertial_dim

# shortcuts
act_dim = env.num_acts
Encoder_ob_dim = historyNum * inertial_dim

# LSTM
hidden_dim = cfg['LSTM']['hiddendim_']
batchNum = cfg['LSTM']['batchNum_']
layerNum = cfg['LSTM']['numLayer_']

# ROA Encoding
ROA_mode = cfg['environment']['ROAMode']
ROA_Encoder_ob_dim = historyNum * (pro_dim + ROA_ext_dim + act_dim)

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs
num_learning_epochs = 16
num_mini_batches = 1

# PPO coeff
entropy_coeff_ = cfg['environment']['entropy_coeff']

# Vision
sample_width = cfg['environment']['vision']['sample_width']
sample_height = cfg['environment']['vision']['sample_height']
world_size = cfg['environment']['vision']['world_size']
output_size = cfg['environment']['vision']['output_size']

#
seq_num = int(cfg['environment']['max_time'] / cfg['environment']['control_dt'])


# Proifle
profile_mode = cfg['environment']['profileMode']

Estimator = ppo_module.Estimator(ppo_module.MLP(cfg['architecture']['estimator']['net'],
                                                nn.LeakyReLU,
                                                hidden_dim,
                                                inertial_dim), device=device)

Encoder_ROA = ppo_module.Encoder(architecture=ppo_module.LSTM(input_dim=int(ROA_Encoder_ob_dim/historyNum),
                                                          hidden_dim=hidden_dim,
                                                          ext_dim=ROA_ext_dim,
                                                          pro_dim=pro_dim,
                                                          act_dim=act_dim,
                                                          inertial_dim=inertial_dim,
                                                          hist_num=historyNum,
                                                          device=device,
                                                          batch_num=batchNum,
                                                          layer_num=layerNum,
                                                          num_minibatch = num_mini_batches,
                                                          num_env=env.num_envs,
                                                              seq_num=seq_num), device=device)


Encoder = ppo_module.Encoder(architecture=ppo_module.LSTM(input_dim=int(Encoder_ob_dim/historyNum),
                          hidden_dim=hidden_dim,
                          ext_dim=ext_dim,
                          pro_dim=pro_dim,
                          act_dim=act_dim,
                          inertial_dim=inertial_dim,
                          hist_num=historyNum,
                          batch_num=batchNum,
                          layer_num=layerNum,
                          device=device,
                          num_minibatch = num_mini_batches,
                          num_env=env.num_envs,
                                                          seq_num=seq_num), device=device)

pre_process_dim = 64

PointTransformer = PointTransformerV3ForGlobalFeature().to(device)

Encoder_lidar = ppo_module.Encoder(architecture=ppo_module.LSTM(input_dim=pre_process_dim,
                                                                hidden_dim=pre_process_dim,
                                                                ext_dim=ext_dim,
                                                                pro_dim=pro_dim,
                                                                act_dim=act_dim,
                                                                inertial_dim=inertial_dim,
                                                                hist_num=historyNum,
                                                                batch_num=batchNum,
                                                                layer_num=layerNum,
                                                                device=device,
                                                                num_minibatch = num_mini_batches,
                                                                num_env=env.num_envs,
                                                                seq_num=seq_num), device=device)



pytorch_total_params = sum(p.numel() for p in Encoder.architecture.parameters())

print(pytorch_total_params)

actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['encoding']['policy_net'], nn.LeakyReLU, hidden_dim + pre_process_dim * 2 + pro_dim + ROA_ext_dim + act_dim, act_dim, actor=True),
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim,
                                                                           env.num_envs,
                                                                           1.0,
                                                                           NormalSampler(act_dim),
                                                                           cfg['seed']),
                         device)
critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['encoding']['value_net'], nn.LeakyReLU, hidden_dim + pre_process_dim * 2 + pro_dim + ROA_ext_dim + act_dim, 1, actor=False),
                           device)

saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/"+task_name,
                           save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp", task_path + "/RaiboController.hpp"])


ppo = PPO.PPO(actor=actor,
              critic=critic,
              encoder=Encoder,
              encoder_DR=None,
              num_envs=cfg['environment']['num_envs'],
              obs_shape=[env.num_obs],
              obs_lidar_shape=[env.num_obs_lidar],
              num_transitions_per_env=n_steps,
              num_learning_epochs=num_learning_epochs,
              gamma=0.995,
              lam=0.95,
              num_mini_batches=num_mini_batches,
              learning_rate=5e-5,
              device=device,
              log_dir=saver.data_dir,
              shuffle_batch=False,
              encoder_ROA=Encoder_ROA,
              encoder_lidar_robot=Encoder_lidar,
              lidar_preprocess=PointTransformer,
              estimator=Estimator,
              desired_kl=0.006,
              num_history_batch=historyNum,
              inertial_dim=inertial_dim,
              entropy_coef=entropy_coeff_,
              ROA_mode=ROA_mode
              )

iteration_number = 0

# map_processor = MapProcess(sample_width, sample_height, world_size)
# height_map = env.observe_height_map()
# map_processor.update_map(height_map)
# processed_height_map = map_processor.heightmap
# cost_map = map_processor.get_sdf_map()
# f = plt.figure(figsize=(30,20))
# ax = f.add_subplot(311)
# im = ax.imshow(cost_map[0], cmap='gray')
# ax2 = f.add_subplot(312)
# im2 = ax2.imshow(map_processor.original_heightmap[0], cmap='gray')
# plt.show()

# cost_map_flatten = cost_map.reshape(env.num_envs, -1)
# env.wrapper.update_cost_map(cost_map_flatten)
# env.wrapper.update_height_map_e()

Encoders = [Encoder, Encoder_ROA, Encoder_lidar]

def by_terminate(dones, encoders):
    if np.sum(dones) > 0:
        arg_dones = np.argwhere(dones).flatten()
        for encoder in encoders:
            encoder.architecture.reset_by_done(arg_dones)

if (args.wandb == True):
    wandb.init(group="jsh",project=task_name,name=name)

if mode == 'retrain':
    iteration_number = load_param(weight_path, env, actor, critic, ppo.optimizer, saver.data_dir)


iter_max = 40000

if(profile_mode):
    profiler = cProfile.Profile()
    profiler.enable()
    iter_max = 3

for update in range(iteration_number, iter_max):
    torch.cuda.empty_cache()
    start = time.time()
    env.reset(compute_path=False)
    for encoder in Encoders:
        encoder.architecture.reset()
    # env.wrapper.compute_Astar_path()
    end = time.time()
    print(end-start)
    # starts = env.get_obj_pos().astype(np.float64)
    # goals = env.get_target_pos().astype(np.float64)
    # paths = vectorized_planning(env=env, map_size=50, max_planning_time=2, starts=starts, goals=goals)
    # env.visualize_analytic_planner_path(paths[0])
    reward_ll_sum = 0
    done_sum = 0
    switch_sum = 0
    success_sum = 0
    average_dones = 0.
    success_batch = np.zeros(shape=(env.num_envs), dtype=bool)
    switch_batch = np.zeros(shape=(env.num_envs), dtype=bool)
#
    if update % cfg['environment']['eval_every_n'] == 0:
        print("Visualizing and evaluating the current policy")

        torch.save({
            'actor_architecture_state_dict': actor.architecture.state_dict(),
            'actor_distribution_state_dict': actor.distribution.state_dict(),
            'critic_architecture_state_dict': critic.architecture.state_dict(),
            'Encoder_state_dict' : Encoder.architecture.state_dict(),
            'Encoder_ROA_state_dict' : Encoder_ROA.architecture.state_dict(),
            'Encoder_lidar_state_dict': Encoder_lidar.architecture.state_dict(),
            'Lidar_preprocess': PointTransformer.state_dict(),
            'Inertial_estimator': Estimator.architecture.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict(),
        }, saver.data_dir+"/full_"+str(update)+'.pt')
        data_tags = env.get_step_data_tag()
        data_size = 0
        data_mean = np.zeros(shape=(len(data_tags), 1), dtype=np.double)
        data_square_sum = np.zeros(shape=(len(data_tags), 1), dtype=np.double)
        data_min = np.inf * np.ones(shape=(len(data_tags), 1), dtype=np.double)
        data_max = -np.inf * np.ones(shape=(len(data_tags), 1), dtype=np.double)

        env.turn_on_visualization()
        # env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')

        for step in range(n_steps):
            with torch.no_grad():
                obs = env.observe(False)
                obs_lidar_robot = env.observe_lidar()
                obs_lidar_robot_tc = torch.Tensor(obs_lidar_robot).to(device).reshape(1, env.num_envs, historyNum, -1, 3)
                lidar_global_features = PointTransformer.process_pointclouds(obs_lidar_robot_tc).reshape(env.num_envs, -1)

                latent_lidar_robot = Encoder_lidar.forward(lidar_global_features)

                # latent = Encoder.forward(torch.from_numpy(obs).to(device))
                latent = Encoder.forward(ppo.filter_for_encode_from_obs(obs).to(device))
                latent_ROA = Encoder_ROA.forward(ppo.get_obs_ROA(obs).to(device))
                actor_input = ppo.filter_for_actor(obs,
                                                   latent,
                                                   latent_lidar_robot,
                                                   lidar_global_features)

                actions, actions_log_prob = actor.sample(actor_input)
                reward, dones = env.step_visualize(actions)
                data_size = env.get_step_data(data_size, data_mean, data_square_sum, data_min, data_max)
                by_terminate(dones, Encoders)

        data_std = np.sqrt((data_square_sum - data_size * data_mean * data_mean) / (data_size - 1 + 1e-16))
        # env.stop_video_recording()
        # env.turn_off_visualization()
        env.reset(compute_path=False)
        for encoder in Encoders:
            encoder.architecture.reset()
        env.save_scaling(saver.data_dir, str(update))

    data_log = {}
    # actual training
    for step in range(n_steps):
        with torch.no_grad():
            obs = env.observe(update < 10000)
            obs_lidar_robot = env.observe_lidar()
            obs_lidar_robot_tc = torch.Tensor(obs_lidar_robot).to(device).reshape(1, env.num_envs, historyNum, -1, 3)
            lidar_global_features = PointTransformer.process_pointclouds(obs_lidar_robot_tc).reshape(env.num_envs, -1)

            latent_lidar_robot = Encoder_lidar.forward(lidar_global_features)

            success_batch = np.logical_or(success_batch, env.get_success_state())
            # print(success_batch)
            switch_batch = np.logical_or(switch_batch, env.get_intrinsic_switch())
            contact = env.get_contact()
            privileged_info = env.get_privileged_info()

            latent = Encoder.forward(ppo.filter_for_encode_from_obs(obs).to(device))
            latent = latent.detach().cpu().numpy()
            actor_input = ppo.filter_for_actor(obs,
                                               latent,
                                               latent_lidar_robot.detach().cpu().numpy(),
                                               lidar_global_features.detach().cpu().numpy())

            action = ppo.act(actor_input)
            reward, dones = env.step(action)

            ppo.step(value_obs=actor_input, obs=obs, obs_lidar=obs_lidar_robot, obs_lidar_object=None, rews=reward, dones=dones, contact=contact, privileged_info=privileged_info)
            done_sum = done_sum + np.sum(dones)
            reward_ll_sum = reward_ll_sum + np.sum(reward)
            by_terminate(dones, Encoders)
            # data_size = env.get_step_data(data_size, data_mean, data_square_sum, data_min, data_max)

    # data_std = np.sqrt((data_square_sum - data_size * data_mean * data_mean) / (data_size - 1 + 1e-16))
    # take st step to get value obs
    obs2 = env.observe(update < 10000)
    obs2_lidar_robot = env.observe_lidar()
    with torch.no_grad():
        # latent = Encoder.forward(torch.from_numpy(obs2).to(device))
        obs2_lidar_robot_tc = torch.Tensor(obs2_lidar_robot).to(device).reshape(1, env.num_envs, historyNum, -1, 3)
        lidar_global_features = PointTransformer.process_pointclouds(obs2_lidar_robot_tc).reshape(env.num_envs, -1)

        latent_lidar_robot = Encoder_lidar.forward(lidar_global_features)

        latent = Encoder.forward(ppo.filter_for_encode_from_obs(obs2).to(device))
        # latent = Encoder.forward(ppo.filter_for_encode_from_obs(obs2).to(device))
        latent = latent.detach().cpu().numpy()

        actor_input = ppo.filter_for_actor(obs2,
                                           latent,
                                           latent_lidar_robot.detach().cpu().numpy(),
                                           lidar_global_features.detach().cpu().numpy())

    ppo.update(actor_obs=actor_input, value_obs=actor_input, log_this_iteration=update % 10 == 0, update=update)


    ### For logging encoder (LSTM)
    # wandb.watch(Encoder.architecture)
    success_sum = np.sum(success_batch)
    switch_sum = np.sum(switch_batch)
    average_ll_performance = reward_ll_sum / total_steps
    average_dones = done_sum / total_steps
    actor.distribution.enforce_minimum_std((torch.ones(act_dim)*(0.6*math.exp(-0.0002*update) + 0.4)).to(device))
    # actor.distribution.enforce_minimum_std((torch.ones(1)*(0.06*math.exp(-0.0002*update) + 0.04)).to(device))
    actor.update()

    if (success_sum / env.num_envs) * 100 > 10 and (update % 50 == 0):
        env.curriculum_callback()
        # height_map = env.observe_height_map()
        # map_processor.update_map(height_map)
        # processed_height_map = map_processor.heightmap
        # cost_map = map_processor.get_sdf_map()
        # cost_map_flatten = cost_map.reshape(env.num_envs, -1)
        # env.wrapper.update_cost_map(cost_map_flatten)

    if update % 10 == 0:
        data_log['Training/average_reward'] = average_ll_performance
        data_log['Training/dones'] = average_dones
        data_log['Training/learning_rate'] = ppo.learning_rate
        data_log['PPO/value_function'] = ppo.mean_value_loss
        data_log['PPO/surrogate'] = ppo.mean_surrogate_loss
        data_log['PPO/mean_noise_std'] = ppo.mean_noise_std
        data_log['PPO/loss_ROA'] = ppo.loss_ROA
        data_log['PPO/lambda_loss_ROA'] = ppo.lambda_loss_ROA
        data_log['PPO/estimator_loss'] = ppo.estimator_loss
        data_log['PPO/entropy'] = ppo.entropy_mean

        for id, data_name in enumerate(data_tags):
            data_log[data_name + '/mean'] = data_mean[id]
            data_log[data_name + '/std'] = data_std[id]

    end = time.time()

    if(args.wandb == True):
        wandb.log(data_log)

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
    print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
    print('{:<40} {:>6}'.format("learning rate: ", '{:0.6f}'.format(ppo.learning_rate)))
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
    print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps / (end - start)
                                                                       * cfg['environment']['control_dt'])))
    print('{:<40} {:>6}'.format("intrinsic switch off rate (%): ", '{:0.3f}'.format((switch_sum / env.num_envs)*100)))
    print('{:<40} {:>6}'.format("success rate (%): ", '{:0.3f}'.format((success_sum / env.num_envs)*100)))
    print('----------------------------------------------------\n')

if(profile_mode):
    profiler.disable()
    profiler.dump_stats("profile.prof")