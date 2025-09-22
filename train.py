import random
import numpy as np
import torch
import argparse
import ma_utils
import algorithms.VIB_CMI_DDPG as VIB_CMI_DDPG
import algorithms.VIB_CMI_DDPG2 as VIB_CMI_DDPG2
import os
from torch.utils.tensorboard import SummaryWriter
import time
from config import GlobalConfig
from agent.action import Action
from dsmec_env.environment_manager import EnvironmentManager

cpu_num = 1
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

if __name__ == "__main__":
    global_config = GlobalConfig()
    train_config = global_config.train_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="VIB_CMI_DDPG")  # Policy name
    parser.add_argument("--seed", default=1, type=int)  # Sets Envs, PyTorch and Numpy seeds
    parser.add_argument("--batch_size", default=1024, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.95, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.01, type=float)  # Target network update rate
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--freq_test_mine", default=5e3, type=float)
    parser.add_argument("--gpu-no", default='1', type=str)  # GPU number, -1 means CPU
    parser.add_argument("--MI_update_freq", default=1, type=int)
    parser.add_argument("--max_adv_c", default=3.0, type=float)
    parser.add_argument("--min_adv_c", default=1.0, type=float)
    parser.add_argument("--discrete_action", action='store_true')
    parser.add_argument("--id", default="MITRA-VIB")
    args = parser.parse_args()

    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    file_name = "MITRA_VIB"
    writer = SummaryWriter(log_dir="./tensorboard/" + file_name + '/' + str(args.id))

    save_model_dir = "./model/"
    model_dir = save_model_dir + file_name
    if os.path.exists(model_dir) is False:
        os.makedirs(model_dir)

    print("---------------------------------------")
    print("Settings: %s" % (file_name))
    print("---------------------------------------")

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_no
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = EnvironmentManager(global_config)
    env.reset()

    n_agents = env.es_cs_set.user_device_num
    print("n_agent:", n_agents)

    obs_shape_n = []
    action_shape_n = []
    for i in range(n_agents):
        each_state = env.get_state_per_user_device(i)
        obs_shape_n.append(len(each_state.get_state_list()))
        action_shape_n.append(len(env.get_random_action(global_config).get_action_list()))
    print("obs_shape_n ", obs_shape_n)
    print("action_shape_n:", action_shape_n)

    if args.policy_name == "VIB_CMI_DDPG":
        policy = VIB_CMI_DDPG.MA_T_DDPG(n_agents, obs_shape_n, sum(obs_shape_n), action_shape_n, 1.0, device, 0.0, 0.0)
    else:
        policy = VIB_CMI_DDPG2.MA_T_DDPG(n_agents, obs_shape_n, sum(obs_shape_n), action_shape_n, 1.0, device, 0.0, 0.0)

    replay_buffer = ma_utils.ReplayBuffer(1e6)
    good_data_buffer = ma_utils.embedding_Buffer(1e3)
    bad_data_buffer = ma_utils.embedding_Buffer(1e3)

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_reward = global_config.interface_config.reward_config.init_reward
    episode_cost_list = []
    episode_cost_max_list = []
    done = True

    get_epoch_Mi = False
    Mi_list = []
    data_recorder = []
    replay_buffer_recorder = []
    moving_avg_reward_list = []
    avg_reward_list_for_best = []
    avg_cost_list_for_best = []
    embedding_recorder = []

    best_reward_start = -1
    reward_list = []
    eposide_reward_list = []
    compare_list = []
    compare_range_count = 0
    recorder_reward = 0
    best_reward = -100000000000
    best_reward_for_save_model = -100000000000
    current_policy_performance = -1000
    episode_timesteps = train_config.step_num
    start_time = time.time()
    
    if train_config.load_model_name is not None:
        policy.load(train_config.load_model_name, train_config.load_model_path)
    if train_config.if_load_buffer:
        if train_config.load_buffer_name == 'good':
            good_data_buffer.load_buffer_from_file(train_config.load_model_name, train_config.load_model_path,buffer_name='good')
        elif train_config.load_buffer_name == 'bad':
            bad_data_buffer.load_buffer_from_file(train_config.load_model_name, train_config.load_model_path,buffer_name='bad')
        replay_buffer.load_buffer_from_file(train_config.load_model_name, train_config.load_model_path,buffer_name='general')

    t1 = time.time()

    while total_timesteps < train_config.total_timesteps:
        if episode_timesteps == train_config.step_num or done: 
            print("episode_num:", episode_num)
            print("episode_timesteps:", episode_timesteps)
            for d in replay_buffer_recorder:
                replay_buffer.add(d, episode_reward)

            if total_timesteps != 0:
                # walker 50
                if len(good_data_buffer.pos_storage_reward) != 0:
                    if len(moving_avg_reward_list) >= 10:
                        move_avg = np.mean(moving_avg_reward_list[-1000:])
                        current_policy_performance = move_avg

                    lowest_reward = good_data_buffer.rank_storage(-100000)
                    mean_reward = np.mean(good_data_buffer.pos_storage_reward)

                    if len(moving_avg_reward_list) >= 10:
                        # writer.add_scalar("data/tp_lowest_reward", lowest_reward, total_timesteps)
                        if lowest_reward < move_avg:
                            lowest_reward = move_avg
                else:
                    lowest_reward = global_config.interface_config.reward_config.lowest_reward
                    mean_reward = global_config.interface_config.reward_config.mean_reward

                if lowest_reward < episode_reward:
                    Obs_list = []
                    State_list = []
                    Action_list = []
                    for d in data_recorder:
                        obs = d[0]
                        state = d[1]
                        Action_list.append(d[2])
                        Obs_list.append(obs)
                        State_list.append(state)
                    for index in range(len(Action_list)):
                        good_data_buffer.add_pos((Action_list[index], Obs_list[index], State_list[index]), episode_reward)
                else:
                    Obs_list = []
                    State_list = []
                    Action_list = []
                    for d in data_recorder:
                        obs = d[0]
                        state = d[1]
                        Action_list.append(d[2])
                        Obs_list.append(obs)
                        State_list.append(state)
                    for index in range(len(Action_list)):
                        bad_data_buffer.add_pos((Action_list[index], Obs_list[index], State_list[index]),episode_reward)

                writer.add_scalar("data/avg_cost", episode_cost_list[-1], episode_num)
                writer.add_scalar("data/success_rate", (episode_timesteps - 1) / (train_config.step_num - 1),episode_num)
                avg_reward_list_for_best.append(episode_reward)
                avg_cost_list_for_best.append(episode_cost_list[-1])

                if episode_num % 100000 == 0: 
                    policy.save("model_" + str(episode_num) + 
                                "cost_" + str(np.around(episode_cost_list[-1], 5)) + 
                                "reward_" + str(np.around(episode_reward, 3)), 
                                save_model_dir + file_name)
                    if train_config.if_save_buffer:
                        good_data_buffer.save_buffer_to_file("model_" + str(episode_num) + "cost_" + str(np.around(episode_cost_list[-1], 5)) + "reward_" + str(np.around(episode_reward, 3)), save_model_dir + file_name, buffer_name='good')
                        bad_data_buffer.save_buffer_to_file("model_" + str(episode_num) + "cost_" + str(np.around(episode_cost_list[-1], 5)) + "reward_" + str(np.around(episode_reward, 3)), save_model_dir + file_name, buffer_name='bad')
                        replay_buffer.save_buffer_to_file("model_" + str(episode_num) + "cost_" + str(np.around(episode_cost_list[-1], 5)) + "reward_" + str(np.around(episode_reward, 3)), save_model_dir + file_name, buffer_name='general')


                if episode_num % 1000 == 0:
                    print('Total T:', total_timesteps, 'Episode Num:', episode_num, 'Episode T:', episode_timesteps,
                          'Reward:', np.mean(moving_avg_reward_list[-1000:]) / 3.0, " time cost:", time.time() - t1)
                    t1 = time.time()

            if (total_timesteps >= 1024 and total_timesteps % 100 == 0) or train_config.load_model_name is not None:
                sp_actor_loss_list = []
                process_Q_list = []
                process_min_MI_list = []
                process_max_MI_list = []
                process_min_MI_loss_list = []
                process_max_MI_loss_list = []
                Q_grads_list = []
                MI_grads_list = []
                MI_upper_bound_list_list = []
                MI_lower_bound_list_list = []
                training_reward_list_list = []
                training_reward_Q_list_list = []
                training_r_MITRAVIB_list_list = []

                for i in range(1):
                    update_signal = False
                    if len(good_data_buffer.pos_storage) < 500:
                        process_Q = policy.train(replay_buffer, 1, args.batch_size, args.discount, args.tau)

                        process_min_MI = 0
                        process_min_MI_loss = 0
                        min_mi = 0.0
                        min_mi_loss = 0.0
                        process_max_MI = 0
                        pr_sp_loss = 0.0
                        Q_grads = 0.0
                        MI_grads = 0.0
                        process_max_MI_loss = 0.0
                    else:
                        if total_timesteps % (args.MI_update_freq * 100) == 0:
                            update_signal = True
                            if args.min_adv_c > 0.0 and len(bad_data_buffer.pos_storage) > 0:
                                process_min_MI_loss = policy.train_club(bad_data_buffer, 1, batch_size=args.batch_size)
                                #process_min_MI_loss = policy.train_cclub(bad_data_buffer, 1, batch_size=args.batch_size) if (args.min_adv_c > 0.0 and len(bad_data_buffer.pos_storage) > 0) else 0.0
                            else:
                                process_min_MI_loss = 0.0

                            if args.max_adv_c > 0.0 and len(good_data_buffer.pos_storage) > 0:
                                process_max_MI_loss, _ = policy.train_mine(good_data_buffer, 1,batch_size=args.batch_size)
                                #process_max_MI_loss = policy.train_cmine(good_data_buffer, 1, batch_size=args.batch_size) if (args.max_adv_c > 0.0 and len(good_data_buffer.pos_storage) > 0) else 0.0
                            else:
                                process_max_MI_loss = 0.0

                        else:
                            process_min_MI_loss = 0.0
                            process_max_MI_loss = 0.0
                        process_Q, process_min_MI, process_max_MI, Q_grads, MI_grads, MI_upper_bound_list, MI_lower_bound_list, training_reward_list, training_reward_Q_list, training_r_MITRAVIB_list = policy.train_actor_with_mine(
                            replay_buffer, 1, args.batch_size, args.discount, args.tau, max_mi_c=0.0, min_mi_c=0.0,
                            min_adv_c=args.min_adv_c, max_adv_c=args.max_adv_c, total_timesteps=total_timesteps,
                            update_signal=update_signal)

                        MI_upper_bound_list_list.append(MI_upper_bound_list)
                        MI_lower_bound_list_list.append(MI_lower_bound_list)
                        training_reward_list_list.append(training_reward_list)
                        training_reward_Q_list_list.append(training_reward_Q_list)
                        training_r_MITRAVIB_list_list.append(training_r_MITRAVIB_list)

                        upper_bound = np.mean(MI_upper_bound_list_list)
                        lower_bound = np.mean(MI_lower_bound_list_list)
                        training_reward = np.mean(training_reward_list_list)
                        training_reward_Q = np.mean(training_reward_Q_list_list)
                        training_r_MITRAVIB = np.mean(training_r_MITRAVIB_list_list)

                    process_max_MI_list.append(process_max_MI)
                    process_Q_list.append(process_Q)
                    Q_grads_list.append(Q_grads)
                    MI_grads_list.append(MI_grads)
                    process_max_MI_loss_list.append(process_max_MI_loss)
                    process_min_MI_list.append(process_min_MI)
                    process_min_MI_loss_list.append(process_min_MI_loss)

            env.reset()
            obs = []
            for user_device_id in range(len(env.es_cs_set.all_user_device_list)):
                each_state = env.get_state_per_user_device(user_device_id)
                state_array = each_state.get_normalized_state_array()
                obs.append(state_array)
            state = np.concatenate(obs, -1)
            moving_avg_reward_list.append(episode_reward)

            done = False 

            explr_pct_remaining = max(0, 25000 - episode_num) / 25000
            policy.scale_noise(0.3 * explr_pct_remaining)
            policy.reset_noise()
            episode_reward = 0
            episode_cost_list = []
            episode_cost_max_list = []
            reward_list = []
            eposide_reward_list = []
            episode_timesteps = 0
            episode_num += 1
            data_recorder = []
            replay_buffer_recorder = []
            best_reward_start = -1
            best_reward = -1000000
            Mi_list = []

            if total_timesteps % (4e5 - 1) == 0:
                policy.save(total_timesteps, save_model_dir + file_name)

        # each step begin
        env.create_task_per_step()
        print("episode_timesteps:", episode_timesteps)
        state_class_list = []
        obs = []
        for user_device_id in range(len(env.es_cs_set.all_user_device_list)):
            each_state = env.get_state_per_user_device(user_device_id)
            state_class_list.append(each_state)
            state_list = each_state.get_state_list()
            state_array = each_state.get_normalized_state_array()
            obs.append(state_array)
        state = np.concatenate(obs, -1)

        # Select action randomly or according to policy
        scaled_a_list = []
        action_class_list = []

        for i in range(n_agents):
            a = policy.select_action(obs[i], i)
            scaled_a = np.multiply(a, 1.0)
            scaled_a = np.clip(scaled_a, -0.9999, 0.9999)
            meaningful_scaled_a = scaled_a
            meaningful_scaled_a[0] = int(np.floor((scaled_a[0] + 1) * env.es_cs_set.es_cs_num / 2))
            meaningful_scaled_a[1] = (scaled_a[1] + 1) / 2
            print("meaningful_scaled_a:", meaningful_scaled_a)
            action_class = Action(meaningful_scaled_a, global_config)
            scaled_a_list.append(scaled_a)
            action_class_list.append(action_class)
        next_state_class_list, cost_array, reward_array, cost_array_max, done_list, _ = env.step(state_class_list,
                                                                                                 action_class_list,
                                                                                                 episode_timesteps)
        reward = reward_array[0]

        next_obs = []
        for user_device_id in range(len(env.es_cs_set.all_user_device_list)):
            each_state = env.get_state_per_user_device(user_device_id)
            state_array = each_state.get_normalized_state_array()
            next_obs.append(state_array)
        next_state = np.concatenate(next_obs, -1)

        done = any(done_list)
        terminal = (episode_timesteps + 1 >= train_config.step_num)
        done = done or terminal
        done_bool = float(done or terminal)

        episode_reward += reward
        episode_cost_list.append(cost_array[0])
        episode_cost_max_list.append(cost_array_max[0])
        eposide_reward_list.append(reward)

        # Store data in replay buffer , replay_buffer_recorder
        replay_buffer_recorder.append(
            (obs, state, next_state, next_obs, np.concatenate(scaled_a_list, -1), reward, done))
        data_recorder.append([obs, state, scaled_a_list])

        obs = next_obs
        state = next_state

        episode_timesteps += 1
        print("episode_timesteps:", episode_timesteps)
        total_timesteps += 1
        timesteps_since_eval += 1

    print("total time ", time.time() - start_time)
    policy.save(total_timesteps, save_model_dir + file_name)

    writer.close()
