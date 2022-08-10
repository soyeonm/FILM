import argparse
import math
import torch
import copy

def get_args():
    parser = argparse.ArgumentParser(description='Active-Neural-SLAM')

    parser.add_argument('--aithor', type=int, default=0)
    parser.add_argument('--alfred', type=int, default=0)
    ## General Arguments
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--auto_gpu_config', type=int, default=1)
    parser.add_argument('--total_num_scenes', type=str, default="auto")
    parser.add_argument('-n', '--num_processes', type=int, default=4,
                        help="""how many training processes to use (default:4)
                                Overridden when auto_gpu_config=1
                                and training on gpus """)
    parser.add_argument('--num_processes_per_gpu', type=int, default=11)
    parser.add_argument('--num_processes_on_first_gpu', type=int, default=0)
    parser.add_argument('--num_training_frames', type=int, default=10000000,
                        help='total number of training frames (default: 1000000)')
    parser.add_argument('--num_episodes', type=int, default=1000000,
                        help='number of training episodes (default: 1000000)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--eval', type=int, default=0,
                        help='0: Train, 1: Evaluate (default: 0)')

    # Logging, loading models, visualization
    parser.add_argument('--log_interval', type=int, default=10,
                        help="""log interval, one log per n updates
                                (default: 10) """)
    parser.add_argument('--save_interval', type=int, default=1,
                        help="""save interval""")
    parser.add_argument('--exp_name', type=str, default="exp1",
                        help='experiment name (default: exp1)')
    parser.add_argument('--save_periodic', type=int, default=500000,
                        help='Model save frequency in number of updates')
    parser.add_argument('--load', type=str, default="0",
                        help="""model path to load,
                                0 to not reload (default: 0)""")
    parser.add_argument('-v', '--visualize', type=int, default=0,
                        help='1:Render the frame (default: 0)')
    parser.add_argument('--print_images', type=int, default=0,
                        help='1: save visualization as images')

    # Environment, dataset and episode specifications
    parser.add_argument('-efw', '--env_frame_width', type=int, default=640,
                        help='Frame width (default:84)')
    parser.add_argument('-efh', '--env_frame_height', type=int, default=480,
                        help='Frame height (default:84)')
    parser.add_argument('-fw', '--frame_width', type=int, default=160,
                        help='Frame width (default:84)')
    parser.add_argument('-fh', '--frame_height', type=int, default=120,
                        help='Frame height (default:84)')
    parser.add_argument('-el', '--max_episode_length', type=int, default=500,
                        help="""Maximum episode length""")
    parser.add_argument("--sim_gpu_id", type=int, default=0,
                        help="gpu id on which scenes are loaded")
    parser.add_argument("--sem_gpu_id", type=int, default=-1,
                        help="gpu id for semantic model, -1: same as sim gpu, -2: cpu")
    parser.add_argument("--task_config", type=str,
                        default="tasks/objectnav_gibson.yaml",
                        help="path to config yaml containing task information")
    parser.add_argument("--split", type=str, default="train",
                        help="dataset split (train | val | val_mini) ")
    parser.add_argument('-na', '--noisy_actions', type=int, default=0)
    parser.add_argument('-no', '--noisy_odometry', type=int, default=0)
    parser.add_argument('--camera_height', type=float, default=1.55,
                        help="agent camera height in metres")
    parser.add_argument('--hfov', type=float, default=60.0,
                        help="horizontal field of view in degrees")
    parser.add_argument('--randomize_env_every', type=int, default=0,
                        help="randomize scene in a thread every k episodes")

    ## Model Hyperparameters
    parser.add_argument('--agent', type=str, default="sem_exp")
    parser.add_argument('--global_lr', type=float, default=2.5e-5,
                        help='global learning rate (default: 2.5e-5)')
    parser.add_argument('--global_hidden_size', type=int, default=256,
                        help='local_hidden_size')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RL Optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RL Optimizer alpha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use_gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy_coef', type=float, default=0.001,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value_loss_coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--num_global_steps', type=int, default=20,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--ppo_epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num_mini_batch', type=str, default="auto",
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip_param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--use_recurrent_global', type=int, default=0,
                        help='use a recurrent global policy')
    parser.add_argument('--num_local_steps', type=int, default=25,
                        help="""Number of steps the local policy
                                between each global step""")

    # Mapping
    parser.add_argument('--global_downscaling', type=int, default=1)
    parser.add_argument('--vision_range', type=int, default=100)
    parser.add_argument('--map_resolution', type=int, default=5)
    parser.add_argument('--du_scale', type=int, default=1)
    parser.add_argument('--map_size_cm', type=int, default=1200)
    parser.add_argument('--cat_pred_threshold', type=float, default=5.0)
    parser.add_argument('--map_pred_threshold', type=float, default=1.0)
    parser.add_argument('--exp_pred_threshold', type=float, default=1.0)
    parser.add_argument('--collision_threshold', type=float, default=0.20)

    # New args
    parser.add_argument('--print_time', type=int, default=0)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--turn_angle', type=float, default=30)
    parser.add_argument('--min_depth', type=float, default=0.5)
    parser.add_argument('--max_depth', type=float, default=5.0)

    # for semantic prediction
    parser.add_argument('--sem_pred_prob_thr', type=float, default=0.9)
    parser.add_argument('--num_sem_categories', type=float, default=16)
    parser.add_argument('--num_episodes_per_scene', type=int, default=200)
    parser.add_argument('--min_d', type=float, default=1.5,
                        help="min distance to goal during training in meters")
    parser.add_argument('--max_d', type=float, default=100.0,
                        help="max distance to goal during training in meters")
    parser.add_argument('--success_dist', type=float, default=1.0)
    parser.add_argument('--floor_thr', type=int, default=50,
                        help="floor_thr in cm")
    parser.add_argument('--version', type=str, default="v1.1")

    ###############
    ## Arguments from ALFRED (no duplicate with OGN arguments)
    ###############
    # settings
    parser.add_argument('--splits', type=str, default="alfred_data_small/splits/oct21.json")
    parser.add_argument('--data', type=str, default="alfred_data_small/json_2.1.0")
    parser.add_argument('--eval_split', type=str, default='valid_seen', choices=['train', 'valid_seen', 'valid_unseen', 'tests_unseen', 'tests_seen'])
    parser.add_argument('--model_path', type=str, default="model.pth")
    parser.add_argument('--model', type=str, default='models.model.seq2seq_im_mask')
    parser.add_argument('--preprocess', dest='preprocess', action='store_true')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true')
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--num_threads', type=int, default=1)

    # eval params
    parser.add_argument('--max_steps', type=int, default=1000, help='max steps before episode termination')
    parser.add_argument('--max_fails', type=int, default=10, help='max API execution failures before episode termination')

    # eval settings
    parser.add_argument('--subgoals', type=str, help="subgoals to evaluate independently, eg:all or GotoLocation,PickupObject...", default="")
    parser.add_argument('--smooth_nav', dest='smooth_nav', action='store_true', help='smooth nav actions (might be required based on training data)')
    parser.add_argument('--skip_model_unroll_with_expert', action='store_true', help='forward model with expert actions')
    parser.add_argument('--no_teacher_force_unroll_with_expert', action='store_true', help='no teacher forcing with expert')

    # debug
    parser.add_argument('--debug_alfred', dest='debug_alfred', action='store_true')
    parser.add_argument('--fast_epoch', dest='fast_epoch', action='store_true')
    
    parser.add_argument('--gts', dest='ground_truth_segmentation', action='store_true')
    parser.add_argument('--learned_depth', dest='use_learned_depth', action='store_true')
    parser.add_argument('--valts_depth', dest='valts_depth', action='store_true')
    parser.add_argument('--depth_checkpoint_path', type=str, default="real_test_ogn/lr1e-5/model-48000-best_d1_0.88987") 

    
    parser.add_argument('--focal', type=float) #one of 518 or 259
    parser.add_argument('--depth_gpu', type=int)

    parser.add_argument('--ignore_cats', dest='ignore_categories', action='store_true')
    parser.add_argument('--wait_time', type=float, default=0)
    
    parser.add_argument('--scene', type=str, default='vase')
    
    parser.add_argument('--look_up_down', type=str, default="Down") #'Up' or 'Down'
    parser.add_argument('--view_angle', type=float, default=0) #look up or down angle
    parser.add_argument('--agent_height_change', type=float, default=0) #decrease or increase agent height
    
    parser.add_argument('--top_thirty', dest='top_thirty', action='store_true')
    parser.add_argument('--no_common_objs', dest='no_common_objs', action='store_true')
    parser.add_argument('--which_gpu', type=str)
    parser.add_argument('--dgx', dest='dgx', action='store_true')
    parser.add_argument('--matrix', dest='matrix', action='store_true')
    
    parser.add_argument('--from_idx', type=int, default=0)
    parser.add_argument('--to_idx', type=int, default=821)
    
    parser.add_argument('--x_display', type=str, default='7')
    parser.add_argument('--docker', dest='docker', action='store_true')
    
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--test_seen', dest='test_seen', action='store_true')
    
    parser.add_argument('--leaderboard', dest='leaderboard', action='store_true')
    parser.add_argument('--debug_max_steps', dest='debug_max_steps', action='store_true')
    parser.add_argument('--tomato', dest='tomato', action='store_true')
    
    parser.add_argument('--no_look_down', dest='no_look_down', action='store_true')
    parser.add_argument('--apple', dest='apple', action='store_true')
    
    parser.add_argument('--debug_local', dest='debug_local', action='store_true')
    parser.add_argument('--save_pictures', dest='save_pictures', action='store_true')
    parser.add_argument('--save_training_pictures', dest='save_training_pictures', action='store_true')
    
    parser.add_argument('--exclude_list',  type=str, default="")
    parser.add_argument('--load_cuda_goals',  dest='load_cuda_goals', action='store_true')
    parser.add_argument('--save_cuda_goals',  dest='save_cuda_goals', action='store_true')

    parser.add_argument('--collision_map_delete',  action="store_true")
    parser.add_argument('--check_before_sidestep',  action="store_true")
    parser.add_argument('--check_before_movebehind',  action="store_true")
    parser.add_argument('--dilation_agent_body',  action="store_true")
    parser.add_argument('--dilation_five',  action="store_true")
    parser.add_argument('--disk_dilation',  action="store_true")
    parser.add_argument('--selem_square',  action="store_true")
    parser.add_argument('--ori_dist',  action="store_true")

    parser.add_argument('--broken_new',  action="store_true")
    parser.add_argument('--no_traversible_check',  action="store_true")

    parser.add_argument('--save_this_outputted',  action="store_true")
    parser.add_argument('--save_10_failures',  action="store_true")

    parser.add_argument('--nonsliced',  action="store_true")

    parser.add_argument('--visibility',  action="store_true")
    parser.add_argument('--previous_target',  action="store_true")
    parser.add_argument('--temp_largest_area',  action="store_true")

    parser.add_argument('--use_temp',  action="store_true")
    parser.add_argument('--use_sem_seg',  action="store_true")

    parser.add_argument('--sem_seg_threshold_large',  type=float, default=0.8)
    parser.add_argument('--sem_seg_threshold_small',  type=float, default=0.8)
    parser.add_argument('--sem_seg_gpu',  type=int, default=0)
    parser.add_argument('--alfworld_mrcnn', action="store_true")

    parser.add_argument('--alfworld_both', action="store_true")

    parser.add_argument('--flip_rgb', dest='flip_rgb', action='store_true')
    parser.add_argument('--alfworld_old_mrcnn', action='store_true')


    parser.add_argument('--map_mask_prop',type=float,default=1)

    parser.add_argument('--delete_from_map_after_move_until_visible',action='store_true')
    
    parser.add_argument('--with_mask_above_05',action='store_true')

    parser.add_argument('--set_dn',type=str, default="")
    parser.add_argument('--no_caution_pointers',action='store_true')

    parser.add_argument('--no_pickup_update',action='store_true')
    parser.add_argument('--obstacle_selem',type=int, default=3)

    parser.add_argument('--valts_trustworthy',action='store_true')
    parser.add_argument('--valts_trustworthy_prop',type=float, default=1.0)
    parser.add_argument('--valts_trustworthy_obj_prop',type=float, default=1.0)
    parser.add_argument('--valts_trustworthy_obj_prop0',type=float, default=1.0)

    parser.add_argument('--no_straight_obs',action='store_true')

    parser.add_argument('--depth_model_old',action='store_true')
    parser.add_argument('--depth_model_45_only',action='store_true')

    parser.add_argument('--separate_depth_for_straight',action='store_true')
    parser.add_argument('--no_depth_mask_interaction',action='store_true')
    parser.add_argument('--learned_visibility',action='store_true')
    parser.add_argument('--learned_visibility_no_mask',action='store_true')
    parser.add_argument('--appended',action='store_true')


    parser.add_argument('--approx_last_action_success',action='store_true')
    parser.add_argument('--approx_error_message',action='store_true')
    parser.add_argument('--approx_horizon',action='store_true')
    parser.add_argument('--approx_pose',action='store_true')

    parser.add_argument('--use_sem_policy',action='store_true')
    parser.add_argument('--explore_prob',type=float, default=0.0)

    parser.add_argument('--step_size',type=int, default=5)
    parser.add_argument('--side_step_step_size',type=int, default=5)

    parser.add_argument('--no_look_down_90',action='store_true')
    parser.add_argument('--sidestep_width',type=int, default=4)
    parser.add_argument('--behind_step_width',type=int, default=5)
    
    parser.add_argument('--look_no_repeat',action='store_true')
    parser.add_argument('--stop_cond',type=float, default=0.55)

    parser.add_argument('--stricter_visibility',type=float, default=2.0)
    parser.add_argument('--bathtubbasin_visibe_dist',type=float, default=1.5)

    parser.add_argument('--dil_coeff',type=float, default=1.0)

    parser.add_argument('--valid_learned_lang',action='store_true')

    parser.add_argument('--no_delete_lamp',action='store_true')
    parser.add_argument('--no_block',action='store_true')
    parser.add_argument('--no_rotate_sidestep',action='store_true')
    parser.add_argument('--no_opp_sidestep',action='store_true')
    parser.add_argument('--no_pickup',action='store_true')

    # parse arguments
    args = parser.parse_args()

    #Add arguments that just have to be added anyways automatically 
    args.agent = 'sem_exp'
    args.split = 'val'
    args.version = 'v1.1'
    args.eval = 1
    args.alfred = 1   
    #args.ground_truth_segmentation = True
    args.ignore_categories = True
    args.no_common_objs = True
    args.matrix = True
    args.leaderboard = True
    args.print_images = 1
    args.focal = 259
    args.broken_new = True
    args.no_traversible_check = True

    args.visibility = True
    args.flip_rgb = True 

    args.approx_pose = True
    args.approx_horizon =True
    args.approx_error_message = True
    args.approx_last_action_success = True

    args.no_look_down_90 = True
    args.look_no_repeat = True
    args.sidestep_width = 0
    args.side_step_step_size = 3
    args.check_before_sidestep = True
    args.stricter_visibility = 1.5

    args.obstacle_selem = 4
    args.selem_square = True
    args.delete_from_map_after_move_until_visible  = True

    if args.use_learned_depth:
        args.map_pred_threshold = 65
        args.no_pickup_update = True
        args.cat_pred_threshold = 10
        args.valts_depth = True
        args.valts_trustworthy = True
        args.valts_trustworthy_prop = 0.9
        args.valts_trustworthy_obj_prop0 = 1.0
        args.valts_trustworthy_obj_prop = 1.0
        args.learned_visibility = True
        args.learned_visibility_no_mask = True
        args.separate_depth_for_straight = True

    if args.use_sem_seg:
        args.with_mask_above_05 = True
        args.sem_seg_threshold_small = 0.8
        args.sem_seg_threshold_large = 0.8
        args.alfworld_mrcnn = True
        args.alfworld_both = True

    if args.use_sem_policy:
        args.explore_prob = 0.0
    else:
        args.explore_prob = 1.0



    args.no_straight_obs = True
    num_proc = copy.deepcopy(args.num_processes)

    if args.eval_split in ['tests_seen', 'tests_unseen']:
        args.test =True
        if args.eval_split == 'tests_seen':
            args.test_seen = True

    if args.eval_split in ['valid_seen', 'valid_unseen']:
        args.valid_learned_lang = True
    
    if args.alfred == 1:
        args.env_frame_width = 300
        args.env_frame_height = 300
        
        args.frame_width = 150
        args.frame_height = 150
        
    args.num_sem_categories = 1 + 1 + 1+ 5*args.num_processes

    if args.use_sem_policy:
        args.num_sem_categories = args.num_sem_categories + 23

    args.cuda = not args.no_cuda and torch.cuda.is_available()


    if args.cuda:
        if args.auto_gpu_config:
            num_gpus = torch.cuda.device_count()
            if args.total_num_scenes != "auto":
                args.total_num_scenes = int(args.total_num_scenes)
            elif "objectnav_gibson" in args.task_config and \
                    "train" in args.split:
                args.total_num_scenes = 25
            elif "objectnav_gibson" in args.task_config and \
                    "val" in args.split:
                args.total_num_scenes = 5
            else:
                assert False, "Unknown task config, please specify" + \
                        " total_num_scenes"

            # Automatically configure number of training threads based on
            # number of GPUs available and GPU memory size
            total_num_scenes = args.total_num_scenes
            gpu_memory = 1000
            for i in range(num_gpus):
                gpu_memory = min(gpu_memory,
                    torch.cuda.get_device_properties(i).total_memory \
                            /1024/1024/1024)
                if i==0:
                    assert torch.cuda.get_device_properties(i).total_memory \
                            /1024/1024/1024 > 10.6, "Insufficient GPU memory"

            num_processes_per_gpu = int(gpu_memory/2.6)
            num_processes_on_first_gpu = int((gpu_memory - 10.6)/2.6)

            if num_gpus == 1:
                args.num_processes_on_first_gpu = num_processes_on_first_gpu
                args.num_processes_per_gpu = 0
                args.num_processes = num_processes_on_first_gpu
            else:
                num_processes_on_first_gpu = 0
                num_threads = num_processes_per_gpu * (num_gpus - 1) \
                                + num_processes_on_first_gpu
                num_threads = min(num_threads, args.total_num_scenes)

                args.num_processes_per_gpu = min(num_processes_per_gpu,
                                        math.ceil(num_threads//(num_gpus-1)))

                args.num_processes_on_first_gpu = max(0,
                    num_threads - args.num_processes_per_gpu*(num_gpus - 1))

                args.num_processes = num_threads

            args.sim_gpu_id = 1

            print("Auto GPU config:")
            print("Number of processes: {}".format(args.num_processes))
            print("Number of processes on GPU 0: {}".format(
                                      args.num_processes_on_first_gpu))
            print("Number of processes per GPU: {}".format(
                                      args.num_processes_per_gpu))
    else:
        args.sem_gpu_id = -2

    if args.num_mini_batch == "auto":
        args.num_mini_batch = max(args.num_processes // 2, 1)
    else:
        args.num_mini_batch = int(args.num_mini_batch)

    args.num_processes = num_proc
    
    return args
