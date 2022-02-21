import os, sys
import matplotlib

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["OMP_NUM_THREADS"] = "1"

if sys.platform == 'darwin':
    matplotlib.use("tkagg")

import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
import math
import time

import cv2
from torchvision import transforms
from PIL import Image
import skimage.morphology

from importlib import import_module
from collections import defaultdict
import json, pickle
from datetime import datetime

from arguments import get_args
from envs import make_vec_envs
import envs.utils.pose as pu

from models.sem_mapping import Semantic_Mapping
from models.instructions_processed_LP.ALFRED_task_helper import determine_consecutive_interx
import alfred_utils.gen.constants as constants
from models.semantic_policy.sem_map_model import UNetMulti


def into_grid(ori_grid, grid_size):
    one_cell_size = math.ceil(240/grid_size)
    return_grid = torch.zeros(grid_size,grid_size)
    for i in range(grid_size):
        for j in range(grid_size):
            if torch.sum(ori_grid[one_cell_size *i: one_cell_size*(i+1),  one_cell_size *j: one_cell_size*(j+1)].bool().float())>0:
                return_grid[i,j] = 1
    return return_grid


def main():
    args = get_args()
    dn = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    args.dn =dn
    if args.set_dn!="":
        args.dn = args.set_dn
        dn =  args.set_dn
    print("dn is ", dn)

    if not os.path.exists("results/logs"):
        os.makedirs("results/logs")
    if not os.path.exists("results/leaderboard"):
        os.makedirs("results/leaderboard")
    if not os.path.exists("results/successes"):
        os.makedirs("results/successes")
    if not os.path.exists("results/fails"):
        os.makedirs("results/fails")
    if not os.path.exists("results/analyze_recs"):
        os.makedirs("results/analyze_recs")        

    completed_episodes = []
    
    skip_indices ={}
    if args.exclude_list!="":
        if args.exclude_list[-2:] == ".p": 
            skip_indices = pickle.load(open(args.exclude_list, 'rb'))
            skip_indices = {int(s):1 for s in skip_indices}
        else:
            skip_indices = [a for a in args.exclude_list.split(',')]
            skip_indices = {int(s):1 for s in skip_indices}
    args.skip_indices = skip_indices
    actseqs = []
    all_completed = [False] * args.num_processes
    successes = []; failures = []  
    analyze_recs = []
    traj_number =[0] * args.num_processes

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    large_objects2idx = {obj:i for i, obj in enumerate(constants.map_save_large_objects)}
    all_objects2idx = {o:i for i, o in enumerate(constants.map_all_objects)}
    softmax = nn.Softmax(dim=1)


    # Logging and loss variables
    num_scenes = args.num_processes
    num_episodes = [0] * args.num_processes
    for e in range(args.from_idx, args.to_idx):  
        remainder = e % args.num_processes
        num_episodes[remainder] +=1
    
    device = args.device = torch.device("cuda:" + args.which_gpu if args.cuda else "cpu")
    if args.use_sem_policy:
        Unet_model = UNetMulti((240,240), num_sem_categories=24).to(device=device)
        sd = torch.load('models/semantic_policy/best_model_multi.pt', map_location = device)
        Unet_model.load_state_dict(sd)
        del sd

    finished = np.zeros((args.num_processes))
    wait_env = np.zeros((args.num_processes))


    # Starting environments
    torch.set_num_threads(1)
    envs = make_vec_envs(args)
    fails = [0] * num_scenes
    prev_cns = [None] * num_scenes
    
    
    obs, infos, actions_dicts = envs.load_initial_scene()
    second_objects = []; list_of_actions_s = []; task_types = []; whether_sliced_s= []
    for e in range(args.num_processes):
        second_object = actions_dicts[e]['second_object']
        list_of_actions = actions_dicts[e]['list_of_actions']
        task_type = actions_dicts[e]['task_type']
        sliced = actions_dicts[e]['sliced']
        second_objects.append(second_object); list_of_actions_s.append(list_of_actions); task_types.append(task_type); whether_sliced_s.append(sliced)

    task_finish = [False] * args.num_processes
    first_steps = [True] * args.num_processes
    num_steps_so_far = [0] * args.num_processes
    load_goal_pointers = [0] * args.num_processes
    list_of_actions_pointer_s = [0] * args.num_processes
    goal_spotted_s = [False] * args.num_processes  
    list_of_actions_pointer_s = [0] * args.num_processes
    goal_logs = [[] for i in range(args.num_processes)]
    goal_cat_before_second_objects = [None] * args.num_processes
  
    do_not_update_cat_s = [None] * args.num_processes
    wheres_delete_s = [np.zeros((240,240))] * args.num_processes
    
    args.num_sem_categories = 1 + 1 + 1 + 5 * args.num_processes
    if args.use_sem_policy:
        args.num_sem_categories = args.num_sem_categories + 23
    obs = torch.tensor(obs).to(device)
    
    torch.set_grad_enabled(False)

    # Initialize map variables
    ### Full map consists of multiple channels containing the following:
    ### 1. Obstacle Map
    ### 2. Exploread Area
    ### 3. Current Agent Location
    ### 4. Past Agent Locations
    ### 5,6,7,.. : Semantic Categories
    nc = args.num_sem_categories + 4 # num channels

    # Calculating full and local map sizes
    map_size = args.map_size_cm // args.map_resolution
    full_w, full_h = map_size, map_size
    local_w, local_h = int(full_w / args.global_downscaling), \
                       int(full_h / args.global_downscaling)

    # Initializing full and local map
    full_map = torch.zeros(num_scenes, nc, full_w, full_h).float().to(device)
    local_map = torch.zeros(num_scenes, nc, local_w, local_h).float().to(device)

    # Initial full and local pose
    full_pose = torch.zeros(num_scenes, 3).float().to(device)
    local_pose = torch.zeros(num_scenes, 3).float().to(device)

    # Origin of local map
    origins = np.zeros((num_scenes, 3))

    # Local Map Boundaries
    lmb = np.zeros((num_scenes, 4)).astype(int)

    ### Planner pose inputs has 7 dimensions
    ### 1-3 store continuous global agent location
    ### 4-7 store local map boundaries
    planner_pose_inputs = np.zeros((num_scenes, 7))

    def get_local_map_boundaries(agent_loc, local_sizes, full_sizes):
        loc_r, loc_c = agent_loc
        local_w, local_h = local_sizes
        full_w, full_h = full_sizes

        if args.global_downscaling > 1:
            gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
            gx2, gy2 = gx1 + local_w, gy1 + local_h
            if gx1 < 0:
                gx1, gx2 = 0, local_w
            if gx2 > full_w:
                gx1, gx2 = full_w - local_w, full_w

            if gy1 < 0:
                gy1, gy2 = 0, local_h
            if gy2 > full_h:
                gy1, gy2 = full_h - local_h, full_h
        else:
            gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

        return [gx1, gx2, gy1, gy2]

    def init_map_and_pose():
        full_map.fill_(0.)
        full_pose.fill_(0.)
        full_pose[:, :2] = args.map_size_cm / 100.0 / 2.0

        locs = full_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]

            full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

            lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                              (local_w, local_h),
                                              (full_w, full_h))

            planner_pose_inputs[e, 3:] = lmb[e]
            origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                          lmb[e][0] * args.map_resolution / 100.0, 0.]

        for e in range(num_scenes):
            local_map[e] = full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
            local_pose[e] = full_pose[e] - \
                            torch.from_numpy(origins[e]).to(device).float()

    def init_map_and_pose_for_env(e):
        full_map[e].fill_(0.)
        full_pose[e].fill_(0.)
        full_pose[e, :2] = args.map_size_cm / 100.0 / 2.0

        locs = full_pose[e].cpu().numpy()
        planner_pose_inputs[e, :3] = locs
        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]

        full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

        lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                          (local_w, local_h),
                                          (full_w, full_h))

        planner_pose_inputs[e, 3:] = lmb[e]
        origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                      lmb[e][0] * args.map_resolution / 100.0, 0.]

        local_map[e] = full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
        local_pose[e] = full_pose[e] - \
                        torch.from_numpy(origins[e]).to(device).float()


    init_map_and_pose()

    # slam
    sem_map_module = Semantic_Mapping(args).to(device)
    sem_map_module.eval()
    sem_map_module.set_view_angles([45] * args.num_processes)

    # Predict semantic map from frame 1
    poses = torch.from_numpy(np.asarray(
            [infos[env_idx]['sensor_pose'] for env_idx in range(num_scenes)])
            ).float().to(device)
    
    _, local_map, _, local_pose = \
        sem_map_module(obs, poses, local_map, local_pose)

    # Compute Global policy input
    locs = local_pose.cpu().numpy()

    for e in range(num_scenes):
        r, c = locs[e, 1], locs[e, 0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]

        local_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.


    #For now
    global_goals = []
    for e in range(num_scenes):
        np.random.seed(e); c1 = np.random.choice(local_w)
        np.random.seed(e + 1000); c2 = np.random.choice(local_h)
        global_goals.append((c1,c2))

    goal_maps = [np.zeros((local_w, local_h)) for _ in range(num_scenes)]

    for e in range(num_scenes):
        goal_maps[e][global_goals[e][0], global_goals[e][1]] = 1
        
    newly_goal_set = False
    planner_inputs = [{} for e in range(num_scenes)]
    for e, p_input in enumerate(planner_inputs):
        p_input['newly_goal_set'] =newly_goal_set
        p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
        p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
        p_input['pose_pred'] = planner_pose_inputs[e]
        p_input['goal'] = goal_maps[e] 
        p_input['new_goal'] = 1
        p_input['found_goal'] = 0
        p_input['wait'] = wait_env[e] or finished[e]
        p_input['list_of_actions'] = list_of_actions_s[e]
        p_input['list_of_actions_pointer'] = list_of_actions_pointer_s[e]
        p_input['consecutive_interaction'] = None
        p_input['consecutive_target'] = None
        if args.visualize or args.print_images:
            local_map[e, -1, :, :] = 1e-5
            p_input['sem_map_pred'] = local_map[e, 4:, :,
                :].argmax(0).cpu().numpy()

    
    obs, rew, done, infos, goal_success_s, next_step_dict_s = envs.plan_act_and_preprocess(planner_inputs, goal_spotted_s)
    goal_success_s = list(goal_success_s)
    view_angles = []
    for e in range(num_scenes):
        next_step_dict = next_step_dict_s[e]
        view_angle = next_step_dict['view_angle']
        view_angles.append(view_angle)
        
        fails[e] += next_step_dict['fails_cur']
        
    sem_map_module.set_view_angles(view_angles)
    
    
    consecutive_interaction_s, target_instance_s = [None]*num_scenes, [None]*num_scenes
    for e in range(num_scenes):
        num_steps_so_far[e] = next_step_dict_s[e]['steps_taken']
        first_steps[e] = False
        if goal_success_s[e]: 
            if list_of_actions_pointer_s[e] == len(list_of_actions_s[e]) -1:
                all_completed[e] = True
            else:
                list_of_actions_pointer_s[e] +=1
                goal_name = list_of_actions_s[e][list_of_actions_pointer_s[e]][0]
                
                reset_goal_true_false = [False]* num_scenes
                reset_goal_true_false[e] = True
                
                    
                #If consecutive interactions, 
                returned, target_instance_s[e] = determine_consecutive_interx(list_of_actions_s[e], list_of_actions_pointer_s[e]-1, whether_sliced_s[e])
                if returned:
                    consecutive_interaction_s[e] = list_of_actions_s[e][list_of_actions_pointer_s[e]][1]

                infos = envs.reset_goal(reset_goal_true_false, goal_name, consecutive_interaction_s)
        
    

    torch.set_grad_enabled(False)
    spl_per_category = defaultdict(list)
    success_per_category = defaultdict(list)

    for step in range(args.num_training_frames//args.num_processes):
        if sum(finished) == args.num_processes:
            print("all finished")
            if args.leaderboard and args.test:
                if args.test_seen:
                    add_str = "seen"
                else:
                    add_str = "unseen"
                pickle.dump(actseqs, open("results/leaderboard/actseqs_test_" + add_str + "_" + dn + ".p", "wb"))
            
            
            break

        l_step = step % args.num_local_steps

        # Reinitialize variables when episode ends
        for e,x in enumerate(task_finish):
            if x:
                spl = infos[e]['spl']
                success = infos[e]['success']
                dist = infos[e]['distance_to_goal']
                spl_per_category[infos[e]['goal_name']].append(spl)
                success_per_category[infos[e]['goal_name']].append(success)
                traj_number[e] +=1
                wait_env[e] = 1.
                init_map_and_pose_for_env(e)
                
                if not(finished[e]):
                    #load next episode for env
                    number_of_this_episode = args.from_idx + traj_number[e] * num_scenes + e
                    print("steps taken for episode# ",  number_of_this_episode-num_scenes , " is ", next_step_dict_s[e]['steps_taken'])
                    completed_episodes.append(number_of_this_episode)
                    pickle.dump(completed_episodes, open("results/completed_episodes_" + args.eval_split + str(args.from_idx) + "_to_" + str(args.to_idx) + "_" + dn +".p", 'wb'))
                    if args.leaderboard and args.test:
                        if args.test_seen:
                            add_str = "seen"
                        else:
                            add_str = "unseen"
                        pickle.dump(actseqs, open("results/leaderboard/actseqs_test_" + add_str + "_" + dn + ".p", "wb"))
                    load = [False] * args.num_processes
                    load[e] = True
                    do_not_update_cat_s[e] = None
                    wheres_delete_s[e] = np.zeros((240,240))
                    obs, infos, actions_dicts = envs.load_next_scene(load)
                    view_angles[e] = 45
                    sem_map_module.set_view_angles(view_angles)
                    if  actions_dicts[e] is None:
                        finished[e] = True
                    else:
                        second_objects[e] = actions_dicts[e]['second_object']
                        print("second object is ", second_objects[e])
                        list_of_actions_s[e] = actions_dicts[e]['list_of_actions']
                        task_types[e] = actions_dicts[e]['task_type']
                        whether_sliced_s[e] = actions_dicts[e]['sliced']
                    
                        task_finish[e] = False
                        num_steps_so_far[e] = 0
                        list_of_actions_pointer_s[e] = 0
                        goal_spotted_s[e] = False   
                        found_goal[e] = 0
                        list_of_actions_pointer_s[e] = 0
                        first_steps[e] = True
                        
                        all_completed[e] = False
                        goal_success_s[e] = False
                        
                        obs = torch.tensor(obs).to(device)
                        fails[e] = 0
                        goal_logs[e] = []
                        goal_cat_before_second_objects[e] = None
                        
        
        # ------------------------------------------------------------------
        # Semantic Mapping Module
        poses = torch.from_numpy(np.asarray(
                    [infos[env_idx]['sensor_pose'] for env_idx
                     in range(num_scenes)])
                ).float().to(device)

        _, local_map, _, local_pose = sem_map_module(obs, poses, local_map, local_pose, build_maps = True, no_update = False)

        locs = local_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs + origins
        local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]
            local_map[e, 2:4, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.

        for e in range(num_scenes):
            if not(do_not_update_cat_s[e] is None):
                cn = do_not_update_cat_s[e] + 4
                local_map[e, cn, :, :] = torch.zeros(local_map[0, 0, :, :].shape)

        for e in range(num_scenes):
            if args.delete_from_map_after_move_until_visible and (next_step_dict_s[e]['move_until_visible_cycled'] or next_step_dict_s[e]['delete_lamp']):
                ep_num = args.from_idx + traj_number[e] * num_scenes + e
                #Get the label that is closest to the current goal
                cn = infos[e]['goal_cat_id'] + 4

                start_x, start_y, start_o, gx1, gx2, gy1, gy2 = planner_pose_inputs[e]
                gx1, gx2, gy1, gy2  = int(gx1), int(gx2), int(gy1), int(gy2)
                r, c = start_y, start_x
                start = [int(r * 100.0/args.map_resolution - gx1),
                         int(c * 100.0/args.map_resolution - gy1)]
                map_pred = np.rint(local_map[e, 0, :, :].cpu().numpy())
                assert local_map[e, 0, :, :].shape[0] == 240
                start = pu.threshold_poses(start, map_pred.shape)

                lm = local_map[e, cn, :, :].cpu().numpy()
                lm = (lm>0).astype(int)
                lm = skimage.morphology.binary_dilation(lm, skimage.morphology.disk(4))
                lm = lm.astype(int)
                connected_regions = skimage.morphology.label(lm, connectivity=2)
                unique_labels = [i for i in range(0, np.max(connected_regions)+1)]
                min_dist = 1000000000
                for lab in unique_labels:
                    wheres = np.where(connected_regions == lab)
                    center = (int(np.mean(wheres[0])), int(np.mean(wheres[1])))
                    dist_pose = math.sqrt((start[0] -center[0])**2 + (start[1] -center[1])**2)
                    min_dist = min(min_dist, dist_pose)
                    if min_dist == dist_pose:
                        min_lab = lab

                #Delete that label
                wheres_delete_s[e][np.where(connected_regions == min_lab)] = 1

                
        for e in range(num_scenes):
            cn = infos[e]['goal_cat_id'] + 4
            wheres = np.where(wheres_delete_s[e])
            local_map[e, cn, :, :][wheres] = 0.0


        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Semantic Policy
        newly_goal_set = False
        if l_step == args.num_local_steps - 1:
            newly_goal_set = True
            for e in range(num_scenes):
                if wait_env[e] == 1: # New episode
                    wait_env[e] = 0.

                full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]] = \
                    local_map[e]
                full_pose[e] = local_pose[e] + \
                               torch.from_numpy(origins[e]).to(device).float()

                locs = full_pose[e].cpu().numpy()
                r, c = locs[1], locs[0]
                loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                int(c * 100.0 / args.map_resolution)]

                lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                                  (local_w, local_h),
                                                  (full_w, full_h))

                planner_pose_inputs[e, 3:] = lmb[e]
                origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                              lmb[e][0] * args.map_resolution / 100.0, 0.]

                local_map[e] = full_map[e, :,
                               lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
                local_pose[e] = full_pose[e] - \
                                torch.from_numpy(origins[e]).to(device).float()


            locs = local_pose.cpu().numpy()
            
            for e in range(num_scenes):
                goal_name = list_of_actions_s[e][list_of_actions_pointer_s[e]][0]
                if  args.use_sem_policy:
                    
                    #Just reconst the common map save objects
                    map_reconst = torch.zeros((4+len(large_objects2idx),240,240))
                    map_reconst[:4] = local_map[e][:4]
                    test_see = {}
                    map_reconst[4+large_objects2idx['SinkBasin']] = local_map[e][4+1]
                    test_see[1] = 'SinkBasin'

                    start_idx = 2
                    for cat, catid in large_objects2idx.items():
                        if not (cat =='SinkBasin'):
                            map_reconst[4+large_objects2idx[cat]] = local_map[e][4+start_idx]
                            test_see[start_idx] = cat
                            start_idx +=1

                    if local_map[e][0][120,120] == 0:
                        mask = np.zeros((240,240))
                        connected_regions = skimage.morphology.label(1-local_map[e][0].cpu().numpy(), connectivity=2)
                        connected_lab = connected_regions[120,120]
                        mask[np.where(connected_regions==connected_lab)] = 1
                        mask[np.where(skimage.morphology.binary_dilation(local_map[e][0].cpu().numpy(), skimage.morphology.square(4)))] = 1
                    else:
                        dilated = skimage.morphology.binary_dilation(local_map[e][0].cpu().numpy(), skimage.morphology.square(4))
                        mask = skimage.morphology.convex_hull_image(dilated).astype(float)
                    mask_grid = into_grid(torch.tensor(mask), 8).cpu()
                    where_ones = len(torch.where(mask_grid)[0])
                    mask_grid = mask_grid.repeat(73,1).view(73, -1).numpy()

                    if  goal_name in all_objects2idx and next_step_dict_s[e]['steps_taken'] >= 30:

                        pred_probs = Unet_model(map_reconst.unsqueeze(0).to(device))
                        pred_probs = pred_probs.view(73, -1)
                        pred_probs = softmax(pred_probs).cpu().numpy()

                        pred_probs = (1-args.explore_prob) * pred_probs + args.explore_prob * mask_grid * 1/ float(where_ones)

                        #Now sample from pred_probs according to goal idx
                        goal_name = list_of_actions_s[e][list_of_actions_pointer_s[e]][0]
                        if goal_name =='FloorLamp':
                            pred_probs = pred_probs[all_objects2idx[goal_name]] + pred_probs[all_objects2idx['DeskLamp']]
                            pred_probs = pred_probs/2.0
                        else:
                            pred_probs = pred_probs[all_objects2idx[goal_name]]


                    else:
                        pred_probs =  mask_grid[0] * 1/ float(where_ones)

                    

                    
                    if args.explore_prob==1.0:
                        mask_wheres = np.where(mask.astype(float))
                        np.random.seed(next_step_dict_s[e]['steps_taken'])
                        s_i= np.random.choice(len(mask_wheres[0]))
                        x_240, y_240 = mask_wheres[0][s_i], mask_wheres[1][s_i]

                    else:
                        #Now sample one index
                        np.random.seed(next_step_dict_s[e]['steps_taken'])
                        pred_probs = pred_probs.astype('float64')
                        pred_probs = pred_probs.reshape(64)
                        pred_probs = pred_probs/ np.sum(pred_probs)

                        chosen_cell = np.random.multinomial(1, pred_probs.tolist())
                        chosen_cell = np.where(chosen_cell)[0][0]                        
                        chosen_cell_x = int(chosen_cell/8)
                        chosen_cell_y = chosen_cell %8
                        
                        
                        #Sample among this mask
                        mask_new = np.zeros((240,240))
                        mask_new[chosen_cell_x*30:chosen_cell_x*30+30, chosen_cell_y*30: chosen_cell_y*30 + 30] = 1 
                        mask_new = mask_new * mask
                        if np.sum(mask_new) == 0:
                            np.random.seed(next_step_dict_s[e]['steps_taken'])
                            chosen_i = np.random.choice(len(np.where(mask)[0]))
                            x_240 = np.where(mask)[0][chosen_i]
                            y_240 = np.where(mask)[1][chosen_i]

                        else:
                            np.random.seed(next_step_dict_s[e]['steps_taken'])
                            chosen_i = np.random.choice(len(np.where(mask_new)[0]))
                            x_240 = np.where(mask_new)[0][chosen_i]
                            y_240 = np.where(mask_new)[1][chosen_i]
                        


                    global_goals[e] = [x_240, y_240]
                    test_goals= np.zeros((240,240))
                    test_goals[x_240,y_240]=1

                            

        # ------------------------------------------------------------------
    
    
        # ------------------------------------------------------------------
        # Take action and get next observation
        found_goal = [0 for _ in range(num_scenes)]
        goal_maps = [np.zeros((local_w, local_h)) for _ in range(num_scenes)]

        for e in range(num_scenes):
            goal_maps[e][global_goals[e][0], global_goals[e][1]] = 1


        for e in range(num_scenes):
            ep_num = args.from_idx + traj_number[e] * num_scenes + e
            cn = infos[e]['goal_cat_id'] + 4
            prev_cns[e] = cn
            cur_goal_sliced = next_step_dict_s[e]['current_goal_sliced']

            if local_map[e, cn, :, :].sum() != 0.: 
                ep_num = args.from_idx + traj_number[e] * num_scenes + e
                cat_semantic_map = local_map[e, cn, :, :].cpu().numpy()
                cat_semantic_scores = cat_semantic_map 

                cat_semantic_scores[cat_semantic_scores > 0] = 1.
                wheres = np.where(wheres_delete_s[e])
                cat_semantic_scores[wheres] = 0
                if np.sum(cat_semantic_scores) !=0:
                    goal_maps[e] = cat_semantic_scores

                if np.sum(cat_semantic_scores) !=0:
                    found_goal[e] = 1
                    goal_spotted_s[e] = True
                else:
                    if args.delete_from_map_after_move_until_visible or args.delete_pick2:
                        found_goal[e] = 0
                        goal_spotted_s[e] = False
            else:
                if args.delete_from_map_after_move_until_visible or args.delete_pick2:
                    found_goal[e] = 0
                    goal_spotted_s[e] = False



        planner_inputs = [{} for e in range(num_scenes)]
        for e, p_input in enumerate(planner_inputs):
            p_input['newly_goal_set'] =newly_goal_set
            p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
            p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
            p_input['pose_pred'] = planner_pose_inputs[e]
            p_input['goal'] = goal_maps[e] 
            p_input['new_goal'] = l_step == args.num_local_steps - 1
            p_input['found_goal'] = found_goal[e]
            p_input['wait'] = wait_env[e] or finished[e]
            p_input['list_of_actions'] = list_of_actions_s[e]
            p_input['list_of_actions_pointer'] = list_of_actions_pointer_s[e]
            p_input['consecutive_interaction'] = consecutive_interaction_s[e]
            p_input['consecutive_target'] = target_instance_s[e]
            if args.visualize or args.print_images:
                local_map[e, -1, :, :] = 1e-5
                p_input['sem_map_pred'] = local_map[e, 4:, :,
                    :].argmax(0).cpu().numpy()
                
            if first_steps[e]:
                p_input['consecutive_interaction'] = None
                p_input['consecutive_target'] = None
                
        ###################################      
        ###################################
        obs, rew, done, infos, goal_success_s, next_step_dict_s = envs.plan_act_and_preprocess(planner_inputs, goal_spotted_s)
        goal_success_s = list(goal_success_s)
        view_angles = []
        for e, p_input in enumerate(planner_inputs):
            next_step_dict = next_step_dict_s[e]
                        
            view_angle = next_step_dict['view_angle']
            
            view_angles.append(view_angle)
            
            num_steps_so_far[e] = next_step_dict['steps_taken']
            first_steps[e] = False 
            
            fails[e] += next_step_dict['fails_cur']
            if args.leaderboard and fails[e] >= args.max_fails:
                print("Interact API failed %d times" % fails[e] )
                task_finish[e] = True

            if not(args.no_pickup) and (args.map_mask_prop !=1 or args.no_pickup_update) and next_step_dict['picked_up'] and goal_success_s[e]:
                do_not_update_cat_s[e] = infos[e]['goal_cat_id']
            elif not(next_step_dict['picked_up']):
                do_not_update_cat_s[e] = None
                    
            
        sem_map_module.set_view_angles(view_angles)
              
        #####################################
        #####################################            
        
        for e, p_input in enumerate(planner_inputs):
            if p_input['wait'] ==1  or next_step_dict_s[e]['keep_consecutive']:
                pass
            else:
                consecutive_interaction_s[e], target_instance_s[e] = None, None
            
            if goal_success_s[e]:
                if list_of_actions_pointer_s[e] == len(list_of_actions_s[e]) -1:
                    all_completed[e] = True
                else:
                    list_of_actions_pointer_s[e] +=1
                    goal_name = list_of_actions_s[e][list_of_actions_pointer_s[e]][0]
                    
                    reset_goal_true_false = [False]* num_scenes
                    reset_goal_true_false[e] = True
                    
                    
                    returned, target_instance_s[e] = determine_consecutive_interx(list_of_actions_s[e], list_of_actions_pointer_s[e]-1, whether_sliced_s[e])
                    if returned:
                        consecutive_interaction_s[e] = list_of_actions_s[e][list_of_actions_pointer_s[e]][1]
                    infos = envs.reset_goal(reset_goal_true_false, goal_name, consecutive_interaction_s)
                    goal_spotted_s[e] = False
                    found_goal[e] = 0
                    wheres_delete_s[e] = np.zeros((240,240))
                           
        
        time.sleep(args.wait_time)

        # ------------------------------------------------------------------
        #End episode and log
        for e in range(num_scenes):
            number_of_this_episode = args.from_idx + traj_number[e] * num_scenes + e
            if number_of_this_episode in skip_indices:
                task_finish[e] = True
        
        for e in range(num_scenes):
            if all_completed[e]:
                if not(finished[e]) and args.test:
                    print("This episode is probably Success!")
                task_finish[e] = True
            
        for e in range(num_scenes):
            if num_steps_so_far[e] >= args.max_episode_length and not(finished[e]):
                print("This outputted")
                task_finish[e] = True
            
        for e in range(num_scenes):
            number_of_this_episode = args.from_idx + traj_number[e] * num_scenes + e
            if task_finish[e] and not(finished[e]) and not(number_of_this_episode in skip_indices): 
                f = open("results/logs/log_" + args.eval_split + "_from_" + str(args.from_idx) + "_to_" + str(args.to_idx) + "_" + dn +".txt" , "a")
                number_of_this_episode = args.from_idx + traj_number[e] * num_scenes + e
                f.write("\n")
                f.write("===================================================\n")
                f.write("episode # is " + str(number_of_this_episode) + "\n")
                
                for log in next_step_dict_s[e]['logs']:
                    f.write(log + "\n")

                if all_completed[e]:
                    if not(finished[e]) and args.test:
                        f.write("This episode is probably Success!\n")

                if num_steps_so_far[e] >= args.max_episode_length and not(finished[e]):
                    f.write("This outputted\n")
                
                if not(args.test):
                    log_entry, success = envs.evaluate(e) #success is  (True,), log_entry is ({..}, )
                    log_entry, success = log_entry[0], success[0]
                    print("success is ", success)
                    f.write("success is " + str(success) + "\n")
                    print("log entry is " + str(log_entry))
                    f.write("log entry is "+ str(log_entry) + "\n")
                    if success:
                        successes.append(log_entry)
                    else:
                        failures.append(log_entry)
                    
                    print("saving success and failures for episode # ", number_of_this_episode  , "and process number is", e)
                    pickle.dump(successes, open("results/successes/" + args.eval_split + "_successes_from_" + str(args.from_idx) + "_to_" + str(args.to_idx) +  "_" + dn +".p", "wb"))
                    pickle.dump(failures, open("results/fails/" + args.eval_split + "_failures_from_" + str(args.from_idx) + "_to_" + str(args.to_idx) + "_" + dn +".p", "wb"))
                
                else:
                    print("episode # ", number_of_this_episode  , "ended and process number is", e)
                
                
                if args.leaderboard and args.test:
                    actseq = next_step_dict_s[e]['actseq']
                    actseqs.append(actseq)
                f.close()

                #Add to analyze recs
                analyze_dict = {'task_type': actions_dicts[e]['task_type'], 'errs':next_step_dict_s[e]['errs'], 'action_pointer':list_of_actions_pointer_s[e], 'goal_found':goal_spotted_s[e],\
                  'number_of_this_episode': number_of_this_episode}
                if not(args.test):
                    analyze_dict['success'] = envs.evaluate(e)[1][0]
                else:
                    analyze_dict['success'] = all_completed[e]
                analyze_recs.append(analyze_dict)
                pickle.dump(analyze_recs, open("results/analyze_recs/" + args.eval_split + "_anaylsis_recs_from_" + str(args.from_idx) + "_to_" + str(args.to_idx) +  "_" + dn +".p", "wb"))
                





if __name__ == "__main__":
    main()
    print("All finsihed!")
