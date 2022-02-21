import os
import sys

import cv2
import copy
import alfred_utils.gen.constants as constants
import numpy as np
from collections import Counter, OrderedDict
from alfred_utils.env.tasks import get_task
from ai2thor.controller import Controller
import alfred_utils.gen.utils.image_util as image_util
from alfred_utils.gen.utils import game_util
from alfred_utils.gen.utils.game_util import get_objects_of_type, get_obj_of_type_closest_to_obj
import torch
from torch.autograd import Variable
from models.depth.alfred_perception_models import AlfredSegmentationAndDepthModel

from arguments import get_args

import quaternion
import gym

import envs.utils.pose as pu

import matplotlib

if sys.platform == 'darwin':
    matplotlib.use("tkagg")
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import pickle
from types import SimpleNamespace
import skimage.morphology


DEFAULT_RENDER_SETTINGS = {'renderImage': True,
                           'renderDepthImage': True,
                           'renderClassImage': False,
                           'renderObjectImage': True,
                           }

def noop(self):
    pass

class ThorEnvCode(Controller):
    '''
    an extension of ai2thor.controller.Controller for ALFRED tasks
    '''
    def __init__(self, args, rank, x_display=constants.X_DISPLAY,
                 player_screen_height=constants.DETECTION_SCREEN_HEIGHT,
                 player_screen_width=constants.DETECTION_SCREEN_WIDTH,
                 quality='MediumCloseFitShadows',
                 build_path=constants.BUILD_PATH,
                 habitat_config=None):

        super().__init__(quality=quality)
        
        self.local_executable_path = build_path
        self.steps_taken = 0
        
        if args.matrix:
            Controller.lock_release = noop
            Controller.unlock_release = noop
            Controller.prune_releases = noop
        
        self.start(x_display=args.x_display,
                   player_screen_height=player_screen_height,
                   player_screen_width=player_screen_width)
        self.task = None

        # internal states
        self.cleaned_objects = set()
        self.cooled_objects = set()
        self.heated_objects = set()

        #initializations compatible with object_goal_env
        if args.visualize:
            plt.ion()
        if args.print_images or args.visualize:
            if args.visualize == 1:
                widths = [4, 4, 3]
                total_w = 11
            elif args.visualize == 2:
                widths = [4, 3, 3]
                total_w = 10
            else:
                widths = [4, 3]
            self.figure, self.ax = plt.subplots(1, len(widths), figsize=(6*16./9., 6),
                                                gridspec_kw={'width_ratios': widths},
                                                facecolor="whitesmoke",
                                                num="Thread {}".format(rank))
        
        self.args = args 
        self.rank = rank
        
        self.action_space = gym.spaces.Discrete(3)

        self.observation_space = gym.spaces.Box(0, 255,
                                                (3, constants.SCREEN_HEIGHT,
                                                    constants.SCREEN_WIDTH),
                                                dtype='uint8')
        
        self.episode_no = 0
        self.last_scene_path = None
        self.info = {}
        self.info['distance_to_goal'] = None
        self.info['spl'] = None
        self.info['success'] = None
        
        self.view_angle = 0
        self.consecutive_steps = False
        self.final_sidestep = False
        
        self.actions = []
        self.max_depth = 5

        if args.use_learned_depth:
            self.depth_gpu =  torch.device("cuda:" + str(args.depth_gpu) if args.cuda else "cpu")
            if not(args.valts_depth):
                bts_args = SimpleNamespace(model_name='bts_nyu_v2_pytorch_densenet161' ,
                encoder='densenet161_bts',
                dataset='alfred',
                input_height=300,
                input_width=300,
                max_depth=5,
                mode = 'test',
                device = self.depth_gpu,
                set_view_angle=False,
                load_encoder=False,
                load_decoder=False,
                bts_size=512)

                if self.args.depth_angle or self.args.cpp:
                    bts_args.set_view_angle = True
                    bts_args.load_encoder = True

                self.depth_pred_model = BtsModel(params=bts_args).to(device=self.depth_gpu)
                print("depth initialized")

                if args.cuda:
                    ckpt_path = 'models/depth/depth_models/' + args.depth_checkpoint_path 
                else:
                    ckpt_path = 'models/depth/depth_models/' + args.depth_checkpoint_path 
                checkpoint = torch.load(ckpt_path, map_location=self.depth_gpu)['model']

                new_checkpoint = OrderedDict()
                for k, v in checkpoint.items():
                    name = k[7:] # remove `module.`
                    new_checkpoint[name] = v
                del checkpoint
                # load params
                self.depth_pred_model.load_state_dict(new_checkpoint)
                self.depth_pred_model.eval()
                self.depth_pred_model.to(device=self.depth_gpu)

            #Use Valts depth
            else:

                model_path ='valts/model-2000-best_silog_10.13741' #45 degrees only model

                if self.args.depth_model_old:
                    model_path ='valts/model-34000-best_silog_16.80614'

                elif self.args.depth_model_45_only:
                    model_path = 'valts/model-500-best_d3_0.98919'

                self.depth_pred_model = AlfredSegmentationAndDepthModel()

                state_dict = torch.load('models/depth/depth_models/' +model_path, map_location=self.depth_gpu)['model']

                new_checkpoint = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_checkpoint[name] = v
                
                state_dict = new_checkpoint
                del new_checkpoint

                self.depth_pred_model.load_state_dict(state_dict)
                self.depth_pred_model.eval()
                self.depth_pred_model.to(device=self.depth_gpu)

                if self.args.separate_depth_for_straight:
                    model_path = 'valts0/model-102500-best_silog_17.00430'

                    

                    self.depth_pred_model_0 = AlfredSegmentationAndDepthModel()
                    state_dict = torch.load('models/depth/depth_models/' +model_path, map_location=self.depth_gpu)['model']

                    new_checkpoint = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:] # remove `module.`
                        new_checkpoint[name] = v
                    
                    state_dict = new_checkpoint
                    del new_checkpoint

                    self.depth_pred_model_0.load_state_dict(state_dict)
                    self.depth_pred_model_0.eval()
                    self.depth_pred_model_0.to(device=self.depth_gpu)
                    
        
        print("ThorEnv started.")
        
    #Added for my convenience
    def setup_scene(self, traj_data, task_type, r_idx, args, reward_type='dense'):
        '''
        intialize the scene and agent from the task info
        '''
        # scene setup
        self.view_angle = 30
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']
        object_toggles = traj_data['scene']['object_toggles']
        self.steps_taken = 0
        self.errs = []
        self.actions = [dict(action="LookDown_15", forceAction=True)]
        if not(self.args.approx_horizon):
            self.camera_horizon =0
        else:
            self.camera_horizon = 0

        scene_name = 'FloorPlan%d' % scene_num
        self.reset(scene_name, return_event=True)
        self.restore_scene(object_poses, object_toggles, dirty_and_empty)
        if not(self.args.test):
            self.set_task(traj_data, task_type, args, task_type_optional=task_type)

        # initialize to start position
        init_dict = dict(traj_data['scene']['init_action'])
        obs, _, _, info = self.step(init_dict)
        if not(self.args.approx_horizon):
            self.camera_horizon =self.event.metadata['agent']['cameraHorizon']
        else:
            self.camera_horizon = 30
        obs, _, _, info = self.step(dict(action='LookDown_15',
                              forceAction=True))
        objects = self.event.metadata["objects"]
        tables = []
        
        if not(r_idx is None):
            print("Task: %s" % (traj_data['turk_annotations']['anns'][r_idx]['task_desc']))

        # setup task for reward
        if not(self.args.test):
            self.set_task(traj_data, task_type, args, reward_type=reward_type, task_type_optional=task_type)
        self.info = info
        
        return obs, info


    def reset(self, scene_name_or_num,
              grid_size=constants.AGENT_STEP_SIZE / constants.RECORD_SMOOTHING_FACTOR,
              camera_y=constants.CAMERA_HEIGHT_OFFSET,
              render_image=constants.RENDER_IMAGE,
              render_depth_image=constants.RENDER_DEPTH_IMAGE,
              render_class_image=constants.RENDER_CLASS_IMAGE,
              render_object_image=constants.RENDER_OBJECT_IMAGE,
              visibility_distance=constants.VISIBILITY_DISTANCE,
              return_event = False):
        '''
        reset scene and task states
        '''
        print("Resetting ThorEnv")

        if type(scene_name_or_num) == str:
            scene_name = scene_name_or_num
        else:
            scene_name = 'FloorPlan%d' % scene_name_or_num

        self.accumulated_pose = np.array([0.0,0.0,0.0])
        super().reset(scene_name)
        event = super().step(dict(
            action='Initialize',
            gridSize=grid_size,
            cameraY=camera_y,
            renderImage=render_image,
            renderDepthImage=render_depth_image,
            renderClassImage=render_class_image,
            renderObjectImage=render_object_image,
            visibility_distance=visibility_distance,
            makeAgentsVisible=False,
        ))
        self.event = event

        # reset task if specified
        if self.task is not None:
            self.task.reset()

        # clear object state changes
        self.reset_states()
        
        self.timestep = 0
        self.stopped = False
        self.path_length = 1e-5
        
        self.info['time'] = self.timestep
        self.info['sensor_pose'] = [0., 0., 0.]
        
        self.last_sim_location = self.get_sim_location()
        self.o = 0.0
        self.o_behind = 0.0
        
        if return_event:
            return event
        else:
            rgb = torch.tensor(event.frame.copy()).numpy() #shape (h, w, 3)
            depth = torch.tensor(event.depth_frame.copy()).numpy() #shape (h, w)
            depth /= 1000.0
            depth = np.expand_dims(depth, 2)
            
            state = np.concatenate((rgb, depth), axis = 2).transpose(2, 0, 1)
            
            return state, self.info


    def get_pose_change_approx(self, last_action, whether_success):
        if not(whether_success):
            return 0.0, 0.0, 0.0
        else:
            if "MoveAhead" in last_action:
                dx, dy, do = 0.25, 0.0, 0.0
            elif "RotateLeft" in last_action:
                dx, dy = 0.0, 0.0
                do = np.pi/2
            elif "RotateRight" in last_action:
                dx, dy = 0.0, 0.0
                do = -np.pi/2
            else:
                dx, dy, do = 0.0, 0.0, 0.0

            return dx, dy, do 

    def get_pose_change_approx_relative(self, last_action, whether_success):
        if not(whether_success):
            return 0.0, 0.0, 0.0
        else:
            if "MoveAhead" in last_action:
                do = 0.0
                if abs(self.o + 2*np.pi) <=1e-1 or abs(self.o) <=1e-1 or abs(self.o - 2*np.pi) <=1e-1: #o is 0
                    dx = 0.25
                    dy = 0.0
                elif abs(self.o + 2*np.pi - np.pi/2) <=1e-1 or abs(self.o - np.pi/2) <=1e-1 or abs(self.o - 2*np.pi - np.pi/2) <=1e-1:
                    dx = 0.0
                    dy = 0.25
                elif abs(self.o + 2*np.pi - np.pi) <=1e-1 or abs(self.o - np.pi) <=1e-1 or abs(self.o - 2*np.pi - np.pi) <=1e-1:
                    dx = -0.25
                    dy = 0.0
                elif abs(self.o + 2*np.pi - 3*np.pi/2) <=1e-1 or abs(self.o - 3*np.pi/2) <=1e-1 or abs(self.o - 2*np.pi - 3*np.pi/2) <=1e-1:
                    dx = 0.0
                    dy = -0.25
                else:
                    raise Exception("angle did not fall in anywhere")
            elif "RotateLeft" in last_action:
                dx, dy = 0.0, 0.0
                do = np.pi/2
            elif "RotateRight" in last_action:
                dx, dy = 0.0, 0.0
                do = -np.pi/2
            else:
                dx, dy, do = 0.0, 0.0, 0.0

        self.o = self.o + do
        if self.o  >= np.pi- 1e-1:
            self.o  -= 2 * np.pi

        return dx, dy, do 

    #Just closest angle by 15 degrees to the horizon
    def view_angle(self):
        hor = int(self.event.metadata['agent']['cameraHorizon']) 
        remainder = hor % 15
        remnant = int(hor/15)
        if remainder <= 7:
            return hor - remainder
        else:
            return 15 * (remnant +1)

    def reset_states(self):
        '''
        clear state changes
        '''
        self.cleaned_objects = set()
        self.cooled_objects = set()
        self.heated_objects = set()

    def restore_scene(self, object_poses, object_toggles, dirty_and_empty):
        '''
        restore object locations and states
        '''
        super().step(dict(
            action='Initialize',
            gridSize=constants.AGENT_STEP_SIZE / constants.RECORD_SMOOTHING_FACTOR,
            cameraY=constants.CAMERA_HEIGHT_OFFSET,
            renderImage=constants.RENDER_IMAGE,
            renderDepthImage=constants.RENDER_DEPTH_IMAGE,
            renderClassImage=constants.RENDER_CLASS_IMAGE,
            renderObjectImage=constants.RENDER_OBJECT_IMAGE,
            visibility_distance=constants.VISIBILITY_DISTANCE,
            makeAgentsVisible=False,
        ))
        if len(object_toggles) > 0:
            super().step((dict(action='SetObjectToggles', objectToggles=object_toggles)))

        if dirty_and_empty:
            super().step(dict(action='SetStateOfAllObjects',
                               StateChange="CanBeDirty",
                               forceAction=True))
            super().step(dict(action='SetStateOfAllObjects',
                               StateChange="CanBeFilled",
                               forceAction=False))
        super().step((dict(action='SetObjectPoses', objectPoses=object_poses)))

    def set_task(self, traj, task_type, args, reward_type='sparse', max_episode_length=2000, task_type_optional=None):
        '''
        set the current task type (one of 7 tasks)
        '''
        self.task = get_task(task_type, traj, self, args, reward_type=reward_type, max_episode_length=max_episode_length, task_type_optional=task_type_optional)
    
    def depth_center(self, depth_est):
        ht, wd, = 300, 300            
        ww = 544; hh = 416 #Change accordingly 
        xx = (ww - wd) // 2
        yy = (hh - ht) // 2

        return depth_est[:, :, yy:yy+ht, xx:xx+wd]

    def pad_rgb(self, rgb, ww=544, hh=416):
        ht, wd, cc= rgb.shape
        ht, wd, = 300, 300 

        # create new image of desired size and color (blue) for padding
        color = (255,255,255) #black
        result = np.full((hh,ww,cc), color, dtype=np.uint8)

        # compute center offset
        xx = (ww - wd) // 2
        yy = (hh - ht) // 2

        # copy img image into center of result image
        result[yy:yy+ht, xx:xx+wd] = rgb
        return result


    def success_for_look(self, action):
        wheres = np.where(self.prev_rgb != self.event.frame)
        wheres_ar = np.zeros(self.prev_rgb.shape)
        wheres_ar[wheres] = 1
        wheres_ar = np.sum(wheres_ar, axis=2).astype(bool)
        connected_regions = skimage.morphology.label(wheres_ar, connectivity=2)
        unique_labels = [i for i in range(1, np.max(connected_regions)+1)]
        max_area = -1
        for lab in unique_labels:
            wheres_lab = np.where(connected_regions == lab)
            max_area = max(len(wheres_lab[0]), max_area)
        if action in ['OpenObject', 'CloseObject'] and max_area > 500:
            success = True
        elif max_area > 100:
            success = True
        else:
            success = False
        return success

    def step(self, action, smooth_nav=False):
        self.prev_rgb = copy.deepcopy(self.event.frame)
        self.prev_depth = copy.deepcopy(self.event.depth_frame)
        self.prev_seg = copy.deepcopy(self.event.instance_segmentation_frame)
        '''
        overrides ai2thor.controller.Controller.step() for smooth navigation and goal_condition updates
        '''
        if action['action'] == '<<stop>>':
            self.stopped = True
        
        else:
            if smooth_nav:
                if "MoveAhead" in action['action']:
                    self.smooth_move_ahead(action)
                elif "Rotate" in action['action']:
                    self.smooth_rotate(action)
                elif "Look" in action['action']:
                    self.smooth_look(action)
                else:
                    super().step(action)
            else:
                if "LookUp" in action['action']:
                    angle = float(action['action'].split("_")[1])
                    self.event =self.look_angle(-angle)
                    if not(self.args.approx_last_action_success) and self.event.metadata['lastActionSuccess']:
                        self.camera_horizon += -angle
                    elif self.args.approx_last_action_success and self.success_for_look(action['action']):
                        self.camera_horizon += -angle
                elif "LookDown" in action['action']:
                    angle = float(action['action'].split("_")[1])
                    self.event= self.look_angle(angle)
                    if not(self.args.approx_last_action_success) and self.event.metadata['lastActionSuccess']:
                        self.camera_horizon += angle
                    elif self.args.approx_last_action_success and  self.success_for_look(action['action']):
                        self.camera_horizon += angle
                else:
                    self.event = super().step(action)
                    

            self.event =self.update_states(action)
            self.check_post_conditions(action)
        
        # Get pose change
        if not(self.args.approx_pose):
            if self.consecutive_steps == False:
                dx, dy, do = self.get_pose_change()
                self.info['sensor_pose'] = [dx, dy, do]
                self.path_length += pu.get_l2_distance(0, dx, 0, dy)
            else:
                if self.final_sidestep:
                    dx, dy, do = self.get_pose_change()
                    self.info['sensor_pose'] = [dx, dy, do]
                    self.path_length += pu.get_l2_distance(0, dx, 0, dy)
                else:
                    pass
        else:
            if self.args.approx_last_action_success:
                whether_success = self.success_for_look(action['action'])
            else:
                whether_success  = self.event.metadata['lastActionSuccess']
            if self.consecutive_steps:
                last_action = action['action']
                dx, dy, do = self.get_pose_change_approx_relative(last_action, whether_success)
                self.accumulated_pose += np.array([dx, dy, do])
                if self.accumulated_pose[2]  >= np.pi -1e-1:
                    self.accumulated_pose[2]  -= 2 * np.pi
                if self.final_sidestep:
                    self.info['sensor_pose'] = copy.deepcopy(self.accumulated_pose).tolist()
                    self.path_length += pu.get_l2_distance(0, dx, 0, dy)
                    self.accumulated_pose = np.array([0.0,0.0,0.0])
                    self.o = 0.0
            else:
                last_action = action['action']
                dx, dy, do = self.get_pose_change_approx(last_action, whether_success)
                self.info['sensor_pose'] = [dx, dy, do]
                self.path_length += pu.get_l2_distance(0, dx, 0, dy)
            
                
        
        done = self.get_done()
        
        if done:
            spl, success, dist = 0,0,0
            self.info['distance_to_goal'] = dist
            self.info['spl'] = spl
            self.info['success'] = success
        
        
        self.timestep += 1
        self.info['time'] = self.timestep
        

        rgb = self.event.frame.copy() #shape (h, w, 3)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        depth = torch.tensor(self.event.depth_frame.copy()).numpy() #shape (h, w)
        depth /= 1000.0 #in meters

        depth = np.expand_dims(depth, 2)
        
        state = np.concatenate((rgb, depth), axis = 2).transpose(2, 0, 1)
                
        rew = 0.0
        
        self.steps_taken +=1
        
        return state, rew, done, self.info
    
    def sim_continuous_to_sim_map(self, sim_loc):
        #x, y, o = self.get_sim_location()
        x, y, o = sim_loc
        min_x, min_y = self.map_obj_origin/100.0
        x, y = int((-x - min_x)*20.), int((-y - min_y)*20.)

        o = np.rad2deg(o) + 180.0
        return y, x, o
    
    def get_pose_change(self):
        curr_sim_pose = self.get_sim_location()
        dx, dy, do = pu.get_rel_pose_change(curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        return dx, dy, do
    

    def get_sim_location(self):
        y = -self.event.metadata['agent']['position']['x']
        x = self.event.metadata['agent']['position']['z'] 
        o = np.deg2rad(-self.event.metadata['agent']['rotation']['y'])
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o
    
    def get_metrics(self):
        curr_loc = self.sim_continuous_to_sim_map(self.get_sim_location())
        dist = self.gt_planner.fmm_dist[curr_loc[0], curr_loc[1]]/20.0
        if dist == 0.0:
            success = 1
        else:
            success = 0
        spl = min(success * self.starting_distance/self.path_length, 1)
        return spl, success, dist

    def get_done(self):
        if self.info['time'] >= self.args.max_episode_length-1:
            done = True
        elif self.stopped:
            done = True
        else:
            done = False
        return done


    def check_post_conditions(self, action):
        '''
        handle special action post-conditions
        '''
        if action['action'] == 'ToggleObjectOn':
            self.check_clean(action['objectId'])

    def update_states(self, action):
        '''
        extra updates to metadata after step
        '''
        event = self.event
        if event.metadata['lastActionSuccess']:
            # clean
            if action['action'] == 'ToggleObjectOn' and "Faucet" in action['objectId']:
                sink_basin = get_obj_of_type_closest_to_obj('SinkBasin', action['objectId'], event.metadata)
                cleaned_object_ids = sink_basin['receptacleObjectIds']
                self.cleaned_objects = self.cleaned_objects | set(cleaned_object_ids) if cleaned_object_ids is not None else set()
            # heat
            if action['action'] == 'ToggleObjectOn' and "Microwave" in action['objectId']:
                microwave = get_objects_of_type('Microwave', event.metadata)[0]
                heated_object_ids = microwave['receptacleObjectIds']
                self.heated_objects = self.heated_objects | set(heated_object_ids) if heated_object_ids is not None else set()
            # cool
            if action['action'] == 'CloseObject' and "Fridge" in action['objectId']:
                fridge = get_objects_of_type('Fridge', event.metadata)[0]
                cooled_object_ids = fridge['receptacleObjectIds']
                self.cooled_objects = self.cooled_objects | set(cooled_object_ids) if cooled_object_ids is not None else set()

        return event

    def get_transition_reward(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for transition_reward")
        else:
            return self.task.transition_reward(self.event)

    def get_goal_satisfied(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for goal_satisfied")
        else:
            return self.task.goal_satisfied(self.event)

    def get_goal_conditions_met(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for goal_satisfied")
        else:
            return self.task.goal_conditions_met(self.event)

    def get_subgoal_idx(self):
        if self.task is None:
            raise Exception("WARNING: no task setup for subgoal_idx")
        else:
            return self.task.get_subgoal_idx()

    def noop(self):
        '''
        do nothing
        '''
        super().step(dict(action='Pass'))

    def smooth_move_ahead(self, action, render_settings=None):
        '''
        smoother MoveAhead
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        smoothing_factor = constants.RECORD_SMOOTHING_FACTOR
        new_action = copy.deepcopy(action)
        new_action['moveMagnitude'] = constants.AGENT_STEP_SIZE / smoothing_factor

        new_action['renderImage'] = render_settings['renderImage']
        new_action['renderClassImage'] = render_settings['renderClassImage']
        new_action['renderObjectImage'] = render_settings['renderObjectImage']
        new_action['renderDepthImage'] = render_settings['renderDepthImage']

        events = []
        for xx in range(smoothing_factor - 1):
            event = super().step(new_action)
            if event.metadata['lastActionSuccess']:
                events.append(event)

        event = super().step(new_action)
        if event.metadata['lastActionSuccess']:
            events.append(event)
        return events

    def smooth_rotate(self, action, render_settings=None):
        '''
        smoother RotateLeft and RotateRight
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.event
        horizon = np.round(event.metadata['agent']['cameraHorizon'], 4)
        position = event.metadata['agent']['position']
        rotation = event.metadata['agent']['rotation']
        start_rotation = rotation['y']
        if action['action'] == 'RotateLeft':
            end_rotation = (start_rotation - 30)
        else:
            end_rotation = (start_rotation + 30)

        events = []
        for xx in np.arange(.1, 1.0001, .1):
            if xx < 1:
                teleport_action = {
                    'action': 'TeleportFull',
                    'rotation': np.round(start_rotation * (1 - xx) + end_rotation * xx, 3),
                    'x': position['x'],
                    'z': position['z'],
                    'y': position['y'],
                    'horizon': horizon,
                    'tempRenderChange': True,
                    'renderNormalsImage': False,
                    'renderImage': render_settings['renderImage'],
                    'renderClassImage': render_settings['renderClassImage'],
                    'renderObjectImage': render_settings['renderObjectImage'],
                    'renderDepthImage': render_settings['renderDepthImage'],
                }
                event = super().step(teleport_action)
            else:
                teleport_action = {
                    'action': 'TeleportFull',
                    'rotation': np.round(start_rotation * (1 - xx) + end_rotation * xx, 3),
                    'x': position['x'],
                    'z': position['z'],
                    'y': position['y'],
                    'horizon': horizon,
                }
                event = super().step(teleport_action)

            if event.metadata['lastActionSuccess']:
                events.append(event)
        return events

    def smooth_look(self, action, render_settings=None):
        '''
        smoother LookUp and LookDown
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.event
        start_horizon = event.metadata['agent']['cameraHorizon']
        rotation = np.round(event.metadata['agent']['rotation']['y'], 4)
        end_horizon = start_horizon + constants.AGENT_HORIZON_ADJ * (1 - 2 * int(action['action'] == 'LookUp'))
        position = event.metadata['agent']['position']

        events = []
        for xx in np.arange(.1, 1.0001, .1):
            if xx < 1:
                teleport_action = {
                    'action': 'TeleportFull',
                    'rotation': rotation,
                    'x': position['x'],
                    'z': position['z'],
                    'y': position['y'],
                    'horizon': np.round(start_horizon * (1 - xx) + end_horizon * xx, 3),
                    'tempRenderChange': True,
                    'renderNormalsImage': False,
                    'renderImage': render_settings['renderImage'],
                    'renderClassImage': render_settings['renderClassImage'],
                    'renderObjectImage': render_settings['renderObjectImage'],
                    'renderDepthImage': render_settings['renderDepthImage'],
                }
                event = super().step(teleport_action)
            else:
                teleport_action = {
                    'action': 'TeleportFull',
                    'rotation': rotation,
                    'x': position['x'],
                    'z': position['z'],
                    'y': position['y'],
                    'horizon': np.round(start_horizon * (1 - xx) + end_horizon * xx, 3),
                }
                event = super().step(teleport_action)

            if event.metadata['lastActionSuccess']:
                events.append(event)
        return events

    def height_change(self, height_change, render_settings=None):
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.event
        horizon = event.metadata['agent']['cameraHorizon']
        position = event.metadata['agent']['position']
        rotation = event.metadata['agent']['rotation']['y']
        
        teleport_action = {
            'action': 'TeleportFull',
            'rotation': rotation,
            'x': position['x'],
            'z': position['z'],
            'y': position['y'] + height_change,
            'horizon': horizon,
            'tempRenderChange': True,
            'renderNormalsImage': False,
            'renderImage': render_settings['renderImage'],
            'renderClassImage': render_settings['renderClassImage'],
            'renderObjectImage': render_settings['renderObjectImage'],
            'renderDepthImage': render_settings['renderDepthImage'],
        }
        #event = super().step(teleport_action)
        #return event
        super().step(teleport_action)
        
        self.event = event
        dx, dy, do = self.get_pose_change()
        #print("pose change is ", dx, dy, do)
        self.info['sensor_pose'] = [dx, dy, do]
        self.path_length += pu.get_l2_distance(0, dx, 0, dy)
        
        done = self.get_done()
        
        if done:
            #spl, success, dist = self.get_metrics()
            spl, success, dist = 0,0,0
            #print("spl, succ, dist: {}, {}, {}".format(spl, success, dist))
            self.info['distance_to_goal'] = dist
            self.info['spl'] = spl
            self.info['success'] = success
        
        #self.info['sensor_pose'] = 
        
        self.timestep += 1
        self.info['time'] = self.timestep
        
        rgb = torch.tensor(self.event.frame.copy()).numpy() #shape (h, w, 3)
        depth = torch.tensor(self.event.depth_frame.copy()).numpy() #shape (h, w)
        depth /= 1000.0
        depth = np.expand_dims(depth, 2)
        
        state = np.concatenate((rgb, depth), axis = 2).transpose(2, 0, 1)
                
        rew = 0.0
        #print("position['y'] is ", position['y'])
        return state, rew, done, self.info

    def look_angle(self, angle, render_settings=None):
        '''
        look at a specific angle
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.event
        start_horizon = event.metadata['agent']['cameraHorizon']
        rotation = np.round(event.metadata['agent']['rotation']['y'], 4)
        end_horizon = start_horizon + angle
        position = event.metadata['agent']['position']

        teleport_action = {
            'action': 'TeleportFull',
            'rotation': rotation,
            'x': position['x'],
            'z': position['z'],
            'y': position['y'],
            'horizon': np.round(end_horizon, 3),
            'tempRenderChange': True,
            'renderNormalsImage': False,
            'renderImage': render_settings['renderImage'],
            'renderClassImage': render_settings['renderClassImage'],
            'renderObjectImage': render_settings['renderObjectImage'],
            'renderDepthImage': render_settings['renderDepthImage'],
        }
        event = super().step(teleport_action)
        self.view_angle += angle
        return event
    
    def set_horizon(self, angle, render_settings=None):
        '''
        look at a specific angle
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.event
        #start_horizon = event.metadata['agent']['cameraHorizon']
        rotation = np.round(event.metadata['agent']['rotation']['y'], 4)
        end_horizon = angle
        position = event.metadata['agent']['position']

        teleport_action = {
            'action': 'TeleportFull',
            'rotation': rotation,
            'x': position['x'],
            'z': position['z'],
            'y': position['y'],
            'horizon': np.round(end_horizon, 3),
            'tempRenderChange': True,
            'renderNormalsImage': False,
            'renderImage': render_settings['renderImage'],
            'renderClassImage': render_settings['renderClassImage'],
            'renderObjectImage': render_settings['renderObjectImage'],
            'renderDepthImage': render_settings['renderDepthImage'],
        }
        event = super().step(teleport_action)
        return event

    def rotate_angle(self, angle, render_settings=None):
        '''
        rotate at a specific angle
        '''
        if render_settings is None:
            render_settings = DEFAULT_RENDER_SETTINGS
        event = self.event
        horizon = np.round(event.metadata['agent']['cameraHorizon'], 4)
        position = event.metadata['agent']['position']
        rotation = event.metadata['agent']['rotation']
        start_rotation = rotation['y']
        end_rotation = start_rotation + angle

        teleport_action = {
            'action': 'TeleportFull',
            'rotation': np.round(end_rotation, 3),
            'x': position['x'],
            'z': position['z'],
            'y': position['y'],
            'horizon': horizon,
            'tempRenderChange': True,
            'renderNormalsImage': False,
            'renderImage': render_settings['renderImage'],
            'renderClassImage': render_settings['renderClassImage'],
            'renderObjectImage': render_settings['renderObjectImage'],
            'renderDepthImage': render_settings['renderDepthImage'],
        }
        super().step(teleport_action)

    def to_thor_api_exec(self, action, object_id="", smooth_nav=False):
        action_received = copy.deepcopy(action)
        self.action_received =action_received
        self.last_action = action_received

        if "RotateLeft" in action:
            action = dict(action="RotateLeft", degrees = "90",
                          forceAction=True)
            obs, rew, done, info = self.step(action, smooth_nav=smooth_nav)
        elif "RotateRight" in action:
            action = dict(action="RotateRight", degrees = "90",
                          forceAction=True)
            obs, rew, done, info = self.step(action, smooth_nav=smooth_nav)
        elif "MoveAhead" in action:
            action = dict(action="MoveAhead",
                          forceAction=True)
            obs, rew, done, info = self.step(action, smooth_nav=smooth_nav)
        elif "LookUp" in action:
            #if abs(self.event.metadata['agent']['cameraHorizon']-0) <5:
            if abs(self.camera_horizon - 0) <5:
                action = dict(action="LookUp_0",
                              forceAction=True)
            else:
                action = dict(action=action,
                              forceAction=True)
            obs, rew, done, info = self.step(action, smooth_nav=smooth_nav)
        elif "LookDown" in action:
            #if abs(self.event.metadata['agent']['cameraHorizon'] - 90) <5:
            if abs(self.camera_horizon - 90) <5:
                action = dict(action="LookDown_0",
                              forceAction=True)
            else:
                action = dict(action=action,
                              forceAction=True)
            obs, rew, done, info= self.step(action, smooth_nav=smooth_nav)
        elif "OpenObject" in action:
            action = dict(action="OpenObject",
                          objectId=object_id,
                          moveMagnitude=1.0)
            obs, rew, done, info = self.step(action)
        elif "CloseObject" in action:
            action = dict(action="CloseObject",
                          objectId=object_id,
                          forceAction=True)
            obs, rew, done, info = self.step(action)
        elif "PickupObject" in action:
            action = dict(action="PickupObject",
                          objectId=object_id)
            obs, rew, done, info = self.step(action)
        elif "PutObject" in action:
            inventory_object_id = self.event.metadata['inventoryObjects'][0]['objectId']
            new_inventory_object_id = self.event.metadata['inventoryObjects'][0]['objectId']
            action = dict(action="PutObject",
                          objectId=inventory_object_id,
                          receptacleObjectId=object_id,
                          forceAction=True,
                          placeStationary=True)
            obs, rew, done, info = self.step(action)
        elif "ToggleObjectOn" in action:
            action = dict(action="ToggleObjectOn",
                          objectId=object_id)
            obs, rew, done, info = self.step(action)

        elif "ToggleObjectOff" in action:
            action = dict(action="ToggleObjectOff",
                          objectId=object_id)
            obs, rew, done, info = self.step(action)
        elif "SliceObject" in action:
            # check if agent is holding knife in hand
            inventory_objects = self.event.metadata['inventoryObjects']
            if len(inventory_objects) == 0 or 'Knife' not in inventory_objects[0]['objectType']:
                raise Exception("Agent should be holding a knife before slicing.")

            action = dict(action="SliceObject",
                          objectId=object_id)
            obs, rew, done, info = self.step(action)
        else:
            raise Exception("Invalid action. Conversion to THOR API failed! (action='" + str(action) + "')")

        return obs, rew, done, info, self.event, action
    

    def check_clean(self, object_id):
        '''
        Handle special case when Faucet is toggled on.
        In this case, we need to execute a `CleanAction` in the simulator on every object in the corresponding
        basin. This is to clean everything in the sink rather than just things touching the stream.
        '''
        event = self.event
        if event.metadata['lastActionSuccess'] and 'Faucet' in object_id:
            # Need to delay one frame to let `isDirty` update on stream-affected.
            self.step({'action': 'Pass'})
            event = self.event
            sink_basin_obj = game_util.get_obj_of_type_closest_to_obj("SinkBasin", object_id, event.metadata)
            for in_sink_obj_id in sink_basin_obj['receptacleObjectIds']:
                if (game_util.get_object(in_sink_obj_id, event.metadata)['dirtyable']
                        and game_util.get_object(in_sink_obj_id, event.metadata)['isDirty']):
                    self.step({'action': 'CleanObject', 'objectId': in_sink_obj_id})
        return self.event

    def prune_by_any_interaction(self, instances_ids):
        '''
        ignores any object that is not interactable in anyway
        '''
        pruned_instance_ids = []
        for obj in self.event.metadata['objects']:
            obj_id = obj['objectId']
            if obj_id in instances_ids:
                if obj['pickupable'] or obj['receptacle'] or obj['openable'] or obj['toggleable'] or obj['sliceable']:
                    pruned_instance_ids.append(obj_id) 

        ordered_instance_ids = [id for id in instances_ids if id in pruned_instance_ids]
        return ordered_instance_ids
    
    
    def print_log(self, *statements):
        statements = [str(s) for s in statements]
        statements = ['step #: ', str(self.steps_taken) , ","] + statements
        joined = ' '.join(statements)
        #print(joined)
        self.logs.append(joined)

    #Interact if within visibility distance
    def va_interact(self, action, interact_mask=None, smooth_nav=False, mask_px_sample=1, debug=False):
        '''
        interact mask based action call
        '''

        all_ids = []

        if type(interact_mask) is str and interact_mask == "NULL":
            raise Exception("NULL mask.")
        elif interact_mask is not None:
            # ground-truth instance segmentation mask from THOR
            instance_segs = np.array(self.event.instance_segmentation_frame)
            color_to_object_id = self.event.color_to_object_id

            # get object_id for each 1-pixel in the interact_mask
            nz_rows, nz_cols = np.nonzero(interact_mask)
            instance_counter = Counter()
            for i in range(0, len(nz_rows), mask_px_sample):
                x, y = nz_rows[i], nz_cols[i]
                instance = tuple(instance_segs[x, y])
                instance_counter[instance] += 1
            if debug:
                print("action_box", "instance_counter", instance_counter)

            # iou scores for all instances
            iou_scores = {}
            for color_id, intersection_count in instance_counter.most_common():
                union_count = np.sum(np.logical_or(np.all(instance_segs == color_id, axis=2), interact_mask.astype(bool)))
                iou_scores[color_id] = intersection_count / float(union_count)
            iou_sorted_instance_ids = list(OrderedDict(sorted(iou_scores.items(), key=lambda x: x[1], reverse=True)))

            # get the most common object ids ignoring the object-in-hand
            inv_obj = self.event.metadata['inventoryObjects'][0]['objectId'] \
                if len(self.event.metadata['inventoryObjects']) > 0 else None
            all_ids = [color_to_object_id[color_id] for color_id in iou_sorted_instance_ids
                       if color_id in color_to_object_id and color_to_object_id[color_id] != inv_obj]

            # print all ids
            if debug:
                print("action_box", "all_ids", all_ids)

            # print instance_ids
            instance_ids = [inst_id for inst_id in all_ids if inst_id is not None]
            instance_ids = self.prune_by_any_interaction(instance_ids)
            if debug:
                print("action_box", "instance_ids", instance_ids)

            # prune invalid instances like floors, walls, etc.
            if self.args.ground_truth_segmentation:
                pass
            else:
                instance_ids_new = self.prune_by_any_interaction(instance_ids)
                for instance_id in instance_ids:
                    if 'Sink' in instance_id:
                        instance_ids_new.append(instance_id)

            # cv2 imshows to show image, segmentation mask, interact mask
            if debug:
                print("action_box", "instance_ids", instance_ids)
                instance_seg = copy.copy(instance_segs)
                instance_seg[:, :, :] = interact_mask[:, :, np.newaxis] == 1
                instance_seg *= 255

                cv2.imshow('seg', instance_segs)
                cv2.imshow('mask', instance_seg)
                cv2.imshow('full', self.event.frame[:,:,::-1])
                cv2.waitKey(0)

            if len(instance_ids) == 0:
                if not ("Rotate" in action or "MoveAhead" in action or "Look" in action):
                    err = "Bad interact mask. Couldn't locate target object"
                    print("Went through bad interaction mask")
                    success = False
                    rgb = self.event.frame.copy() #shape (h, w, 3)
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                    depth = torch.tensor(self.event.depth_frame.copy()).numpy() #shape (h, w)
                    depth /= 1000.0 #in meters
                    depth = np.expand_dims(depth, 2)

                    state = np.concatenate((rgb, depth), axis = 2).transpose(2, 0, 1)
                    return state, 0, False, self.info, False, None, "", err, None
                else:
                    target_instance_id = ""
            if len(instance_ids) != 0:
                target_instance_id = instance_ids[0]
            
        else:
            target_instance_id = ""

        if debug:
            print("taking action: " + str(action) + " on target_instance_id " + str(target_instance_id))
        try:
            obs, rew, done, infos, event, api_action = self.to_thor_api_exec(action, target_instance_id, smooth_nav)
        except Exception as err:
            success = False
            rgb = self.event.frame.copy() #shape (h, w, 3)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            depth = torch.tensor(self.event.depth_frame.copy()).numpy() #shape (h, w)
            depth /= 1000.0 #in meters
            depth = np.expand_dims(depth, 2)

            state = np.concatenate((rgb, depth), axis = 2).transpose(2, 0, 1)
            return state, 0, False, self.info, success, None, "", err, None

        if not event.metadata['lastActionSuccess']:
            if interact_mask is not None and debug:
                print("Failed to execute action!", action, target_instance_id)
                print("all_ids inside BBox: " + str(all_ids))
                instance_seg = copy.copy(instance_segs)
                instance_seg[:, :, :] = interact_mask[:, :, np.newaxis] == 1
                cv2.imshow('seg', instance_segs)
                cv2.imshow('mask', instance_seg)
                cv2.imshow('full', self.event.frame[:,:,::-1])
                cv2.waitKey(0)
                print(event.metadata['errorMessage'])
            success = False
            if len(event.metadata['errorMessage']) >0:
                print("action that causes below error is ", api_action)
            return obs, rew, done, infos, success, event, target_instance_id, event.metadata['errorMessage'], api_action

        success = True

        return obs, rew, done, infos, success, event, target_instance_id, '', api_action

    @staticmethod
    def bbox_to_mask(bbox):
        return image_util.bbox_to_mask(bbox)

    @staticmethod
    def point_to_mask(point):
        return image_util.point_to_mask(point)

    @staticmethod
    def decompress_mask(compressed_mask):
        return image_util.decompress_mask(compressed_mask)
    
    def get_instance_mask(self):
        return self.event.instance_masks
    
    def close(self):
        self.stop()
