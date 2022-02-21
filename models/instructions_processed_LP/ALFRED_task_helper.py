#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 16:49:38 2021

@author: soyeonmin
"""
import pickle
import alfred_utils.gen.constants as constants
import string

exclude = set(string.punctuation)
task_type_dict = {2: 'pick_and_place_simple',
 5: 'look_at_obj_in_light',
 1: 'pick_and_place_with_movable_recep',
 3: 'pick_two_obj_and_place',
 6: 'pick_clean_then_place_in_recep',
 4: 'pick_heat_then_place_in_recep',
 0: 'pick_cool_then_place_in_recep'}


def read_test_dict(test, appended, unseen):
    if test:
        if appended:
            if unseen:
                return pickle.load(open("models/instructions_processed_LP/instruction2_params_test_unseen_appended.p", "rb"))
            else:
                return pickle.load(open("models/instructions_processed_LP/instruction2_params_test_seen_appended.p", "rb"))
        else:
            if unseen:
                return pickle.load(open("models/instructions_processed_LP/instruction2_params_test_unseen_916_noappended.p", "rb"))
            else:
                return pickle.load(open("models/instructions_processed_LP/instruction2_params_test_seen_916_noappended.p", "rb"))
    else:
        if appended:
            if unseen:
                return pickle.load(open("models/instructions_processed_LP/instruction2_params_val_unseen_appended.p", "rb"))
            else:
                return pickle.load(open("models/instructions_processed_LP/instruction2_params_val_seen_appended.p", "rb"))
        else:
            if unseen:
                return pickle.load(open("models/instructions_processed_LP/instruction2_params_val_unseen_916_noappended.p", "rb"))
            else:
                return pickle.load(open("models/instructions_processed_LP/instruction2_params_val_seen_916_noappended.p", "rb"))

def exist_or_no(string):
    if string == '' or string == False:
        return 0
    else:
        return 1

def none_or_str(string):
    if string == '':
        return None
    else:
        return string

def get_arguments_test(test_dict, instruction):
    task_type, mrecep_target, object_target, parent_target, sliced = \
        test_dict[instruction]['task_type'],  test_dict[instruction]['mrecep_target'], test_dict[instruction]['object_target'], test_dict[instruction]['parent_target'],\
             test_dict[instruction]['sliced']

    if isinstance(task_type, int):
        task_type = task_type_dict[task_type]
    return instruction, task_type, mrecep_target, object_target, parent_target, sliced 
        

def get_arguments(traj_data):
    task_type = traj_data['task_type']
    try:
        r_idx = traj_data['repeat_idx']
    except:
        r_idx = 0
    language_goal_instr = traj_data['turk_annotations']['anns'][r_idx]['task_desc']
    
    sliced = exist_or_no(traj_data['pddl_params']['object_sliced'])
    mrecep_target = none_or_str(traj_data['pddl_params']['mrecep_target'])
    object_target = none_or_str(traj_data['pddl_params']['object_target'])
    parent_target = none_or_str(traj_data['pddl_params']['parent_target'])
    #toggle_target = none_or_str(traj_data['pddl_params']['toggle_target'])
    
    return language_goal_instr, task_type, mrecep_target, object_target, parent_target, sliced

def add_target(target, target_action, list_of_actions):
    if target in [a for a in constants.OPENABLE_CLASS_LIST if not(a == 'Box')]:
        list_of_actions.append((target, "OpenObject"))
    list_of_actions.append((target, target_action))
    if target in [a for a in constants.OPENABLE_CLASS_LIST if not(a == 'Box')]:
        list_of_actions.append((target, "CloseObject"))
    return list_of_actions

def determine_consecutive_interx(list_of_actions, previous_pointer, sliced=False):
    returned, target_instance = False, None
    if previous_pointer <= len(list_of_actions)-1:
        if list_of_actions[previous_pointer][0] == list_of_actions[previous_pointer+1][0]:
            returned = True
            #target_instance = list_of_target_instance[-1] #previous target
            target_instance = list_of_actions[previous_pointer][0]
        #Micorwave or Fridge
        elif list_of_actions[previous_pointer][1] == "OpenObject" and list_of_actions[previous_pointer+1][1] == "PickupObject":
            returned = True
            #target_instance = list_of_target_instance[0]
            target_instance = list_of_actions[0][0]
            if sliced:
                #target_instance = list_of_target_instance[3]
                target_instance = list_of_actions[3][0]
        #Micorwave or Fridge
        elif list_of_actions[previous_pointer][1] == "PickupObject" and list_of_actions[previous_pointer+1][1] == "CloseObject":
            returned = True
            #target_instance = list_of_target_instance[-2] #e.g. Fridge
            target_instance = list_of_actions[previous_pointer-1][0]
        #Faucet
        elif list_of_actions[previous_pointer+1][0] == "Faucet" and list_of_actions[previous_pointer+1][1] in ["ToggleObjectOn", "ToggleObjectOff"]:
            returned = True
            target_instance = "Faucet"
        #Pick up after faucet 
        elif list_of_actions[previous_pointer][0] == "Faucet" and list_of_actions[previous_pointer+1][1] == "PickupObject":
            returned = True
            #target_instance = list_of_target_instance[0]
            target_instance = list_of_actions[0][0]
            if sliced:
                #target_instance = list_of_target_instance[3]
                target_instance = list_of_actions[3][0]
    return returned, target_instance
            
def get_list_of_highlevel_actions(traj_data, test=False, test_dict=None, args_nonsliced=False, appended=False):
    if not(test):
        language_goal, task_type, mrecep_target, obj_target, parent_target,  sliced = get_arguments(traj_data)
    if test:
        r_idx = traj_data['repeat_idx']
        instruction = traj_data['turk_annotations']['anns'][r_idx]['task_desc']
        #if appended:
        instruction = instruction.lower()
        instruction = ''.join(ch for ch in instruction if ch not in exclude)
        language_goal, task_type, mrecep_target, obj_target, parent_target,  sliced  = get_arguments_test(test_dict, instruction)
            
    #obj_target = 'Tomato'
    #mrecep_target = "Plate"
    if parent_target == "Sink":
        parent_target = "SinkBasin"
    if parent_target == "Bathtub":
        parent_target = "BathtubBasin"
    
    #Change to this after the sliced happens
    if args_nonsliced:
        if sliced == 1:
            obj_target = obj_target +'Sliced'
        #Map sliced as the same place in the map, but like "|SinkBasin" look at the objectid

    
    categories_in_inst = []
    list_of_highlevel_actions = []
    second_object = []
    caution_pointers = []
    #obj_target = "Tomato"

    #if sliced:
    #    obj_target = obj_target +'Sliced'

    if sliced == 1:
       list_of_highlevel_actions.append(("Knife", "PickupObject"))
       list_of_highlevel_actions.append((obj_target, "SliceObject"))
       caution_pointers.append(len(list_of_highlevel_actions))
       list_of_highlevel_actions.append(("SinkBasin", "PutObject"))
       categories_in_inst.append(obj_target)
       
    if sliced:
        obj_target = obj_target +'Sliced'

    
    if task_type == 'pick_cool_then_place_in_recep': #0 in new_labels 
       list_of_highlevel_actions.append((obj_target, "PickupObject"))
       caution_pointers.append(len(list_of_highlevel_actions))
       list_of_highlevel_actions = add_target("Fridge", "PutObject", list_of_highlevel_actions)
       list_of_highlevel_actions.append(("Fridge", "OpenObject"))
       list_of_highlevel_actions.append((obj_target, "PickupObject"))
       list_of_highlevel_actions.append(("Fridge", "CloseObject"))
       caution_pointers.append(len(list_of_highlevel_actions))
       list_of_highlevel_actions = add_target(parent_target, "PutObject", list_of_highlevel_actions)
       categories_in_inst.append(obj_target)
       categories_in_inst.append("Fridge")
       categories_in_inst.append(parent_target)
       
    elif task_type == 'pick_and_place_with_movable_recep': #1 in new_labels
        list_of_highlevel_actions.append((obj_target, "PickupObject"))
        caution_pointers.append(len(list_of_highlevel_actions))
        list_of_highlevel_actions = add_target(mrecep_target, "PutObject", list_of_highlevel_actions)
        list_of_highlevel_actions.append((mrecep_target, "PickupObject"))
        caution_pointers.append(len(list_of_highlevel_actions))
        list_of_highlevel_actions = add_target(parent_target, "PutObject", list_of_highlevel_actions)
        categories_in_inst.append(obj_target)
        categories_in_inst.append(mrecep_target)
        categories_in_inst.append(parent_target)
    
    elif task_type == 'pick_and_place_simple':#2 in new_labels 
        list_of_highlevel_actions.append((obj_target, "PickupObject"))
        caution_pointers.append(len(list_of_highlevel_actions))
        list_of_highlevel_actions = add_target(parent_target, "PutObject", list_of_highlevel_actions)
        #list_of_highlevel_actions.append((parent_target, "PutObject"))
        categories_in_inst.append(obj_target)
        categories_in_inst.append(parent_target)
        
    
    elif task_type == 'pick_heat_then_place_in_recep': #4 in new_labels
        list_of_highlevel_actions.append((obj_target, "PickupObject"))  
        caution_pointers.append(len(list_of_highlevel_actions))
        list_of_highlevel_actions = add_target("Microwave", "PutObject", list_of_highlevel_actions)
        list_of_highlevel_actions.append(("Microwave", "ToggleObjectOn" ))
        list_of_highlevel_actions.append(("Microwave", "ToggleObjectOff" ))
        list_of_highlevel_actions.append(("Microwave", "OpenObject"))
        list_of_highlevel_actions.append((obj_target, "PickupObject"))
        list_of_highlevel_actions.append(("Microwave", "CloseObject"))
        caution_pointers.append(len(list_of_highlevel_actions))
        list_of_highlevel_actions = add_target(parent_target, "PutObject", list_of_highlevel_actions)
        categories_in_inst.append(obj_target)
        categories_in_inst.append("Microwave")
        categories_in_inst.append(parent_target)
        
    elif task_type == 'pick_two_obj_and_place': #3 in new_labels
        list_of_highlevel_actions.append((obj_target, "PickupObject"))  
        caution_pointers.append(len(list_of_highlevel_actions))
        list_of_highlevel_actions = add_target(parent_target, "PutObject", list_of_highlevel_actions)
        if parent_target in constants.OPENABLE_CLASS_LIST:
            second_object = [False] * 4
        else:
            second_object = [False] * 2
        if sliced:
            second_object = second_object + [False] * 3
        caution_pointers.append(len(list_of_highlevel_actions))
        list_of_highlevel_actions.append((obj_target, "PickupObject"))  
        #caution_pointers.append(len(list_of_highlevel_actions))
        second_object.append(True)
        caution_pointers.append(len(list_of_highlevel_actions))
        list_of_highlevel_actions = add_target(parent_target, "PutObject", list_of_highlevel_actions)
        second_object.append(False)
        categories_in_inst.append(obj_target)
        categories_in_inst.append(parent_target)
        
        
    elif task_type == 'look_at_obj_in_light': #5 in new_labels
        list_of_highlevel_actions.append((obj_target, "PickupObject"))
        #if toggle_target == "DeskLamp":
        #    print("Original toggle target was DeskLamp")
        toggle_target = "FloorLamp"
        list_of_highlevel_actions.append((toggle_target, "ToggleObjectOn" ))
        categories_in_inst.append(obj_target)
        categories_in_inst.append(toggle_target)
        
    elif task_type == 'pick_clean_then_place_in_recep': #6 in new_labels
        list_of_highlevel_actions.append((obj_target, "PickupObject"))  
        caution_pointers.append(len(list_of_highlevel_actions))
        list_of_highlevel_actions.append(("SinkBasin", "PutObject"))  #Sink or SinkBasin? 
        list_of_highlevel_actions.append(("Faucet", "ToggleObjectOn"))
        list_of_highlevel_actions.append(("Faucet", "ToggleObjectOff"))
        list_of_highlevel_actions.append((obj_target, "PickupObject"))  
        caution_pointers.append(len(list_of_highlevel_actions))
        list_of_highlevel_actions = add_target(parent_target, "PutObject", list_of_highlevel_actions)
        categories_in_inst.append(obj_target)
        categories_in_inst.append("SinkBasin")
        categories_in_inst.append("Faucet")
        categories_in_inst.append(parent_target)
    else:
        raise Exception("Task type not one of 0, 1, 2, 3, 4, 5, 6!")

    if sliced == 1:
       if not(parent_target == "SinkBasin"):
            categories_in_inst.append("SinkBasin")
    
    #return [(goal_category, interaction), (goal_category, interaction), ...]
    print("instruction goal is ", language_goal)
    #list_of_highlevel_actions = [ ('Microwave', 'OpenObject'), ('Microwave', 'PutObject'), ('Microwave', 'CloseObject')]
    #list_of_highlevel_actions = [('Microwave', 'OpenObject'), ('Microwave', 'PutObject'), ('Microwave', 'CloseObject'), ('Microwave', 'ToggleObjectOn'), ('Microwave', 'ToggleObjectOff'), ('Microwave', 'OpenObject'), ('Apple', 'PickupObject'), ('Microwave', 'CloseObject'), ('Fridge', 'OpenObject'), ('Fridge', 'PutObject'), ('Fridge', 'CloseObject')]
    #categories_in_inst = ['Microwave', 'Fridge']
    return list_of_highlevel_actions, categories_in_inst, second_object, caution_pointers