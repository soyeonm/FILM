#Class for segmentation support


import cv2
import torch
import torchvision.transforms as T
import copy
import numpy as np


import skimage.morphology

from .alfworld_mrcnn import load_pretrained_model 

from . import alfworld_constants
class SemgnetationHelper:
	
	def __init__(self, agent):
		self.agent = agent
		args = self.args = self.agent.args
		self.H = HelperFunctions
		self.transform = T.Compose([T.ToTensor()])

		if args.use_sem_seg:
			self.sem_seg_gpu =  torch.device("cuda:" + str(args.sem_seg_gpu) if args.cuda else "cpu")
			#LARGE
			self.sem_seg_model_alfw_large = load_pretrained_model('models/segmentation/maskrcnn_alfworld/receps_lr5e-3_003.pth', torch.device("cuda:0" if args.cuda else "cpu"), 'recep')
			self.sem_seg_model_alfw_large.eval()
			self.sem_seg_model_alfw_large.to(torch.device("cuda:0" if args.cuda else "cpu"))                  

			#SMALL		    
			self.sem_seg_model_alfw_small = load_pretrained_model('models/segmentation/maskrcnn_alfworld/objects_lr5e-3_005.pth', self.sem_seg_gpu, 'obj')
			self.sem_seg_model_alfw_small.eval()
			self.sem_seg_model_alfw_small.to(self.sem_seg_gpu)

			self.large = alfworld_constants.STATIC_RECEPTACLES
			self.large_objects2idx = {k:i for i, k in enumerate(self.large)}
			self.large_idx2large_object = {v:k for k,v in self.large_objects2idx.items()}

			self.small = alfworld_constants.OBJECTS_DETECTOR
			self.small_objects2idx = {k:i for i, k in enumerate(self.small)}
			self.small_idx2small_object = {v:k for k,v in self.small_objects2idx.items()}

	def update_agent(self, agent):
		self.agent = agent
		self.args = self.agent.args

	#Visualize
	def get_torch_tensor_box(self):
		gt_boxes = self.agent.event.instance_detections2D
		array_list = []
		for k, v in gt_boxes.items():
			category = k.split('|')[0]
			if category in self.agent.total_cat2idx:
				array_list.append(v.tolist())
		return torch.tensor(array_list)

	#Visualize
	def visualize_sem(self, torch_tensor=None):
		from detectron2.utils.visualizer import ColorMode, Visualizer
		from detectron2.structures import Boxes, Instances

		visualizer = Visualizer(self.agent.event.frame, instance_mode=ColorMode.IMAGE)
		if self.args.use_sem_seg:
			v_outputs = Instances(
			image_size=(300,300),
			scores=torch.cat([self.segmented_dict['small']['scores'], self.segmented_dict['large']['scores']], dim=0),
			pred_classes=np.array([self.small_idx2small_object[int(s.item())] for s in self.segmented_dict['small']['classes']] + [self.large_idx2large_object[int(s.item())] for s in self.segmented_dict['large']['classes']]),
			pred_masks=torch.cat([torch.tensor(self.segmented_dict['small']['masks'].astype(bool)), torch.tensor(self.segmented_dict['large']['masks'].astype(bool))], dim=0),
			)
			vis_output = visualizer.draw_instance_predictions(v_outputs.to("cpu"))
		else:
			instances = Instances((300, 300)).to(torch.device("cpu"))
			boxes = Boxes(torch_tensor.to(torch.device("cpu")))
			instances.set('pred_boxes', boxes)
			vis_output = visualizer.draw_instance_predictions(predictions=instances)
		return vis_output.get_image()

	def segmentation_for_map(self):
		small_len = len(self.segmented_dict['small']['scores'])
		large_len = len(self.segmented_dict['large']['scores'])

		semantic_seg = np.zeros((self.args.env_frame_height, self.args.env_frame_width, self.args.num_sem_categories))
		for i in range(small_len):
			category = self.small_idx2small_object[self.segmented_dict['small']['classes'][i].item()]
			v = self.segmented_dict['small']['masks'][i]

			if not(category in self.agent.total_cat2idx):
				pass
			else:
				cat = self.agent.total_cat2idx[category]
				semantic_seg[:, :, cat] +=  v.astype('float')

		for i in range(large_len):
			category = self.large_idx2large_object[self.segmented_dict['large']['classes'][i].item()]
			v = self.segmented_dict['large']['masks'][i]

			if not(category in self.agent.total_cat2idx):
				pass
			else:
				cat = self.agent.total_cat2idx[category]
				semantic_seg[:, :, cat] +=  v.astype('float')

		return semantic_seg

	def get_instance_mask_seg_alfworld_both(self):
		rgb = copy.deepcopy(self.agent.event.frame)

		ims = [rgb]
		im_tensors = [self.transform(i).to(device=self.sem_seg_gpu) if self.args.cuda else self.transform(i) for i in ims]
		results_small = self.sem_seg_model_alfw_small(im_tensors)[0]        

		im_tensors = [self.transform(i).to(device=torch.device("cuda:0" if self.args.cuda else "cpu")) if self.args.cuda else self.transform(i) for i in ims] 
		results_large = self.sem_seg_model_alfw_large(im_tensors)[0]

		desired_classes_small = []
		desired_classes_large = []


		desired_goal_small = []
		desired_goal_large = []
		for cat_name in self.agent.total_cat2idx:
			if not(cat_name in ["None", "fake"]):
				if cat_name in self.large:
					large_class = self.large_objects2idx[cat_name]
					desired_classes_large.append(large_class)
					if cat_name == self.agent.goal_name or (cat_name in self.agent.cat_equate_dict and self.agent.cat_equate_dict[cat_name] == self.agent.goal_name):
						desired_goal_large.append(large_class)

				elif cat_name in self.small:
					small_class = self.small_objects2idx[cat_name]
					desired_classes_small.append(small_class)
					if cat_name == self.agent.goal_name or (cat_name in self.agent.cat_equate_dict and self.agent.cat_equate_dict[cat_name] == self.agent.goal_name):
						desired_goal_small.append(small_class)
				else:
					pass

		desired_goal_small = list(set(desired_goal_small))
		desired_goal_large = list(set(desired_goal_large))

		#FROM here
		indices_small = []     
		indices_large = []    

		for k in range(len(results_small['labels'])):
			if (
				results_small['labels'][k].item() in desired_classes_small
				and results_small['scores'][k] > self.args.sem_seg_threshold_small
			):
				indices_small.append(k)


		for k in range(len(results_large['labels'])):
			if (
				results_large['labels'][k].item() in desired_classes_large
				and results_large['scores'][k] > self.args.sem_seg_threshold_large
			):
				indices_large.append(k)


		#Done until here
		pred_boxes_small=results_small['boxes'][indices_small].detach().cpu()
		pred_classes_small=results_small['labels'][indices_small].detach().cpu()
		pred_masks_small=results_small['masks'][indices_small].squeeze(1).detach().cpu().numpy() #pred_masks[i] has shape (300,300)
		if self.args.with_mask_above_05:
			pred_masks_small = (pred_masks_small>0.5).astype(float)
		pred_scores_small = results_small['scores'][indices_small].detach().cpu()

		for ci in range(len(pred_classes_small)):
			if self.small_idx2small_object[int(pred_classes_small[ci].item())] in self.agent.cat_equate_dict:
				cat = self.small_idx2small_object[int(pred_classes_small[ci].item())]
				pred_classes_small[ci] = self.small_objects2idx[self.agent.cat_equate_dict[cat]]

		pred_boxes_large=results_large['boxes'][indices_large].detach().cpu()
		pred_classes_large=results_large['labels'][indices_large].detach().cpu()
		pred_masks_large=results_large['masks'][indices_large].squeeze(1).detach().cpu().numpy() #pred_masks[i] has shape (300,300)
		if self.args.with_mask_above_05:
			pred_masks_large = (pred_masks_large>0.5).astype(float)
		pred_scores_large = results_large['scores'][indices_large].detach().cpu()


		#Make the above into a dictionary
		self.segmented_dict  = {'small': {'boxes': pred_boxes_small,
											'classes': pred_classes_small,
											'masks': pred_masks_small,
											'scores': pred_scores_small
								},
								'large':{'boxes': pred_boxes_large,
										'classes': pred_classes_large,
										'masks': pred_masks_large,
										'scores': pred_scores_large}}

	#Get pseudo instance names as keys
	def segmentation_ground_truth(self):
		instance_mask = self.agent.event.instance_masks
		semantic_seg = np.zeros((self.args.env_frame_height, self.args.env_frame_width, self.args.num_sem_categories))
		
		if self.args.ignore_categories:
			for k, v in instance_mask.items():
				category = k.split('|')[0]
				category_last = k.split('|')[-1]
				if 'Sliced' in category_last:
					category = category + 'Sliced'

				if 'Sink' in category and 'SinkBasin' in category_last:
					category =  'SinkBasin' 
				if 'Bathtub' in category and 'BathtubBasin' in category_last:
					category =  'BathtubBasin'
				
				if not(category in self.agent.total_cat2idx):
					pass
				else:
					cat = self.agent.total_cat2idx[category]
					semantic_seg[:, :, cat] =  v.astype('float')

		else:
			for k, v in instance_mask.items():
				category = k.split('|')[0]
				try:
					cat = self.agent.total_cat2idx[category]
					success = 1
				except:
					success = 0
				
				if success ==1:
					semantic_seg[:, :, cat] =  v.astype('float')
				else:
					pass
		
		return semantic_seg.astype('uint8')

	def get_sem_pred(self, rgb):
		if not(self.args.use_sem_seg):
			semantic_pred = self.segmentation_ground_truth() #(300, 300, num_cat)
			if self.args.visualize or self.args.save_pictures:
				torch_tensor = self.get_torch_tensor_box()
				sem_vis = self.visualize_sem(torch_tensor)

		else:
			#1. First get dict
			self.get_instance_mask_seg_alfworld_both()

			#2. Get segmentation for map making
			semantic_pred = self.segmentation_for_map()

			#3. visualize (get sem_vis)
			if self.args.visualize or self.args.save_pictures:
				sem_vis = self.visualize_sem()

			if (self.agent.pointer==2  or (self.agent.pointer==1 and self.agent.last_success and self.agent.last_action_ogn == "PutObject"))and self.agent.task_type == 'pick_two_obj_and_place':
				small_len = len(self.segmented_dict['small']['scores'])
				large_len = len(self.segmented_dict['large']['scores'])

				wheres = []
				for i in range(large_len):
					category = self.large_idx2large_object[self.segmented_dict['large']['classes'][i].item()]
					v = self.segmented_dict['large']['boxes'][i]

					if category == self.agent.actions_dict['list_of_actions'][1][0]:
						wheres.append(v)

				avoid_idx = []
				for i in range(small_len):
					category = self.small_idx2small_object[self.segmented_dict['small']['classes'][i].item()]
					v = self.segmented_dict['small']['boxes'][i]
					if category == self.agent.actions_dict['list_of_actions'][0][0]:
						for where in wheres:
							#if np.sum(v[where]) == len(np.where(v)[0]):
							if v[0]>= where[0] and v[1] >= where[1] and v[2] <= where[2] and v[3]<=where[3]:
								avoid_idx.append(i)

				avoid_idx = list(set(avoid_idx))
				incl_idx = [i for i in range(small_len) if not(i in avoid_idx)]


				self.segmented_dict['small']['boxes'] = self.segmented_dict['small']['boxes'][incl_idx]
				self.segmented_dict['small']['classes'] = self.segmented_dict['small']['classes'][incl_idx]
				self.segmented_dict['small']['masks'] = self.segmented_dict['small']['masks'][incl_idx]
				self.segmented_dict['small']['scores'] = self.segmented_dict['small']['scores'][incl_idx]
				

		if self.args.visualize:
			sem_vis = cv2.resize(sem_vis, (500, 500)) 
			cv2.imshow("Sem", sem_vis)
			cv2.waitKey(1)

		if self.args.save_pictures:
			cv2.imwrite(self.agent.picture_folder_name + "Sem/" + "Sem_" + str(self.agent.steps_taken) + ".png", sem_vis)  
		semantic_pred = semantic_pred.astype(np.float32)

		return semantic_pred

	def get_instance_mask_from_obj_type(self, object_type):
		mask = np.zeros((300, 300))
		for k, v in self.agent.event.instance_masks.items():
			category = k.split('|')[0]
			category_last = k.split('|')[-1]
			if category in self.agent.cat_equate_dict:
				category = self.agent.cat_equate_dict[category]
			if 'Sliced' in category_last:
				category = category + 'Sliced'
			if 'Sink' in category and 'SinkBasin' in category_last:
				category =  'SinkBasin' 
			if 'Bathtub' in category and 'BathtubBasin' in category_last:
				category =  'BathtubBasin'
			if category == object_type:
				mask = v
		if np.sum(mask) == 0:
			mask = None
		return mask

	def get_instance_mask_from_obj_type_largest(self, object_type):
		mask = np.zeros((300, 300))
		max_area = 0
		for k, v in self.agent.event.instance_masks.items():
			category = k.split('|')[0]
			category_last = k.split('|')[-1]
			if category in self.agent.cat_equate_dict:
				category = self.agent.cat_equate_dict[category]
			if 'Sliced' in category_last:
				category = category + 'Sliced'
			if 'Sink' in category and 'SinkBasin' in category_last:
				category =  'SinkBasin' 
			if 'Bathtub' in category and 'BathtubBasin' in category_last:
				category =  'BathtubBasin'
			if category == object_type:
				if np.sum(v)>= max_area:
					max_area = np.sum(v)
					mask = v
		if np.sum(mask) == 0:
			mask = None
		return mask


	def sem_seg_get_instance_mask_from_obj_type_largest_only(self, object_type):
		mask = np.zeros((300, 300))
		small_len = len(self.segmented_dict['small']['scores'])
		large_len = len(self.segmented_dict['large']['scores'])
		max_area = -1

		if object_type in self.agent.cat_equate_dict:
			object_type = self.agent.cat_equate_dict[object_type]

		if object_type in self.large_objects2idx:
			#Get the highest score
			for i in range(large_len):
				category = self.large_idx2large_object[self.segmented_dict['large']['classes'][i].item()]
				if category == object_type:
					v = self.segmented_dict['large']['masks'][i]
					score = self.segmented_dict['large']['scores'][i]
					area = np.sum(self.segmented_dict['large']['masks'][i])
					max_area = max(area, max_area)
					if max_area == area:    
						mask =  v.astype('float')

		else:
			for i in range(small_len):
				category = self.small_idx2small_object[self.segmented_dict['small']['classes'][i].item()]
				if category == object_type:
					v = self.segmented_dict['small']['masks'][i]
					score = self.segmented_dict['small']['scores'][i]
					area = np.sum(self.segmented_dict['small']['masks'][i])
					max_area = max(area, max_area)
					if max_area == area:   
						mask =  v.astype('float')

		if np.sum(mask) == 0:
			mask = None
		return mask


	def sem_seg_get_instance_mask_from_obj_type(self, object_type):
		mask = np.zeros((300, 300))
		small_len = len(self.segmented_dict['small']['scores'])
		large_len = len(self.segmented_dict['large']['scores'])
		max_score = -1

		if object_type in self.agent.cat_equate_dict:
			object_type = self.agent.cat_equate_dict[object_type]

		if object_type in self.large_objects2idx:
			#Get the highest score
			for i in range(large_len):
				category = self.large_idx2large_object[self.segmented_dict['large']['classes'][i].item()]
				if category == object_type:
					v = self.segmented_dict['large']['masks'][i]
					score = self.segmented_dict['large']['scores'][i]
					max_score = max(score, max_score)
					if score == max_score:    
						mask =  v.astype('float')

		else:
			for i in range(small_len):
				category = self.small_idx2small_object[self.segmented_dict['small']['classes'][i].item()]
				if category == object_type:
					v = self.segmented_dict['small']['masks'][i]
					score = self.segmented_dict['small']['scores'][i]
					max_score = max(score, max_score)
					if score == max_score:
						mask =  v.astype('float')

		if np.sum(mask) == 0:
			mask = None
		return mask

	def get_target(self, interact_mask, mask_px_sample=1):
		if interact_mask is None:
			return ""
		instance_segs = np.array(self.agent.event.instance_segmentation_frame)
		color_to_object_id = self.agent.event.color_to_object_id

		# get object_id for each 1-pixel in the interact_mask
		nz_rows, nz_cols = np.nonzero(interact_mask)
		instance_counter = Counter()
		for i in range(0, len(nz_rows), mask_px_sample):
			x, y = nz_rows[i], nz_cols[i]
			instance = tuple(instance_segs[x, y])
			instance_counter[instance] += 1

		# iou scores for all instances
		iou_scores = {}
		for color_id, intersection_count in instance_counter.most_common():
			union_count = np.sum(np.logical_or(np.all(instance_segs == color_id, axis=2), interact_mask.astype(bool)))
			iou_scores[color_id] = intersection_count / float(union_count)
		iou_sorted_instance_ids = list(OrderedDict(sorted(iou_scores.items(), key=lambda x: x[1], reverse=True)))

		# get the most common object ids ignoring the object-in-hand
		inv_obj = self.agent.event.metadata['inventoryObjects'][0]['objectId'] \
			if len(self.agent.event.metadata['inventoryObjects']) > 0 else None
		all_ids = [color_to_object_id[color_id] for color_id in iou_sorted_instance_ids
				   if color_id in color_to_object_id and color_to_object_id[color_id] != inv_obj]

		# print all ids

		# print instance_ids
		instance_ids = [inst_id for inst_id in all_ids if inst_id is not None]
		instance_ids = self.agent.prune_by_any_interaction(instance_ids)
		if self.args.ground_truth_segmentation:
			pass
		else:
			instance_ids_new = self.agent.prune_by_any_interaction(instance_ids)
			for instance_id in instance_ids:
				if 'Sink' in instance_id:
					instance_ids_new.append(instance_id)
		# prune invalid instances like floors, walls, etc. 
		if len(instance_ids) == 0:
			return ""
		else:
			target_instance_id = instance_ids[0]
			return target_instance_id

	def _visible_objects(self):
		objects_meta_data = self.agent.event.metadata["objects"]
		objectIDs = []
		for i, obj_dict in enumerate(objects_meta_data):
			if obj_dict['visible']:
				objectID = obj_dict['objectId']
				objectIDs.append(objectID)
		return objectIDs

class HelperFunctions:
	def diff_two_frames(f1, f2):
		diff1 = np.where(f1[:, :, 0] != f2[:, :, 0])
		diff2 = np.where(f1[:, :, 1] != f2[:, :, 1])
		diff3 = np.where(f1[:, :, 2] != f2[:, :, 2])
		diff_mask = np.zeros((300,300)); return_mask = np.zeros((300,300))
		diff_mask[diff1] =1.0 
		diff_mask[diff2] =1.0 
		diff_mask[diff3] =1.0 

		#Divide diff mask into 2 regions with scikit image
		connected_regions = skimage.morphology.label(diff_mask, connectivity=2)
		unique_labels = [i for i in range(0, np.max(connected_regions)+1)]
		lab_area = {lab:0 for lab in unique_labels}
		min_ar = 10000000
		smallest_lab = None
		for lab in unique_labels:
			wheres = np.where(connected_regions == lab)
			lab_area[lab] = np.sum(len(wheres[0]))
			if lab_area[lab] > 100:
				min_ar = min(lab_area[lab], min_ar)
				if min_ar == lab_area[lab]:
					smallest_lab = lab

		if not(smallest_lab) is None:
			return_mask[np.where(connected_regions == smallest_lab)] = 1


		#return_mask[np.where(picked_up_mask)] = 0.0
		return return_mask

		return None, vis_output.get_image()

	def diff_two_frames_max(f1, f2):
		diff1 = np.where(f1[:, :, 0] != f2[:, :, 0])
		diff2 = np.where(f1[:, :, 1] != f2[:, :, 1])
		diff3 = np.where(f1[:, :, 2] != f2[:, :, 2])
		diff_mask = np.zeros((300,300)); return_mask = np.zeros((300,300))
		diff_mask[diff1] =1.0 
		diff_mask[diff2] =1.0 
		diff_mask[diff3] =1.0 

		#Divide diff mask into 2 regions with scikit image
		connected_regions = skimage.morphology.label(diff_mask, connectivity=2)
		unique_labels = [i for i in range(0, np.max(connected_regions)+1)]
		lab_area = {lab:0 for lab in unique_labels}
		max_ar = 0
		largest_lab = None
		for lab in unique_labels:
			wheres = np.where(connected_regions == lab)
			lab_area[lab] = np.sum(len(wheres[0]))
			if lab_area[lab] > 100:
				max_ar = max(lab_area[lab], max_ar)
				if max_ar == lab_area[lab]:
					largest_lab = lab

		if not(largest_lab) is None:
			return_mask[np.where(connected_regions == largest_lab)] = 1

		return return_mask