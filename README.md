# FILM: Following Instructions in Language with Modular Methods

[FILM: Following Instructions in Language with Modular Methods](https://arxiv.org/abs/2110.07342)<br />
So Yeon Min, Devendra Singh Chaplot, Pradeep Ravikumar, Yonatan Bisk, Ruslan Salakhutdinov<br />
Carnegie Mellon University, Facebook AI Research

Project Website: https://soyeonm.github.io/FILM_webpage/

![example](./miscellaneous/tests_unseen_33.gif)


## Installing Dependencies

We will also provide docker and singularity setup; see [here](https://github.com/soyeonm/FILM/tree/public/miscellaneous).  

- First download the requirements:

```
$ pip install -r requirements.txt
```

- We use an earlier version of [habitat-lab](https://github.com/facebookresearch/habitat-lab) as specified below:

Installing habitat-lab:
```
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab; git checkout tags/v0.1.5; 
pip install -e .
```

- Install [pytorch](https://pytorch.org/) according to your system configuration. The code is tested on pytorch v1.6.0 and cudatoolkit v10.2. If you are using conda:
```
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 #(Linux with GPU)
conda install pytorch==1.6.0 torchvision==0.7.0 -c pytorch #(Mac OS)
```

- **If you want to visualize semantic segmentation outputs**, install [detectron2](https://github.com/facebookresearch/detectron2/) according to your system configuration. 
(You do **not** need to install this unless you want to visualize segemtnation outputs on the egocentric rgb frame as in the "segmented RGB" [here](https://github.com/soyeonm/FILM/blob/control_helper/miscellaneous/tests_unseen_33.gif)).
If you are using conda:
```
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.6/index.html #(Linux with GPU)
CC=clang CXX=clang++ ARCHFLAGS="-arch x86_64" python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' #(Mac OS)
```

Alternatively, you can use docker or singularity following instructions [here](https://github.com/soyeonm/FILM/tree/public/miscellaneous).

## Important: AI2THOR depth bug fix for MacOS

**If you want to run AI2THOR 2.1.0 on MacOS**, you need to download the data zip of the exactly same version on Linux64.
For AI2THOR 2.1.0, you can simply follow the instructions here.
For example, thor-201909061227-OSXIntel64.zip refers to thor-201909061227-OSXIntel64.zip.

Then unzip the file and replace all the files on MacOS at
```
~/.ai2thor/releases/thor-201909061227-OSXIntel64/thor-201909061227-OSXIntel64.app/Contents/Resources/Data/*
```
with those for Linux64 at
```
thor-201909061227-Linux64_Data/*.
```

Download thor-201909061227-Linux64.zip [here](http://s3-us-west-2.amazonaws.com/ai2-thor/builds/thor-201909061227-Linux64.zip).

## Additional preliminaries to use ALFRED scenes

Please make sure that ai2thor's version is 2.1.0 (if version is newer, ALFRED will break).

1. Download alfred_data_small.zip from [here](https://drive.google.com/file/d/1uujBcxJXzrcQtWZbT91MN8gL4psNBBMO/view?usp=sharing) and unzip it, so that alfred_data_small lives as FILM/alfred_data_small. 

2. Now,
* Go to 'FILM' directory (this repository). 
```
$ export FILM=$(pwd)
```

* Go to a directory you would like to git clone "alfred". Then run,
```
$ git clone https://github.com/askforalfred/alfred.git
$ export ALFRED_ROOT=$(pwd)/alfred
```

* Now run
```
$ cd $ALFRED_ROOT
$ python models/train/train_seq2seq.py --data data/json_feat_2.1.0 --model seq2seq_im_mask --dout exp/model:{model},name:pm_and_subgoals_01 --splits data/splits/oct21.json --gpu --batch 8 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --preprocess
```

The will take 5~15 minutes. You will see this:

<img width="578" alt="스크린샷 2021-04-27 오후 5 51 07" src="https://user-images.githubusercontent.com/77866067/116317384-437fc980-a781-11eb-8c01-f6cdee98f824.png">

Once the bars for preprocessing are all filled, the code will break with an error message. (You can ignore and proceed). 

* Now run,  
```
$ cd $FILM
$ mkdir alfred_data_all 
$ ln -s $ALFRED_ROOT/data/json_2.1.0 $FILM/alfred_data_all
```

## Download Trained models
0. Download "Pretrained_Models_FILM" from [this link](https://drive.google.com/file/d/1mkypSblrc0U3k3kGcuPzVOaY1Rt9Lqpa/view?usp=sharing)

1. Semantic segmentation

```
mv Pretrained_Models_FILM/maskrcnn_alfworld models/segmentation/maskrcnn_alfworld
```

2. Depth prediction

```
mv Pretrained_Models_FILM/depth_models models/depth/depth_models
```

3. Semantic Search Policy

To use the model in the original leaderboard entry (26.49%), 

```
mv Pretrained_Models_FILM/best_model_multi.pt models/semantic_policy/best_model_multi.pt
```

To use a better perfoming model trained with a new seed (27.80%),

```
mv Pretrained_Models_FILM/new_best_model.pt models/semantic_policy/best_model_multi.pt
```

## Run FILM on Valid/ Tests Sets

**Caveat**: Multiprocessing (using --num_processes > 1) will make the construction of semantic mapping slower. We recommend that you use "--num_processes 2" (or a number around 2) and just run several jobs. (E.g. one job with episodes from 0 to 200, another job with episodes from 200 to 400, etc)

**On your laptop (use learned depth and learned segmentation):** 
```
$ python main.py  -n1 --max_episode_length 1000 --num_local_steps 25  --num_processes 2 --eval_split tests_unseen --from_idx 0 --to_idx 120 --max_fails 10    --debug_local   --learned_depth  --use_sem_seg --set_dn first_run   --use_sem_policy  --save_pictures -v 1
```
For example, to use ground truth depth and learned segmentation, run the above command without "--learned_depth". 

```
$ python main.py  -n1 --max_episode_length 1000 --num_local_steps 25  --num_processes 2 --eval_split tests_unseen --from_idx 0 --to_idx 120 --max_fails 10   --debug_local  --use_sem_seg --set_dn first_run   --use_sem_policy  --save_pictures -v 1
```


**On a headless machine with gpu:**

You need to first run a Xserver with 
```
tmux
python alfred_utils/scripts/startx.py 0
```
If you set a Xdisplay other than 0 (if you ran python alfred_utils/scripts/startx.py 1, for example), run
```
export DISPLAY=:1
```
(change 1 accordingly to the Xdisplay you set up.)

Now, get out of tmux and run the following (change DISPLAY with what you set up e.g. 1): 

```
$ python main.py  -n1 --max_episode_length 1000 --num_local_steps 25  --num_processes 2 --eval_split tests_unseen --from_idx 0 --to_idx 120 --x_display DISPLAY  --max_fails 10  --debug_local --learned_depth  --use_sem_seg --which_gpu 1 --sem_gpu_id 0 --sem_seg_gpu 0 --depth_gpu 1 --set_dn first_run   --use_sem_policy  --save_pictures
```
Do not enable "-v 1" on a headless machine.



**Arguments**

--max_episode_length: The episode automatically ends after this number of time steps. 

--num_local_steps: Number of steps by which a new goal is sampled from the random/ semantic search policy. 
  
--num_processes: Number of processes. 

--eval_split: One of valid_unseen, valid_seen, tests_unseen, tests_seen.

--from_idx: The index of episode to start in the "eval_split".

--to_idx: The index of episode to end in the "eval_split" (e.g. "--from_idx 0 --to_idx 1" will run the 0th episode).

--x_display: Set this to the display number you have used for xserver (can use any number on a computer with a monitor).

--max_fails: The episode automatically ends after this number of failed actions.
  
--debug_local: Debug argument that will print statements that can help debugging.

--learned_depth: Use learned depth (ground truth depth used without it).

--use_sem_seg: Use learned segmentation (ground truth segmentation used without it).

--which_gpu, --sem_gpu_id, --sem_seg_gpu, --depth_gpu (not required): Indices of gpus for semantic mapping, semantic search policy, semantic segmentation, depth. If you assign "--use_sem_seg --which_gpu 1 --sem_gpu_id 0 --sem_seg_gpu 0 --depth_gpu 1", gpu's of indices 0 and 1 will get almost equal loads for running 2 processes simultaneously.

--set_dn (not required): Set the "name" of this run. The results will be saved in "/results/" under this name. Pictures will also be saved under this name with the --save_pictures flag.

--use_sem_policy: Use semantic policy (random policy used without this).

--save_pictures: Save the map, fmm_dist (visualization of fast marching method), RGB frame pictures. The pictures will be saved to "pictures/$args.eval_split$/$args.set_dn$"

--appended: Use **low-level language + high-level language**

-v: Visualize (show windows of semantic map/ rgb on the monitor). **Do not use this on headless mode**

## Evaluate results for valid sets
The output of your runs are saved in the pickles of "results/analyze_recs/".
For example, you may see results like the following in your "results/analyze_recs/".
![스크린샷 2022-02-13 오후 3 27 38](https://user-images.githubusercontent.com/77866067/153773547-86239f10-0255-4789-a4ca-4028b7f2f182.png)

Change directory to "results/analyze_recs/" and inside a python3 console,

```
import pickle
result1 = pickle.load(open('valid_unseen_anaylsis_recs_from_0_to_120_errorbar_seed1_1100_0_120.p', 'rb'))
#num of episodes
len(result1)
#num of succeeded epsiodes
sum([s['success'] for s in result1])
```
Aggregate results over multiple runs if you need to.

**Caveat**: The above calculates the ground truth SR for **valid** splits. If you apply this process to **test** splits, you will only get internal/ approximate results. 

## Export results for the [leaderboard](https://leaderboard.allenai.org/alfred/submissions/public)

Run 

```
$ python3 utils/leaderboard_script.py --dn_startswith WHAT_YOUR_PICKLES_START_WITH --json_name DESIRED_JSON_NAME
```

For example, if the "ls" of "results/leaderboard/" looks like:

![스크린샷 2022-02-13 오후 3 11 29](https://user-images.githubusercontent.com/77866067/153772964-0d2d258b-7578-4ce8-84a5-321efbbf731e.png)

you can run 

```
$ python3 utils/leaderboard_script.py --dn_startswith errorbar_seed3_1100 --json_name errorbar_seed3_1100
```

The json is saved in "leaderboard_jsons/"; upload this to the leaderboard. Look inside "utils/leaderboard_script.py" for the arguments.

## Train Models (Mask-RCNN & Depth, BERT/ Semantic Policy)

### Data collection for Mask-RCNN/ Depth
While this repo contains FILM's pre-trained models for Mask-RCNN/ Depth, please respectively refer to repositories of [ALFWORLD](https://github.com/alfworld/alfworld) and [HLSM](https://github.com/valtsblukis/hlsm) for their training.  
To collect RGB/Segmentation mask/Depth mask data as in FILM, follow the protocol below:

Run FILM with ground truth everything and a random semantic policy and at each step,

- Execute the current action 

- With probability of 1/3, 
   - Change horizon 0 and Rotate left 4 times and keep track of the number of objects at each orientation
   - Rotate to the direction with the most # of objects
   - Save RGB/ segmentation mask/ depth mask at horizons 0/ 15
   - With probability of 0.5, save RGB/ segmentation mask/ depth mask at horizons 30/45/60 (0 is when the agent looks straight)
      - If chosen here, rotate 180 degrees and again save RGB/ segmentation mask/ depth mask at horizons 30/45/60 (0 is when the agent looks straight)
   - Come back to the original pose


- Else if the current action is an interaction action, with probability of 1/2,
    - Save RGB/ segmentation mask/ depth mask at horizons 0/15/30/45/60 
    - Make the agent rotate 180 degrees and save RGB/ segmentation mask/ depth mask at horizons 0/15/30/45/60
    - Come back to the original pose and let it follow FILM's alg



### Train BERT models

- Download [data](https://drive.google.com/file/d/1m0q7QYmmhSTOOoS62FdxSSXmQl2TY4gj/view?usp=sharing) and extract it as 
```
models/instructions_processed_LP/BERT/data/alfred_data
```

- Change directory:

```
$ cd models/instructions_processed_LP/BERT 
```

- To train BERT type classification ("_base.pt_"),

```
$ python3 train_bert_base.py -lr 1e-5
```
(Use _--no_appended_ to use high level instructions only for training data.)

- To train BERT argument classification, 

```
$ python3 train_bert_args.py --no_divided_label --task mrecep -lr 5e-4
```
(Use _--no_appended_ to use high level instructions only for training data.)
Similarly, train models for _--task object, --task parent, --task toggle, --task sliced_.

- To generate the finial output for "agent/sem_exp_thor.py",
```
$ python3 end_to_end_outputs.py -sp YOURSPLIT -m MODEL_SAVED_FOLDER_NAME -o OUTPUT_PICKLE_NAME
```
(Again, use _--no_appended_ to use high level instructions only for training data.)


Download pretraind models: 
- [low level instructions only](https://drive.google.com/file/d/1bSBVEdBK5ZcBk8bSOynaFPNZkTtajbWc/view?usp=sharing)
- [low level instructions + high level instructions](https://drive.google.com/file/d/1KQSpEBPd51x7tpF0rE1PVGCliRhuNhue/view?usp=sharing)


### Train the Semantic Policy

Download data from [here](https://drive.google.com/file/d/1TWxKSAxvYKA8hi1RyUgQ0CGmLSe4UR_n/view?usp=sharing) and extract it as "models/semantic_policy/data/maps".
Run 

```
$ python3 models/semantic_policy/train_map_multi.py --eval_freq 50 --dn YOUR_DESIRED_NAME --seed YOUR_SEED --lr 0.001 --num_epochs 1 
```

Look at logs and pick the model with the lowest test loss subject to train loss < 0.62. 

**Load** your model in main.py and run FILM.
If you want to download a smaller version of the data, it is [here](); refer to the README inside to use it.

**Reproduced results on _Tests Unseen_ across multiple runs**

<!-- Model Name | SR on tests unseen | Test Loss | Train Loss
 -->

Model Name | SR with high-level language |  SR with high-level + low-level language
| :--- | ---: | ---: 
Model in ICLR submission  | 24.46 | 26.49
Seed 1 (step 1100)  | 25.51 | 27.86
Seed 2 (step 1300)  | 23.48 | 25.96
Seed 3 (step 1100)  | 23.68 | 25.64
Seed 4 (step 1400)  | 25.18 | 26.62
Avg  |  24.87 | 26.51




## Acknowledgement

Most of [alfred_utils](https://github.com/soyeonm/FILM/control_helper/alfred_utils) comes from [Mohit Shridhar's ALFRED repository](https://github.com/askforalfred/alfred).
[models/depth](https://github.com/soyeonm/FILM/tree/public/models/depth) comes from [Valts Blukis' HLSM](https://github.com/valtsblukis/hlsm).

Code for semantic mapping comes from [Devendra Singh Chaplot's OGN](https://github.com/devendrachaplot/Object-Goal-Navigation)

### Bibtex:
```
@misc{min2021film,
      title={FILM: Following Instructions in Language with Modular Methods}, 
      author={So Yeon Min and Devendra Singh Chaplot and Pradeep Ravikumar and Yonatan Bisk and Ruslan Salakhutdinov},
      year={2021},
      eprint={2110.07342},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

