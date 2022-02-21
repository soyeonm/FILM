# Docker and Singularity Instructions:

We provide [docker](https://www.docker.com/) and [singularity](https://sylabs.io/) images with all the dependencies installed.
Following either of Using Docker or Using Singularity, you can skip [Installing Dependencies](https://github.com/soyeonm/FILM/blob/public/README.md#installing-dependencies) of the original [ReadME](https://github.com/soyeonm/FILM/blob/public/README.md#additional-preliminaries-to-use-alfred-scenes). You still have to follow all other section of the original [ReadME](https://github.com/soyeonm/FILM/blob/public/README.md#additional-preliminaries-to-use-alfred-scenes).

We note that the docker/ singuarity images we provide were built based on the docker support of the original [ALFRED repository](https://github.com/askforalfred/alfred#docker-setup).

## Using Docker 

0. Download this repository first with 
```
git clone https://github.com/soyeonm/FILM.git
cd FILM/;
```

1. Pull the docker image with 
```
docker pull symin95/combined_alfred_ogn:latest
```

2. After pulling the docker image, modify 

```
cmd += " -v FILM_PATH:PATH_INSIDE_DOCKER_CONTAINER"
```
of line 66 in alfred_utils/scripts/docker_run.py.

e.g. 
```
cmd += "-v /usr0/soyeonm/raid/FILM:/home/soyeonm/FILM"
```

3. Now run a docker continaer using:
```
python alfred_utils/scripts/docker_run.py -i b885cd0b71bd  -c container1 --headless 
```
Attach the --headless argument if you are running on a **headless** machine. 

**Inside** the docker container, follow instructions from [the rest of the ReadME](https://github.com/soyeonm/FILM/blob/public/README.md#additional-preliminaries-to-use-alfred-scenes).
For headless machines, the Xserver has to be turned on **inside** the docker container.

## Using Singularity

Note that with headless nodes, only 1 Xserver can be turned on per node on singularity. This is because for singualrity, the Xserver has to be turned on **outside** the conatainer in many cases. If your cluster allows the Xserver running inside each container, you can use multiple Xservers per node. 

0. Download this repository first with 
```
git clone https://github.com/soyeonm/FILM.git
cd FILM/;
```

1. Pull the singualrity image with
```
singularity pull docker://symin95/combined_alfred_ogn:latest
```

2. **Outside** the singularity container, 
```
TMUX=tmux; tmux
python alfred_utils/scripts/startx.py 0
```
As in [here](https://github.com/soyeonm/FILM/tree/public#run-film-on-valid-tests-sets), if you use an XDisplay that is not 0 (e.g. python alfred_utils/scripts/startx.py 1), then make sure to additionally run 

```
export DISPLAY=:1
```  
with the exported DISPLAY being whatever display number you used for startx.py. 

3. Now, go to where you donwloaded the singularity image and run the singularity container with 
```
/opt/singularity/bin/singularity  shell --nv   --cleanenv may30_gcc_2_latest.sif
```

or 

```
module load singularity
singularity  shell --nv   --cleanenv may30_gcc_2_latest.sif
```

depending on your system. 

Go to [this part of the ReadME](https://github.com/soyeonm/FILM/blob/public/README.md#additional-preliminaries-to-use-alfred-scenes) and follow instructions from here, except for the part on **setting up Xserver**
