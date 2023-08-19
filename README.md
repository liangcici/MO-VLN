# MO-VLN

[![arXiv](https://img.shields.io/badge/arXiv-2307.04725-b31b1b.svg)](https://arxiv.org/abs/2306.10322)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://mligg23.github.io/MO-VLN-Site/index.html)

This repository is the official PyTorch implementation of [MO-VLN](https://arxiv.org/abs/2306.10322).

[MO-VLN: A Multi-Task Benchmark for Open-set Zero-Shot Vision-and-Language Navigation](https://arxiv.org/abs/2306.10322)
</br>
Xiwen Liang*, 
Liang Ma*, 
Shanshan Guo, 
Jianhua Han, 
Hang Xu, 
Shikui Ma, 
Xiaodan Liang<sup>$\dagger$</sup>
<p style="font-size: 0.8em; margin-top: -1em">*Equal contribution <sup>$\dagger$</sup>Corresponding Author</p>


### Update


<details>
<summary>🚀🚀[8/17/2023]v0.2.0: More assets!2 new scenes,50 new walkers,954 new objects,1k+ new instructions</summary>
 
 We have released [![version](https://img.shields.io/badge/version-0.2.0-blue)](https://drive.google.com/drive/folders/1padFHXi9VrTfDR2_8UmxB8NyZf2NfiZB?usp=drive_link) of the MO-VLN benchmark simulator.

- Support for **grabbing and navigation tasks**.
- Added many different walker states, including **50 unique walkers across gender, skin color, and age groups, with smooth walking or running motions**.
- Added **walker control interface**.This interface supports:
  - Selecting the walker type to generate
  - Specifying where walkers are generated
  - Setting whether they move freely
  - Controlling the speed of their movement
- **Added 1k+ instructions** to our four tasks.
- We modeled an **additional 954 classes of models** to construct the indoor scene.
- Two **new scenes have been added**, bringing the total to five:
  - Coffee
  - Restaurant
  - Nursing Room
  - **Home scene** -- A home suite consisting of a living room, kitchen, dining room, and multiple bedrooms
  - **Separate tables** -- Multiple tables can provide a large and efficient grasping parallel training
</details>

<details>
<summary>[6/18/2023]v0.1.0: 3 scenes,2165 objects, real light and shadow characteristics</summary>
 
 We have released [![version](https://img.shields.io/badge/version-0.1.0-blue)](https://drive.google.com/drive/folders/1PijMeLZV6OUvB7HZIJph0bbsMfZWx9YJ?usp=drive_link) of the MO-VLN benchmark simulator.

- Built on UE5.
- 3 scene types:
  - Coffee -- Modelled on a 1:1 ratio to a Coffee
  - Restaurant -- Modelled on a 1:1 ratio to a restaurant
  - Nursing Room -- Modelled on a 1:1 ratio to a Nursing Room
- We handcrafted **2,165 classes of models** at a 1:1 ratio to real-life scenarios in order to construct these three scenes. These three scenes were ultimately constructed from a total of **4,230 models**.
- We selected **129 representative classes** from the models built and supported **navigation testing**.
- With **real light and shadow characteristics**
- Support instruction tasks with **four tasks**: 
  - goal-conditioned navigation given a specific object category (e.g., "fork"); 
  - goal-conditioned navigation given simple instructions (e.g., "Search for and move towards a tennis ball"); 
  - step-by-step instructions following; 
  - finding abstract objects based on high-level instruction (e.g., "I am thirsty").
</details>

<details>
<summary>To-Do List</summary>

 - Provide **more classes of generative objects**.
- **10+ scenes** are under construction and will be updated successively in the future.
- Generate more high-quality instruction-ground truth pairs for the newly constructed scenes.
- Continue to update the simulator's **physics engine effects** to achieve more **realistic dexterous hand-grabbing effects**
- Adding **more interactive properties to objects in the environment**, such as a coffee machine that can be controlled to make coffee.
- **Construct complex tasks involving combined navigation and grasping.**
</details>

## Overview
MO-VLN provides four tasks: 1) goal-conditioned navigation given a specific object category (e.g., "fork"); 2) goal-conditioned navigation given simple instructions (e.g., "Search for and move towards a tennis ball"); 3) step-by-step instruction following; 4) finding abstract object based on high-level instruction (e.g., "I am thirsty"). The earlier version of our simulator covers three high-quality scenes: cafe, restaurant, and nursing house.

![scene](./docs/scenes.png)

![task](./docs/tasks.png)


## Installing Dependencies
- Installing the simulator following [here](https://mligg23.github.io/MO-VLN-Site/Simulation%20Environment%20API.html).

- Installing [GLIP](https://github.com/microsoft/GLIP).

- Installing [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything).


## Setup
Clone the repository and install other requirements:
```
git clone https://github.com/liangcici/MO-VLN.git
cd MO-VLN/
pip install -r requirements.txt
```

### Setting up the dataset
- Downloading original datasets from [here](https://drive.google.com/drive/folders/1khtQ9zRfWQX0WtsMWq3NkRNMvjH0JiZi).

- Generate data for ObjectNav (goal-conditioned navigation given a specific object category).
```
python data_preprocess/gen_objectnav.py --map_id 3
```
map_id indicates specific scene: `{3: Starbucks; 4: TG; 5: NursingRoom}`.


## Usage
The implementation is based on frontier-based exploration (FBE). Exploration with commonsense knowledge as in our paper is based on [ESC](https://sites.google.com/ucsc.edu/escnav/home), which is not allowed to be released. `dataset/objectnav/*.npy` are knowledge extracted from LLMs, and can be used to reproduce exploration with commonsense knowledge.

Run models with FBE:

- For ObjectNav:
```
python zero_shot_eval.py --sem_seg_model_type glip --map_id 3
```


## Related Projects
- The Semantic Mapping module is based on [SemExp](https://github.com/devendrachaplot/Object-Goal-Navigation).


## Citation
```
@article{liang2023mo,
  title={MO-VLN: A Multi-Task Benchmark for Open-set Zero-Shot Vision-and-Language Navigation},
  author={Liang, Xiwen and Ma, Liang and Guo, Shanshan and Han, Jianhua and Xu, Hang and Ma, Shikui and Liang, Xiaodan},
  journal={arXiv preprint arXiv:2306.10322},
  year={2023}
}
```
