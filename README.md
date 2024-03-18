# CGR
Generalist segmentation models are increasingly favored for diverse tasks involving various objects from different image sources.
Task-Incremental Learning (TIL) offers a privacy-preserving training paradigm using tasks arriving sequentially, instead of gathering them due to strict data sharing policies.
However, the task evolution can span a wide scope that involves shifts in both image appearance and segmentation semantics with intricate correspondence, causing concurrent appearance and semantic forgetting.
To solve this issue, we propose a Comprehensive Generative Replay (CGR) framework to restore appearance and semantic knowledge by synthesizing image-mask pairs to mimic past task data, which focuses on two aspects: modeling image-mask correspondence and promoting scalability for diverse tasks.
Specifically, we propose a novel Bayesian Joint Diffusion (BJD) model for high-quality synthesis of image-mask pairs with their correspondence explicitly preserved by conditional denoising.
Furthermore, we develop a Task-Oriented Adapter (TOA) that recalibrates prompt embeddings to modulate data synthesis with higher scalability for diverse tasks.
Experiments on incremental tasks (cardiac, fundus and prostate segmentation) demonstrate the clear advantage of our method for relieving concurrent appearance and semantic forgetting.

![overview](overview_small_3.pdf)

## Usage

### 1. Data Pre-processing
Data of three tasks: Cardiac, Fundus, Prostate can be downloaded from [Cardiac](https://www.ub.edu/mnms/), [Fundus](https://ieeexplore.ieee.org/document/9163289) and [Prostate](https://ieeexplore.ieee.org/document/9000851). The pre-processing pipeline of three tasks follows their works described in their papers.

### 2. Model Training
Firstly, run `train_bjdwithtoa.sh` to learn three tasks sequentially to simulate image-mask pair of each task. The training hyper-parameters can be set in the code. Notably, the pre-trained weights of CLIP text encoder can be downloaded from [CLIP text encoder](https://huggingface.co/stabilityai/stable-diffusion-2). Then, run `train_segnetwork.py` to update the segmentation network sequentially. Similarly, the training hyper-parameters can be set in the `options\base_options.py`. 
