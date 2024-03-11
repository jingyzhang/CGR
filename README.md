# CGR
Generalist segmentation models are increasingly favored for diverse tasks involving various objects from different image sources.
Task-Incremental Learning (TIL) offers a privacy-preserving training paradigm using tasks arriving sequentially, instead of gathering them due to strict data sharing policies.
However, the task evolution can span a wide scope that involves shifts in both image appearance and segmentation semantics with intricate correspondence, causing concurrent appearance and semantic forgetting.
To solve this issue, we propose a Comprehensive Generative Replay (CGR) framework to restore appearance and semantic knowledge by synthesizing image-mask pairs to mimic past task data, which focuses on two aspects: modeling image-mask correspondence and promoting scalability for diverse tasks.
Specifically, we propose a novel Bayesian Joint Diffusion (BJD) model for high-quality synthesis of image-mask pairs with their correspondence explicitly preserved by conditional denoising.
Furthermore, we develop a Task-Oriented Adapter (TOA) that recalibrates prompt embeddings to modulate data synthesis with higher scalability for diverse tasks.
Experiments on incremental tasks (cardiac, fundus and prostate segmentation) demonstrate the clear advantage of our method for relieving concurrent appearance and semantic forgetting.


# 1. Data Pre-processing
