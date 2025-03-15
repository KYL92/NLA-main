## VideoNLA : Navigating Label Noisy for Facial Expression Recognition in Video (Navigating Label Ambiguity for ABAW 8th)
### Abstact
Facial Expression Recognition (FER) is a crucial role in human affective analysis and has been applied in various fields, such as human-computer interaction. However, previous methods primarily depend on image datasets, which has limitation extract spatial-temporal features. To evaluate human emotions, the 8th Affective Behavior Analysis in-the-Wild (ABAW) \cite{abaw8} competition utilizes the video-based Aff-Wild2 dataset. This challenge consists of three tasks: Expression (EXPR) Recognition, Action Unit (AU) Detection, and Valence-Arousal (VA) Estimation. In this paper, we propose a method for expression recognition. Typically, in the EXPR, label ambiguity and class imbalance issues are consistently encountered. To address these challenges, we propose an improved version of NLA \cite{NLA}, called VideoNLA. VideoNLA is specifically designed to effectively handle both label ambiguity and class imbalance in video-based FER. Additionally, we incorporate a random erasing augmentation module to mitigate redundancy between video frames. Extensive experiments are conducted to validate the effectiveness of our proposed method.


### Training
Step 1: download basic facial expression dataset of ABAW 8th

Step 2: load the model weight in the ./weights (will be uploaded)

Step 3: change dataset_path, label_path and  in train.py to your path

(optional) : If you want to train with noise or imbalance, set noise or imbalanced to True.

Step 4: run python train.py 
