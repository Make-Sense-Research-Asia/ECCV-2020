[toc]

# ECCV 2020

## VLN

### 1. Counterfactual Vision-and-Language Navigation via Adversarial Path Sampler

<details>
<summary>
Abstract
</summary>
<br/>
Vision-and-Language Navigation (VLN) is a task where agents must decide how to move through a 3D environment to reach a goal by grounding natural language instructions to the visual surroundings. One of the problems of the VLN task is data scarcity since it is difficult to collect enough navigation paths with human-annotated instructions for interactive environments. In this paper, we explore the use of counterfactual thinking as a human-inspired data augmentation method that results in robust models. Counterfactual thinking is a concept that describes the human propensity to create possible alternatives to life events that have already occurred. We propose an adversarial-driven counterfactual reasoning model that can consider effective conditions instead of low-quality augmented data. In particular, we present a model-agnostic adversarial path sampler (APS) that learns to sample challenging paths that force the navigator to improve based on the navigation performance. APS also serves to do pre-exploration of unseen environments to strengthen the model's ability to generalize. We evaluate the influence of APS on the performance of different VLN baseline models using the room-to-room dataset (R2R). The results show that the adversarial training process with our proposed APS benefits VLN models under both seen and unseen environments. And the pre-exploration process can further gain additional improvements under unseen environments.
</details>

[[pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510069.pdf)] [[supplementary material](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510069-supp.pdf)]

### 2. Improving Vision-and-Language Navigation with Image-Text Pairs from the Web

Following a navigation instruction such as 'Walk down the stairs and stop at the brown sofa' requires embodied AI agents to ground referenced scene elements referenced (e.g. 'stairs') to visual content in the environment (pixels corresponding to 'stairs'). We ask the following question -- can we leverage abundant `disembodied' web-scraped vision-and-language corpora (e.g. Conceptual Captions) to learn the visual groundings that improve performance on a relatively data-starved embodied perception task (Vision-and-Language Navigation)? Specifically, we develop VLN-BERT, a visiolinguistic transformer-based model for scoring the compatibility between an instruction ('...stop at the brown sofa') and a trajectory of panoramic RGB images captured by the agent. We demonstrate that pretraining VLN-BERT on image-text pairs from the web before fine-tuning on embodied path-instruction data significantly improves performance on VLN -- outperforming prior state-of-the-art in the fully-observed setting by 4 absolute percentage points on success rate. Ablations of our pretraining curriculum show each stage to be impactful -- with their combination resulting in further gains.

[[pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510256.pdf)] [[supplementary material](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510256-supp.pdf)]

### 3. Soft Expert Reward Learning for Vision-and-Language Navigation

Vision-and-Language Navigation (VLN) requires an agent to find a specified spot in an unseen environment by following natural language instructions. Dominant methods based on supervised learning clone expert's behaviours and thus perform better on seen environments, while showing restricted performance on unseen ones. Reinforcement Learning (RL) based models show better generalisation ability but have issues as well, requiring large amount of manual reward engineering is one of which. In this paper, we introduce a Soft Expert Reward Learning (SERL) model to overcome the reward engineering designing and generalisation problems of the VLN task. Our proposed method consists of two complementary components: Soft Expert Distillation (SED) module encourages agents to behave like an expert as much as possible, but in a soft fashion; Self Perceiving (SP) module targets at pushing the agent towards the final destination as fast as possible. Empirically, we evaluate our model on the VLN seen, unseen and test splits and the model outperforms the state-of-the-art methods on most of the evaluation metrics.

[[pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540120.pdf)]

### 4. Object-and-Action Aware Model for Visual Language Navigation

Vision-and-Language Navigation (VLN) is unique in that it requires turning relatively general natural-language instructions into robot agent actions, on the basis of visible environments. This requires to extract value from two very different types of natural-language information. The first is object description (e.g., `table', `door'), each presenting as a tip for the agent to determine the next action by finding the item visible in the environment, and the second is action specification (e.g., `go straight', `turn left') which allows the robot to directly predict the next movements without relying on visual perceptions. However, most existing methods pay few attention to distinguish these information from each other during instruction encoding and mix together the matching between textual object/action encoding and visual perception/orientation features of candidate viewpoints. In this paper, we propose an Object-and-Action Aware Model (OAAM) that processes these two different forms of natural language based instruction separately. This enables each process to match object-centered/action-centered instruction to their own counterpart visual perception/action orientation flexibly. However, one side-issue caused by above solution is that an object mentioned in instructions may be observed in the direction of two or more candidate viewpoints, thus the OAAM may not predict the viewpoint on the shortest path as the next action. To handle this problem, we design a simple but effective path loss to penalize trajectories deviating from the ground truth path. Experimental results demonstrate the effectiveness of the proposed model and path loss, and the superiority of their combination with a 50% SPL score on the R2R dataset and a 40% CLS score on the R4R dataset in unseen environments, outperforming the previous state-of-the-art.

[[pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550307.pdf)]

### 5. Active Visual Information Gathering for Vision-Language Navigation

Vision-language navigation (VLN) is the task of entailing an agent to carry out navigational instructions inside photo-realistic environments. One of the key challenges in VLN is how to conduct a robust navigation by mitigating the uncertainty caused by ambiguous instructions and insufficient observation of the environment. Agents trained by current approaches typically suffer from this and would consequently struggle to avoid random and inefficient actions at every step. In contrast, when humans face such a challenge, they can still maintain robust navigation by actively exploring the surroundings to gather more information and thus make more confident navigation decisions. This work draws inspiration from human navigation behavior and endows an agent with an active information gathering ability for a more intelligent vision-language navigation policy. To achieve this, we propose an end-to-end framework for learning an exploration policy that decides i) when and where to explore, ii) what information is worth gathering during exploration, and extbf{iii)} how to adjust the navigation decision after the exploration. The experimental results show promising exploration strategies emerged from training, which leads to significant boost in navigation performance. On the R2R challenge leaderboard, our agent gets promising results all three VLN settings, i.e., single run, pre-exploration, and beam search.

[[pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670307.pdf)] [[supplementary material](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670307-supp.zip)] 

### 6. Environment-agnostic Multitask Learning for Natural Language Grounded Navigation

Recent research efforts enable study for natural language grounded navigation in photo-realistic environments, e.g., following natural language instructions or dialog. However, existing methods tend to overfit training data in seen environments and fail to generalize well in previously unseen environments. To close the gap between seen and unseen environments, we aim at learning a generalized navigation model from two novel perspectives:(1) we introduce a multitask navigation model that can be seamlessly trained on both Vision-Language Navigation (VLN) and Navigation from Dialog History (NDH) tasks, which benefits from richer natural language guidance and effectively transfers knowledge across tasks;(2) we propose to learn environment-agnostic representations for the navigation policy that are invariant among the environments seen during training, thus generalizing better on unseen environments. Extensive experiments show that environment-agnostic multitask learning significantly reduces the performance gap between seen and unseen environments, and the navigation agent trained so outperforms baselines on unseen environments by 16\% (relative measure on success rate) on VLN and 120\% (goal progress) on NDH. Our submission to the CVDN leaderboard establishes a new state-of-the-art for the NDH task on the holdout test set. Code is available at https://github.com/google-research/valan .

[[pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690409.pdf)] [[supplementary material](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690409-supp.pdf)] 

### 7. Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments

We develop a language-guided navigation task set in a continuous 3D environment where agents must execute low-level actions to follow natural language navigation directions. By being situated in continuous environments, this setting lifts a number of assumptions implicit in prior work that represents environments as a sparse graph of panoramas with edges corresponding to navigability. Specifically, our setting drops the presumptions of known environment topologies, short-range oracle navigation, and perfect agent localization. To contextualize this new task, we develop models that mirror many of the advances made in prior settings as well as single-modality baselines. While some transfer, we find significantly lower absolute performance in the continuous setting – suggesting that performance in prior ‘navigation-graph’ settings may be inflated by the strong implicit assumptions.

[[pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730103.pdf)] [[supplementary material](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730103-supp.pdf)]

## VQA

### 1. AiR: Attention with Reasoning Capability

While attention has been an increasingly popular component in deep neural networks to both interpret and boost performance of models, little work has examined how attention progresses to accomplish a task and whether it is reasonable. In this work, we propose an Attention with Reasoning capability (AiR) framework that uses attention to understand and improve the process leading to task outcomes. We first define an evaluation metric based on a sequence of atomic reasoning operations, enabling quantitative measurement of attention that considers the reasoning process. We then collect human eye-tracking and answer correctness data, and analyze various machine and human attentions on their reasoning capability and how they impact task performance. Furthermore, we propose a supervision method to jointly and progressively optimize attention, reasoning, and task performance so that models learn to look at regions of interests by following a reasoning process. We demonstrate the effectiveness of the proposed framework in analyzing and modeling attention with better reasoning capability and task performance. The code and data are available at https://github.com/szzexpoi/AiR

[[pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460086.pdf)] [[supplementary material](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460086-supp.zip)]

## Image Caption

### 1. Describing Textures using Natural Language

Textures in natural images can be characterized by color, shape, periodicity of elements within them, and other attributes that can be described using natural language. In this paper, we study the problem of describing visual attributes of texture on a novel dataset containing rich descriptions of textures, and conduct a systematic study of current generative and discriminative models for grounding language to images on this dataset. We find that while these models capture some properties of texture, they fail to capture several compositional properties, such as the colors of dots. We provide critical analysis of existing models by generating synthetic but realistic textures with different descriptions. Our dataset also allows us to train interpretable models and generate language-based explanations of what discriminative features are learned by deep networks for fine-grained categorization where texture plays a key role. We present visualizations of several fine-grained domains and show that texture attributes learned on our dataset offer improvements over expert-designed attributes on the Caltech-UCSD Birds dataset.

[[pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460052.pdf)] [[supplementary material](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460052-supp.pdf)]

### 2. 

## Visual Grounding

