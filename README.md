- [ECCV 2020](#eccv-2020)
  * [VLN](#vln)
    + [1. Counterfactual Vision-and-Language Navigation via Adversarial Path Sampler](#1-counterfactual-vision-and-language-navigation-via-adversarial-path-sampler)
    + [2. Improving Vision-and-Language Navigation with Image-Text Pairs from the Web](#2-improving-vision-and-language-navigation-with-image-text-pairs-from-the-web)
    + [3. Soft Expert Reward Learning for Vision-and-Language Navigation](#3-soft-expert-reward-learning-for-vision-and-language-navigation)
    + [4. Object-and-Action Aware Model for Visual Language Navigation](#4-object-and-action-aware-model-for-visual-language-navigation)
    + [5. Active Visual Information Gathering for Vision-Language Navigation](#5-active-visual-information-gathering-for-vision-language-navigation)
    + [6. Environment-agnostic Multitask Learning for Natural Language Grounded Navigation](#6-environment-agnostic-multitask-learning-for-natural-language-grounded-navigation)
    + [7. Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments](#7-beyond-the-nav-graph--vision-and-language-navigation-in-continuous-environments)
    + [8. Occupancy Anticipation for Efficient Exploration and Navigation](#8-occupancy-anticipation-for-efficient-exploration-and-navigation)
    + [9. Learning Object Relation Graph and Tentative Policy for Visual Navigation](#9-learning-object-relation-graph-and-tentative-policy-for-visual-navigation)
    + [10. Seeing the Un-Scene: Learning Amodal Semantic Maps for Room Navigation](#10-seeing-the-un-scene--learning-amodal-semantic-maps-for-room-navigation)
    + [* SoundSpaces: Audio-Visual Navigation in 3D Environments](#--soundspaces--audio-visual-navigation-in-3d-environments)
  * [VQA](#vqa)
    + [1. AiR: Attention with Reasoning Capability](#1-air--attention-with-reasoning-capability)
    + [2. A Competence-aware Curriculum for Visual Concepts Learning via Question Answering](#2-a-competence-aware-curriculum-for-visual-concepts-learning-via-question-answering)
    + [3. Reducing Language Biases in Visual Question Answering with Visually-Grounded Question Encoder](#3-reducing-language-biases-in-visual-question-answering-with-visually-grounded-question-encoder)
    + [4. Multi-Agent Embodied Question Answering in Interactive Environments](#4-multi-agent-embodied-question-answering-in-interactive-environments)
    + [5. Knowledge-Based Video Question Answering with Unsupervised Scene Descriptions](#5-knowledge-based-video-question-answering-with-unsupervised-scene-descriptions)
    + [6. Semantic Equivalent Adversarial Data Augmentation for Visual Question Answering](#6-semantic-equivalent-adversarial-data-augmentation-for-visual-question-answering)
    + [7. Visual Question Answering on Image Sets](#7-visual-question-answering-on-image-sets)
    + [8. VQA-LOL: Visual Question Answering under the Lens of Logic](#8-vqa-lol--visual-question-answering-under-the-lens-of-logic)
    + [9. TRRNet: Tiered Relation Reasoning for Compositional Visual Question Answering](#9-trrnet--tiered-relation-reasoning-for-compositional-visual-question-answering)
    + [10. Interpretable Visual Reasoning via Probabilistic Formulation under Natural Supervision](#10-interpretable-visual-reasoning-via-probabilistic-formulation-under-natural-supervision)
    + [* VisualCOMET: Reasoning about the Dynamic Context of a Still Image](#--visualcomet--reasoning-about-the-dynamic-context-of-a-still-image)
  * [Visual Grounding](#visual-grounding)
    + [1. Contrastive Learning for Weakly Supervised Phrase Grounding](#1-contrastive-learning-for-weakly-supervised-phrase-grounding)
    + [2. Propagating Over Phrase Relations for One-Stage Visual Grounding](#2-propagating-over-phrase-relations-for-one-stage-visual-grounding)
    + [3. Improving One-stage Visual Grounding by Recursive Sub-query Construction](#3-improving-one-stage-visual-grounding-by-recursive-sub-query-construction)


## ECCV 2020

### VLN

#### 1. Counterfactual Vision-and-Language Navigation via Adversarial Path Sampler

Tsu-Jui Fu, Xin Eric Wang, Matthew F. Peterson,Scott T. Grafton, Miguel P. Eckstein, William Yang Wang

<details>
<summary>
Abstract
</summary>
<br/>
Vision-and-Language Navigation (VLN) is a task where agents must decide how to move through a 3D environment to reach a goal by grounding natural language instructions to the visual surroundings. One of the problems of the VLN task is data scarcity since it is difficult to collect enough navigation paths with human-annotated instructions for interactive environments. In this paper, we explore the use of counterfactual thinking as a human-inspired data augmentation method that results in robust models. Counterfactual thinking is a concept that describes the human propensity to create possible alternatives to life events that have already occurred. We propose an adversarial-driven counterfactual reasoning model that can consider effective conditions instead of low-quality augmented data. In particular, we present a model-agnostic adversarial path sampler (APS) that learns to sample challenging paths that force the navigator to improve based on the navigation performance. APS also serves to do pre-exploration of unseen environments to strengthen the model's ability to generalize. We evaluate the influence of APS on the performance of different VLN baseline models using the room-to-room dataset (R2R). The results show that the adversarial training process with our proposed APS benefits VLN models under both seen and unseen environments. And the pre-exploration process can further gain additional improvements under unseen environments.
</details>

[[pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510069.pdf)] 

[[supplementary material](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510069-supp.pdf)]

#### 2. Improving Vision-and-Language Navigation with Image-Text Pairs from the Web

Arjun Majumdar, Ayush Shrivastava, Stefan Lee, Peter Anderson, Devi Parikh, Dhruv Batra

<details>
<summary>
Abstract
</summary>
<br/>
Following a navigation instruction such as 'Walk down the stairs and stop at the brown sofa' requires embodied AI agents to ground referenced scene elements referenced (e.g. 'stairs') to visual content in the environment (pixels corresponding to 'stairs'). We ask the following question -- can we leverage abundant `disembodied' web-scraped vision-and-language corpora (e.g. Conceptual Captions) to learn the visual groundings that improve performance on a relatively data-starved embodied perception task (Vision-and-Language Navigation)? Specifically, we develop VLN-BERT, a visiolinguistic transformer-based model for scoring the compatibility between an instruction ('...stop at the brown sofa') and a trajectory of panoramic RGB images captured by the agent. We demonstrate that pretraining VLN-BERT on image-text pairs from the web before fine-tuning on embodied path-instruction data significantly improves performance on VLN -- outperforming prior state-of-the-art in the fully-observed setting by 4 absolute percentage points on success rate. Ablations of our pretraining curriculum show each stage to be impactful -- with their combination resulting in further gains.
</details>

[[pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510256.pdf)] 

[[supplementary material](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510256-supp.pdf)]

#### 3. Soft Expert Reward Learning for Vision-and-Language Navigation

Hu Wang, Qi Wu, Chunhua Shen

<details>
<summary>
Abstract
</summary>
<br/>
    Vision-and-Language Navigation (VLN) requires an agent to find a specified spot in an unseen environment by following natural language instructions. Dominant methods based on supervised learning clone expert's behaviours and thus perform better on seen environments, while showing restricted performance on unseen ones. Reinforcement Learning (RL) based models show better generalisation ability but have issues as well, requiring large amount of manual reward engineering is one of which. In this paper, we introduce a Soft Expert Reward Learning (SERL) model to overcome the reward engineering designing and generalisation problems of the VLN task. Our proposed method consists of two complementary components: Soft Expert Distillation (SED) module encourages agents to behave like an expert as much as possible, but in a soft fashion; Self Perceiving (SP) module targets at pushing the agent towards the final destination as fast as possible. Empirically, we evaluate our model on the VLN seen, unseen and test splits and the model outperforms the state-of-the-art methods on most of the evaluation metrics.
</details>

[[pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540120.pdf)]

#### 4. Object-and-Action Aware Model for Visual Language Navigation

Yuankai Qi, Zizheng Pan, Shengping Zhang, Anton van den Hengel, Qi Wu

<details>
<summary>
Abstract
</summary>
<br/>
    Vision-and-Language Navigation (VLN) is unique in that it requires turning relatively general natural-language instructions into robot agent actions, on the basis of visible environments. This requires to extract value from two very different types of natural-language information. The first is object description (e.g., `table', `door'), each presenting as a tip for the agent to determine the next action by finding the item visible in the environment, and the second is action specification (e.g., `go straight', `turn left') which allows the robot to directly predict the next movements without relying on visual perceptions. However, most existing methods pay few attention to distinguish these information from each other during instruction encoding and mix together the matching between textual object/action encoding and visual perception/orientation features of candidate viewpoints. In this paper, we propose an Object-and-Action Aware Model (OAAM) that processes these two different forms of natural language based instruction separately. This enables each process to match object-centered/action-centered instruction to their own counterpart visual perception/action orientation flexibly. However, one side-issue caused by above solution is that an object mentioned in instructions may be observed in the direction of two or more candidate viewpoints, thus the OAAM may not predict the viewpoint on the shortest path as the next action. To handle this problem, we design a simple but effective path loss to penalize trajectories deviating from the ground truth path. Experimental results demonstrate the effectiveness of the proposed model and path loss, and the superiority of their combination with a 50% SPL score on the R2R dataset and a 40% CLS score on the R4R dataset in unseen environments, outperforming the previous state-of-the-art.
</details>

[[pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550307.pdf)]

#### 5. Active Visual Information Gathering for Vision-Language Navigation

Hanqing Wang, Wenguan Wang, Tianmin Shu, Wei Liang, Jianbing Shen

<details>
<summary>
Abstract
</summary>
<br/>
    Vision-language navigation (VLN) is the task of entailing an agent to carry out navigational instructions inside photo-realistic environments. One of the key challenges in VLN is how to conduct a robust navigation by mitigating the uncertainty caused by ambiguous instructions and insufficient observation of the environment. Agents trained by current approaches typically suffer from this and would consequently struggle to avoid random and inefficient actions at every step. In contrast, when humans face such a challenge, they can still maintain robust navigation by actively exploring the surroundings to gather more information and thus make more confident navigation decisions. This work draws inspiration from human navigation behavior and endows an agent with an active information gathering ability for a more intelligent vision-language navigation policy. To achieve this, we propose an end-to-end framework for learning an exploration policy that decides i) when and where to explore, ii) what information is worth gathering during exploration, and extbf{iii)} how to adjust the navigation decision after the exploration. The experimental results show promising exploration strategies emerged from training, which leads to significant boost in navigation performance. On the R2R challenge leaderboard, our agent gets promising results all three VLN settings, i.e., single run, pre-exploration, and beam search.
</details>

[[pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670307.pdf)] 

[[supplementary material](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670307-supp.zip)] 

#### 6. Environment-agnostic Multitask Learning for Natural Language Grounded Navigation

Xin Eric Wang, Vihan Jain, Eugene Ie, William Yang Wang, Zornitsa Kozareva, Sujith Ravi[2]

<details>
<summary>
Abstract
</summary>
<br/>
    Recent research efforts enable study for natural language grounded navigation in photo-realistic environments, e.g., following natural language instructions or dialog. However, existing methods tend to overfit training data in seen environments and fail to generalize well in previously unseen environments. To close the gap between seen and unseen environments, we aim at learning a generalized navigation model from two novel perspectives:(1) we introduce a multitask navigation model that can be seamlessly trained on both Vision-Language Navigation (VLN) and Navigation from Dialog History (NDH) tasks, which benefits from richer natural language guidance and effectively transfers knowledge across tasks;(2) we propose to learn environment-agnostic representations for the navigation policy that are invariant among the environments seen during training, thus generalizing better on unseen environments. Extensive experiments show that environment-agnostic multitask learning significantly reduces the performance gap between seen and unseen environments, and the navigation agent trained so outperforms baselines on unseen environments by 16\% (relative measure on success rate) on VLN and 120\% (goal progress) on NDH. Our submission to the CVDN leaderboard establishes a new state-of-the-art for the NDH task on the holdout test set. Code is available at https://github.com/google-research/valan .
</details>

[[pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690409.pdf)] 

[[supplementary material](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690409-supp.pdf)] 

#### 7. Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments

Jacob Krantz, Erik Wijmans, Arjun Majumdar, Dhruv Batra, Stefan Lee

<details>
<summary>
Abstract
</summary>
<br/>
    We develop a language-guided navigation task set in a continuous 3D environment where agents must execute low-level actions to follow natural language navigation directions. By being situated in continuous environments, this setting lifts a number of assumptions implicit in prior work that represents environments as a sparse graph of panoramas with edges corresponding to navigability. Specifically, our setting drops the presumptions of known environment topologies, short-range oracle navigation, and perfect agent localization. To contextualize this new task, we develop models that mirror many of the advances made in prior settings as well as single-modality baselines. While some transfer, we find significantly lower absolute performance in the continuous setting – suggesting that performance in prior ‘navigation-graph’ settings may be inflated by the strong implicit assumptions.
</details>

[[pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730103.pdf)] 

[[supplementary material](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730103-supp.pdf)]

#### 8. Occupancy Anticipation for Efficient Exploration and Navigation

Santhosh K. Ramakrishnan, Ziad Al-Halah, Kristen Grauman

<details>
<summary>
Abstract
</summary>
<br/>
    State-of-the-art navigation methods leverage a spatial memory to generalize to new environments, but their occupancy maps are limited to capturing the geometric structures directly observed by the agent. We propose occupancy anticipation, where the agent uses its egocentric RGB-D observations to infer the occupancy state beyond the visible regions. In doing so, the agent builds its spatial awareness more rapidly, which facilitates efficient exploration and navigation in 3D environments. By exploiting context in both the egocentric views and top-down maps our model successfully anticipates a broader map of the environment, with performance significantly better than strong baselines. Furthermore, when deploying our model for the sequential decision-making tasks of exploration and navigation, we outperform state-of-the-art methods on the Gibson and Matterport3D datasets. Our approach is the winning entry in the 2020 Habitat PointNav Challenge. Project page: http://vision.cs.utexas.edu/projects/occupancy_anticipation
</details>

[[pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500392.pdf)] 

[[supplementary material](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500392-supp.zip)] 

#### 9. Learning Object Relation Graph and Tentative Policy for Visual Navigation

Heming Du, Xin Yu, Liang Zheng

<details>
<summary>
Abstract
</summary>
<br/>
    Target-driven visual navigation aims at navigating an agent towards a given target based on the observation of the agent. In this task, it is critical to learn informative visual representation and robust navigation policy. Aiming to improve these two components, this paper proposes three complementary techniques, object relation graph (ORG), trial-driven imitation learning (IL), and a memory-augmented tentative policy network (TPN). ORG improves visual representation learning by integrating object relationships, including category closeness and spatial correlations, mph{e.g.,} a TV usually co-occurs with a remote spatially. Both Trial-driven IL and TPN underlie robust navigation policy, instructing the agent to escape from deadlock states, such as looping or being stuck. Specifically, trial-driven IL is a type of supervision used in policy network training, while TPN, mimicking the IL supervision in unseen environment, is applied in testing. %instructing an agent to escape from deadlock states. Experiment in the artificial environment AI2-Thor validates that each of the techniques is effective. When combined, the techniques bring significantly improvement over baseline methods in navigation effectiveness and efficiency in unseen environments. We report 22.8\% and 23.5\% increase in success rate and Success weighted by Path Length (SPL), respectively. The code is available at \url{https://github.com/xiaobaishu0097/ECCV-VN.git}.
</details>

[[pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123520018.pdf)] 

[[supplementary material](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123520018-supp.zip)]

#### 10. Seeing the Un-Scene: Learning Amodal Semantic Maps for Room Navigation

Medhini Narasimhan, Erik Wijmans, Xinlei Chen, Trevor Darrell, Dhruv Batra, Devi Parikh, Amanpreet Singh

<details>
<summary>
Abstract
</summary>
<br/>
    We introduce a learning-based approach for room navigation using semantic maps. Our proposed architecture learns to predict top-down belief maps of regions that lie beyond the agent’s field of view while modeling architectural and stylistic regularities in houses. First, we train a model to generate amodal semantic top-down maps indicating beliefs of location, size, and shape of rooms by learning the underlying architectural patterns in houses. Next, we use these maps to predict a point that lies in the target room and train a policy to navigate to the point. We empirically demonstrate that by predicting semantic maps, the model learns common correlations found in houses and generalizes to novel environments. We also demonstrate that reducing the task of room navigation to point navigation improves the performance further. We will make our code publicly available and hope our work paves the way for further research in this space. 
</details>

[[pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630494.pdf)] 

[[supplementary material](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630494-supp.zip)]

#### * SoundSpaces: Audio-Visual Navigation in 3D Environments

Changan Chen, Unnat Jain, Carl Schissler, Sebastia Vicenc Amengual Gari, Ziad Al-Halah, Vamsi Krishna Ithapu, Philip Robinson, and Kristen Grauman

<details>
<summary>
Abstract
</summary>
<br/>
    Moving around in the world is naturally a multi-sensory experience, but today's embodied agents are deaf - restricted to solely their visual perception of the environment. We introduce audio-visual navigation for complex, acoustically and visually realistic 3D environments. By both seeing and hearing, the agent must learn to navigate to an audio-based target. We develop a multi-modal deep reinforcement learning pipeline to train navigation policies end-to-end from a stream of egocentric audio-visual observations, allowing the agent to (1) discover elements of the geometry of the physical space indicated by the reverberating audio and (2) detect and follow sound-emitting targets. We further introduce audio renderings based on geometrical acoustic simulations for a set of publicly available 3D assets and instrument AI-Habitat to support the new sensor, making it possible to insert arbitrary sound sources in an array of apartment, office, and hotel environments. Our results show that audio greatly benefits embodied visual navigation in 3D spaces.
</details>

[[pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510018.pdf)] 

[[supplementary material](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510018-supp.pdf)]



### VQA

#### 1. AiR: Attention with Reasoning Capability

Shi Chen, Ming Jiang, Jinhui Yang, Qi Zhao

<details>
<summary>
Abstract
</summary>
<br/>
    While attention has been an increasingly popular component in deep neural networks to both interpret and boost performance of models, little work has examined how attention progresses to accomplish a task and whether it is reasonable. In this work, we propose an Attention with Reasoning capability (AiR) framework that uses attention to understand and improve the process leading to task outcomes. We first define an evaluation metric based on a sequence of atomic reasoning operations, enabling quantitative measurement of attention that considers the reasoning process. We then collect human eye-tracking and answer correctness data, and analyze various machine and human attentions on their reasoning capability and how they impact task performance. Furthermore, we propose a supervision method to jointly and progressively optimize attention, reasoning, and task performance so that models learn to look at regions of interests by following a reasoning process. We demonstrate the effectiveness of the proposed framework in analyzing and modeling attention with better reasoning capability and task performance. The code and data are available at https://github.com/szzexpoi/AiR
</details>
[[pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460086.pdf)] 

[[supplementary material](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460086-supp.zip)]

#### 2. A Competence-aware Curriculum for Visual Concepts Learning via Question Answering

Qing Li, Siyuan Huang, Yining Hong, Song-Chun Zhu

<details>
<summary>
Abstract
</summary>
<br/>
    Humans can progressively learn visual concepts from easy to hard questions. To mimic this efficient learning ability, we propose a competence-aware curriculum for visual concept learning in a question-answering manner. Specifically, we design a neural-symbolic concept learner for learning the visual concepts and a multi-dimensional Item Response Theory (mIRT) model for guiding the learning process with an adaptive curriculum. The mIRT effectively estimates the concept difficulty and the model competence at each learning step from accumulated model responses. The estimated concept difficulty and model competence are further utilized to select the most profitable training samples. Experimental results on CLEVR show that with a competence-aware curriculum, the proposed method achieves state-of-the-art performances with superior data efficiency and convergence speed. Specifically, the proposed model only uses 40% of training data and converges three times faster compared with other state-of-the-art methods. 
</details>

[[pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470137.pdf)] 

[[supplementary material](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470137-supp.pdf)]

#### 3. Reducing Language Biases in Visual Question Answering with Visually-Grounded Question Encoder

Gouthaman KV, Anurag Mittal

<details>
<summary>
Abstract
</summary>
<br/>
    Recent studies have shown that current VQA models are heavily biased on the language priors in the train set to answer the question, irrespective of the image. E.g., overwhelmingly answer ""what sport is"" as ""tennis"" or ""what color banana"" as ""yellow."" This behavior restricts them from real-world application scenarios. In this work, we propose a novel model-agnostic question encoder, Visually-Grounded Question Encoder (VGQE), for VQA that reduces this effect. VGQE utilizes both visual and language modalities equally while encoding the question. Hence the question representation itself gets sufficient visual-grounding, and thus reduces the dependency of the model on the language priors. We demonstrate the effect of VGQE on three recent VQA models and achieve state-of-the-art results on the bias-sensitive split of the VQAv2 dataset; VQA-CPv2. Further, unlike the existing bias-reduction techniques, on the standard VQAv2 benchmark, our approach does not drop the accuracy; instead, it improves the performance.
</details>

[[pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580018.pdf)] 

[[supplementary material](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580018-supp.pdf)]

#### 4. Multi-Agent Embodied Question Answering in Interactive Environments

Sinan Tan, Weilai Xiang, Huaping Liu, Di Guo, Fuchun Sun

<details>
<summary>
Abstract
</summary>
<br/>
    We investigate a new AI task --- Multi-Agent Interactive Question Answering --- where several agents explore the scene jointly in interactive environments to answer a question. To cooperate efficiently and answer accurately, agents must be well-organized to have balanced work division and share knowledge about the objects involved. We address this new problem in two stages: Multi-Agent 3D Reconstruction in Interactive Environments and Question Answering. Our proposed framework features multi-layer structural and semantic memories shared by all agents, as well as a question answering model built upon a 3D-CNN network to encode the scene memories. During the reconstruction, agents simultaneously explore and scan the scene with a clear division of work, organized by next viewpoints planning. We evaluate our framework on the IQuADv1 dataset and outperform the IQA baseline in a single-agent scenario. In multi-agent scenarios, our framework shows favorable speedups while remaining high accuracy.
</details>

[[pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580647.pdf)]

#### 5. Knowledge-Based Video Question Answering with Unsupervised Scene Descriptions

Noa Garcia, Yuta Nakashima

<details>
<summary>
Abstract
</summary>
<br/>
    To understand movies, humans constantly reason over the dialogues and actions shown in specific scenes and relate them to the overall storyline already seen. Inspired by this behaviour, we design ROLL, a model for knowledge-based video story question answering that leverages three crucial aspects of movie understanding: dialog comprehension, scene reasoning, and storyline recalling. In ROLL, each of these tasks is in charge of extracting rich and diverse information by 1) processing scene dialogues, 2) generating unsupervised video scene descriptions, and 3) obtaining external knowledge in a weakly supervised fashion. To answer a given question correctly, the information generated by each inspired-cognitive task is encoded via Transformers and fused through a modality weighting mechanism, which balances the information from the different sources. Exhaustive evaluation demonstrates the effectiveness of our approach, which yields a new state-of-the-art on two challenging video question answering datasets: KnowIT VQA and TVQA+.
</details>

[[pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630562.pdf)] 

[[supplementary material](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630562-supp.pdf)] 

#### 6. Semantic Equivalent Adversarial Data Augmentation for Visual Question Answering

Ruixue Tang, Chao Ma, Wei Emma Zhang, Qi Wu, Xiaokang Yang

<details>
<summary>
Abstract
</summary>
<br/>
    Visual Question Answering (VQA) has achieved great success thanks to the fast development of deep neural networks (DNN). On the other hand, the data augmentation, as one of the major tricks for DNN, has been widely used in many computer vision tasks. However, there are few works studying the data augmentation problem for VQA and none of the existing image based augmentation schemes (such as rotation and flipping) can be directly applied to VQA due to its semantic structure -- an $\langle image, question, answer angle$ triplet needs to be maintained correctly. For example, a direction related Question-Answer (QA) pair may not be true if the associated image is rotated or flipped. In this paper, instead of directly manipulating images and questions, we use generated adversarial examples for both images and questions as the augmented data. The augmented examples do not change the visual properties presented in the image as well as the extbf{semantic} meaning of the question, the correctness of the $\langle image, question, answer angle$ is thus still maintained. We then use adversarial learning to train a classic VQA model (BUTD) with our augmented data. We find that we not only improve the overall performance on VQAv2, but also can withstand adversarial attack effectively, compared to the baseline model. The source code is available at https://github.com/zaynmi/seada-vqa.
</details>

[[pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123640426.pdf)] 

[[supplementary material](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123640426-supp.pdf)]

#### 7. Visual Question Answering on Image Sets

Ankan Bansal, Yuting Zhang, Rama Chellappa

<details>
<summary>
Abstract
</summary>
<br/>
    We introduce the task of Image-Set Visual Question Answering (ISVQA), which generalizes the commonly studied single-image VQA problem to multi-image settings. Taking a natural language question and a set of images as input, it aims to answer the question based on the content of the images. The questions can be about objects and relationships in one or more images or about the entire scene depicted by the image set. To enable research in this new topic, we introduce two ISVQA datasets - indoor and outdoor scenes. They simulate the real-world scenarios of indoor image collections and multiple car-mounted cameras, respectively. The indoor-scene dataset contains 91,479 human-annotated questions for 48,138 image sets, and the outdoor-scene dataset has 49,617 questions for 12,746 image sets. We analyze the properties of the two datasets, including question-and-answer distributions, types of questions, biases in dataset, and question-image dependencies. We also build new baseline models to investigate new research challenges in ISVQA.
</details>

[[pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660052.pdf)] 

[[supplementary material](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660052-supp.zip)]

#### 8. VQA-LOL: Visual Question Answering under the Lens of Logic

Tejas Gokhale, Pratyay Banerjee, Chitta Baral, Yezhou Yang

<details>
<summary>
Abstract
</summary>
<br/>
    Logical connectives and their implications on the meaning of a natural language sentence are a fundamental aspect of understanding. In this paper, we investigate whether visual question answering (VQA) systems trained to answer a question about an image, are able to answer the logical composition of multiple such questions. When put under this extit{Lens of Logic}, state-of-the-art VQA models have difficulty in correctly answering these logically composed questions. We construct an augmentation of the VQA dataset as a benchmark, with questions containing logical compositions and linguistic transformations (negation, disjunction, conjunction, and antonyms). We propose our {Lens of Logic (LOL)} model which uses question-attention and logic-attention to understand logical connectives in the question, and a novel Fréchet-Compatibility Loss, which ensures that the answers of the component questions and the composed question are consistent with the inferred logical operation. Our model shows substantial improvement in learning logical compositions while retaining performance on VQA. We suggest this work as a move towards robustness by embedding logical connectives in visual understanding.
</details>

[[pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660375.pdf)] 

[[supplementary material](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660375-supp.pdf)]

#### 9. TRRNet: Tiered Relation Reasoning for Compositional Visual Question Answering

Xiaofeng Yang, Guosheng Lin, Fengmao Lv, Fayao Liu

<details>
<summary>
Abstract
</summary>
<br/>
    Compositional visual question answering requires reasoning over both semantic and geometry object relations. We propose a novel tiered reasoning method that dynamically selects object level candidates based on language representations and generates robust pairwise relations within the selected candidate objects. The proposed tiered relation reasoning method can be compatible with the majority of the existing visual reasoning frameworks, leading to significant performance improvement with very little extra computational cost. Moreover, we propose a policy network that decides the appropriate reasoning steps based on question complexity and current reasoning status. In experiments, our model achieves state-of-the-art performance on two VQA datasets. 
</details>

[[pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660409.pdf)]

#### 10. Interpretable Visual Reasoning via Probabilistic Formulation under Natural Supervision

Xinzhe Han, Shuhui Wang, Chi Su, Weigang Zhang, Qingming Huang, Qi Tian

<details>
<summary>
Abstract
</summary>
<br/>
    Visual reasoning is crucial for visual question answering (VQA). However, without labelled programs, implicit reasoning under natural supervision is still quite challenging and previous models are hard to interpret. In this paper, we rethink implicit reasoning process in VQA, and propose a new formulation which maximizes the log-likelihood of joint distribution for the observed question and predicted answer. Accordingly, we derive a Temporal Reasoning Network (TRN) framework which models the implicit reasoning process as sequential planning in latent space. Our model is interpretable on both model design in probabilist and reasoning process via visualization. We experimentally demonstrate that TRN can support implicit reasoning across various datasets. The experiment results of our model are competitive to existing implicit reasoning models and surpass baseline by large margin on complicated reasoning tasks without extra computation cost in forward stage.
</details>

[[pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540528.pdf)] 

[[supplementary material](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540528-supp.pdf)]



#### * VisualCOMET: Reasoning about the Dynamic Context of a Still Image

Jae Sung Park, Chandra Bhagavatula, Roozbeh Mottaghi, Ali Farhadi, Yejin Choi

<details>
<summary>
Abstract
</summary>
<br/>
    Even from a single frame of a still image, people can reason about the dynamic story of the image before, after, and beyond the frame. For example, given an image of a man struggling to stay afloat in water, we can reason that the man fell into the water sometime in the past, the intent of that man at the moment is to stay alive, and he will need help in the near future or else he will get washed away. We propose Visual COMET, the novel framework of visual common-sense reasoning tasks to predict events that might have happened before, events that might happen next, and the intents of the people at present. To support research toward visual commonsense reasoning, we introduce the first large-scale repository of Visual Commonsense Graphs that consists of over 1.4 million textual descriptions of visual commonsense inferences carefully annotated over a diverse set of 59,000 images, each paired with short video summaries of before and after. In addition, we provide person-grounding (i.e., co-reference links) between people appearing in the image and people mentioned in the textual commonsense descriptions, allowing for tighter integration between images and text. We establish strong baseline performances on this task and demonstrate that integration between visual and textual commonsense reasoning is the key and wins over non-integrative alternatives.
</details>

[[pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500494.pdf)] 

[[supplementary material](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500494-supp.pdf)]



### Visual Grounding

#### 1. Contrastive Learning for Weakly Supervised Phrase Grounding

Tanmay Gupta, Arash Vahdat, Gal Chechik, Xiaodong Yang, Jan Kautz, Derek Hoiem

<details>
<summary>
Abstract
</summary>
<br/>
    Phrase grounding, the problem of associating image regions to caption words, is a crucial component of vision-language tasks. We show that phrase grounding can be learned by optimizing word-region attention to maximize a lower bound on mutual information between images and caption words. Given pairs of images and captions, we maximize compatibility of the attention-weighted regions and the words in the corresponding caption, compared to non-corresponding pairs of images and captions. A key idea is to construct effective negative captions for learning through language model guided word substitutions. Training with our negatives yields a $\sim10\%$ absolute gain in accuracy over randomly-sampled negatives from the training data. Our weakly supervised phrase grounding model trained on COCO-Captions shows a healthy gain of $5.7\%$ to achieve $76.7\%$ accuracy on Flickr30K Entities benchmark. Our code and project material will be available at http://tanmaygupta.info/info-ground."
</details>

[[pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480749.pdf)] 

[[supplementary material](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480749-supp.zip)]

#### 2. Propagating Over Phrase Relations for One-Stage Visual Grounding

Sibei Yang, Guanbin Li, Yizhou Yu

<details>
<summary>
Abstract
</summary>
<br/>
    Phrase level visual grounding aims to locate in an image the corresponding visual regions referred to by multiple noun phrases in a given sentence. Its challenge comes not only from large variations in visual contents and unrestricted phrase descriptions but also from unambiguous referrals derived from phrase relational reasoning. In this paper, we propose a linguistic structure guided propagation network for one-stage phrase grounding. It explicitly explores the linguistic structure of the sentence and performs relational propagation among noun phrases under the guidance of the linguistic relations between them. Specifically, we first construct a linguistic graph parsed from the sentence and then capture multimodal feature maps for all the phrasal nodes independently. The node features are then propagated over the edges with a tailor-designed relational propagation module and ultimately integrated for final prediction. Experiments on Flicker30K Entities dataset show that our model outperforms state-of-the-art methods and demonstrate the effectiveness of propagating among phrases with linguistic relations.
</details>

[[pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123640579.pdf)] 

#### 3. Improving One-stage Visual Grounding by Recursive Sub-query Construction

Zhengyuan Yang, Tianlang Chen, Liwei Wang, Jiebo Luo

<details>
<summary>
Abstract
</summary>
<br/>
    We improve one-stage visual grounding by addressing current limitations on grounding long and complex queries. Existing one-stage methods encode the entire language query as a single sentence embedding vector, e.g., taking the embedding from BERT or the hidden state from LSTM. This single vector representation is prone to overlooking the detailed descriptions in the query. To address this query modeling deficiency, we propose a recursive sub-query construction framework, which reasons between image and query for multiple rounds and reduces the referring ambiguity step by step. We show our new one-stage method obtains 5.0%, 4.5%, 7.5%, 12.8% absolute improvements over the state-of-the-art one-stage approach on ReferItGame, RefCOCO, RefCOCO+, and RefCOCOg, respectively. In particular, superior performances on longer and more complex queries validates the effectiveness of our query modeling. Code is available at https://github.com/zyang-ur/ReSC .
</details>

[[pdf](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590375.pdf)] 

[[supplementary material](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590375-supp.pdf)]

