# Moving Average Equipped Gated Attention

## Background

> Definition
* A simple, theoretically grounded, single-head gated attention mechanism equiped with (exponential) moving average to incorporate inductive bias of position-aware local dependencies into the position-agnostic attention mechanism.

> Reason
* MEGA was proposed to combat the attention mechanism's two common drawbacks:
1. Weak inductive bias
2. Quadratic computational complexity

![GitHub Logo](/Images/XFM.png)


## EMA Combination
> Why combine attention head with EMA?
* EMA and attention mechanisms each have their own limitations
* Exponential Moving Average (EMA) captures local dependencies that exponentially decay over time
* Combination allows us to leverage their strengths to complement each other
* The combined model enjoys the benefit from strong inductive bias and maintains the capacity to learn complex dependency patterns
* ![GitHub Logo](/Images/EMA.png)


## Multi-dimensional Damped EMA
* Modification of the standard exponential moving average to improve flexibility and capacity
* The relaxing of the coupled weights of the previous and current observations
![GitHub Logo](/Images/Damped.png)
* Learnable coefficients
* Develop MEGA mechanism by integrating the EMA with a variant of the single-head gated attention
* Extend the shape of α and δ from one-dimensional vector to two-dimensional matrix
![GitHub Logo](/Images/MEGA.png)
* Query and Key sequences are computed by applying scalars and offsets to **Z** while value sequence is from the original **X**
* ![GitHub Logo](/Images/QKV.png)

## MEGA with Linear Complexity
![GitHub Logo](/Images/MEGA_Chunk.png)
* Introduce MEGA-chunk, a variant of MEGA with linear complexity, which applies attention to each local chunk of fixed length
* First, split the sequences of queries, keys, and values into chunks of length c
* Attention operation is applied to each chunk, yielding linear complexity
* However, this method loses contextual information from other chunks, but the EMA sub-layer in MEGA captures local contextual information near each token whose outputs are used as the inputs to the attention sub-layer



## Experiments
> Types of Experiments Ran:
* Long-Context Sequence Modeling
* Raw Speech Classification
* Auto-regressive Language Modeling
* Neural Machine Translation
* Image Classification


#### Long-Context Sequence Modeling
* MEGA evaluation on the Long Range Arena (LRA) benchmark (2021)
* Designed for the purpose of evaluating sequence models under the long-context scenario
* Input sequences range from 1,000 - 16,000 tokens

![GitHub Logo](/Images/LRA.png)


#### Image Classification
* 1,280,000 training images & 50,000 validation images from 1,000 classes
* Top-1 accuracy on the validation set
* Experiment mostly followed DeiT's approach of applying several data augmentation and regularization methods that facilitate the training process and MEGA still outperformed DeiT
![GitHub Logo](/Images/ImageClass.png)

## Code Demonstration

https://github.com/thedon58/MEGA/blob/main/Code%20Example.ipynb


## Questions

1. ??? What is an advantage of having strong inductive bias? Why do we want this?
2.
3.

## Critical Analysis
Having its one month anniversary a few days ago, this paper is still very fresh and new to the world of AI/machine learning, but I can see it sticking around for some time. This "hybridization" of the attention layer has already posted some great results in the experiments run in this study where it has a better accuracy than most models and runs much faster and uses less memory than a baseline transformer model.


## References
* https://arxiv.org/abs/2209.10655
* https://github.com/lucidrains/Mega-pytorch
* https://twitter.com/violet_zct
* https://strathprints.strath.ac.uk/25635/
* https://github.com/facebookresearch/mega
