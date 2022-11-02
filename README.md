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
> Why?
* EMA and attention mechanisms each have their own limitations
* Exponential Moving Average (EMA) captures local dependencies that exponentially decay over time
* Combination allows us to leverage their strengths to complement each other
* The combined model enjoys the benefit from strong inductive bias and maintains the capacity to learn complex dependency patterns.


## Multi-dimensional Damped EMA
* Learnable coefficients
* Develop MEGA mechanism by integrating the EMA with a variant of the single-head gated attention
![GitHub Logo](/Images/MEGA.png)


##



## MEGA Blocks



## MEGA with Linear Complexity

![GitHub Logo](/Images/MEGA_Chunk.png)


## Experiments



#### Raw Speech Classification



#### Image Classification



## Code Demonstration

https://github.com/thedon58/MEGA/blob/main/Code%20Example.ipynb


## Questions

1.
2.
3.

## Critical Analysis



## References

