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



## Multi-dimensional Damped EMA
![GitHub Logo](/Images/MEGA.png)


##



## MEGA Blocks



## MEGA with Linear Complexity
![GitHub Logo](/Images/MEGA_Chunk.png)


## Experiments



#### Raw Speech Classification



#### Image Classification



## Code Demonstration

```
import torch
from mega_pytorch import MegaLayer

layer = MegaLayer(
    dim = 128,                   # model dimensions
    ema_heads = 16,              # number of EMA heads
    attn_dim_qk = 64,            # dimension of queries / keys in attention
    attn_dim_value = 256,        # dimension of values in attention
    laplacian_attn_fn = False,   # whether to use softmax (false) or laplacian attention activation fn (true)
)

x = torch.randn(1, 1024, 128)     # (batch, seq, dim)

out = layer(x) # (1, 1024, 128)
```


## Critical Analysis



## References

