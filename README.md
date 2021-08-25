# MedSRGAN
## PyTorch implementation of "MedSRGAN: medical images super-resolution using generative adversarial networks"

<img src="./img/medsrgan.PNG" width="500px"></img>

```python
import torch
from generator import MLPMixer

generator = Generator(
      in_channels= 3,
      blocks= 8
)

discriminator = Discriminator(in_channels= 3)
```
