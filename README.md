
# Implementation of deep learning framework -- Attention UNet

The architecture was inspired by U-Net: Convolutional Networks for Biomedical Image Segmentation.

## Overview
This repository contains an unofficial implementation of Attention U-Net using PyTorch.<br/>
Please refer to the paper at the following page: 
[Attention-Guided Version of 2D UNet for Automatic Brain Tumor Segmentation](https://ieeexplore.ieee.org/document/8964956?denie] "Visit")

## Model
![Local Image](Attention_UNet.png "Attention UNet")
## Paper
If you use this software for your research, please cite:

```bibtex
@article{park2023gstnet,
  title={{GSTNet}: Flexible architecture under budget constraint for real-time human activity recognition},
  author={Park, Jaegyun and Lim, Won-Seon and Kim, Dae-Won and Lee, Jaesung},
  journal={Engineering Applications of Artificial Intelligence},
  volume={124},
  number={106543},
  year={2023},
  publisher={Elsevier}
}
```

## License
This program is available for download for non-commercial use, licensed under the GNU General Public License. This allows its use for research purposes or other free software projects but does not allow its incorporation into any type of commercial software.

## Files
The repository contains the following files:

- `model.py`: Python script file, containing the PyTorch implementation of the Attention UNet.
- `README.md`: Markdown file explaining the project and providing instructions.
