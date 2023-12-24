
# Implementation of deep learning framework -- Attention UNet

The architecture was inspired by U-Net: Convolutional Networks for Biomedical Image Segmentation.

## Overview
This program (Attention UNet) is designed to perform the real-time sensor-based activity recognition. The GTSNet includes a Grouped Temporal Shift (GTS) module that allows the network architecture to be flexibly modified by predefining the theoretical computational cost.
This software is a PyTorch implementation of the proposed method. The original version of this program was written by Jaegyun Park.

## Model
!["Attention UNet"]("Attention_UNet.png")
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
