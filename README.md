"UNet_Attention" 
# GTSNet: Flexible Architecture under Budget Constraint for Real-Time Human Activity Recognition from Wearable Sensor

## Abstract
This program (GTSNet) is designed to perform the real-time sensor-based activity recognition. The GTSNet includes a Grouped Temporal Shift (GTS) module that allows the network architecture to be flexibly modified by predefining the theoretical computational cost.
This software is a PyTorch implementation of the proposed method. The original version of this program was written by Jaegyun Park.

## Paper
If you use this software for your research, please cite:

 ```python
@article{park2023gtsnet,
title={GTSNet: Flexible architecture under budget constraint for real-time human activity recognition from wearable sensor},
author={Park, Jaegyun and Lim, Won-Seon and Kim, Tae-Hoon and Lee, Jaesun},
journal={Engineering Applications of Artificial Intelligence},
volume={124},
number={106543},
year={2023},
publisher={Elsevier}
}
'''

## License
This program is available for download for non-commercial use, licensed under the GNU General Public License. This allows its use for research purposes or other free software projects but does not allow its incorporation into any type of commercial software.

## Files
The repository contains the following files:

- `main.py`: Python script file, containing the implementation for training and test phases of the GTSNet.
- `model.py`: Python script file, containing the PyTorch implementation of the GTSNet.
- `utils.py`: Python script file, containing a collection of small Python functions.
- `README.md`: Markdown file explaining the project and providing instructions.
