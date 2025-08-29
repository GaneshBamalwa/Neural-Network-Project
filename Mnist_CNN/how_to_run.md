## Prerequisites

- Python 3.8 or higher
- PyTorch (>= 1.12.0)
- torchvision
- matplotlib
- numpy

You can install all dependencies using:

```bash
pip install torch torchvision matplotlib numpy
```

## How to Run

- Clone the repository:
```bash
git clone https://github.com/your-username/cnn-mnist.git
cd cnn-mnist
```
- Run the training script:
```bash
python train.py
```
-After training, the model weights will be saved as: **cnn_mnist.pth**.

-To evaluate on the test set:
```bash
python evaluate.py
```
- To test on a single image:
```bash
python predict.py --image path_to_image.png
```
