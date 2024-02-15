# Tutorial

# Installation

Install PyTorch, Numpy, and Matplot without CUDA support.
This works on _all_ computers.

Admin

    pip3 install numpy matplotlib torch torchvision torchaudio

Local user

    python -m pip install numpy matplotlib torch torchvision torchaudio


Install PyTorch, NumPy, and Matplot for CUDA.
This works only if you have installed a graphic card with Nvidia Chip.
Then you can use GPU for tensor calculation which is way faster than CPU.

Admin

    pip3 install numpy matplotlib torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Local user

    python -m pip install numpy matplotlib torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


Create `requirements.txt`

Admin

    pip freeze > requirements.txt

Local user

    python -m pip freeze > requirements.txt

