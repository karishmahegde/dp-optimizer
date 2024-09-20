# Differentially Private Optimizer <br> \[CSCI8960: Privacy Preserving Data Analysis\]

## Description
The goal of this project is to develop a highly accurate, efficient, and differentially private optimizer. The objectives include surpassing the current state-of-the-art accuracy of 70.7% (with ε = 3.0, δ = 10⁻⁵), ensuring the optimizer fulfills (ε, δ)-differential privacy (DP), and achieving scalability comparable to DP-SGD. Additionally, a key focus is on the reproducibility of results, with the final report highlighting the project’s contributions.

 **Current Stage: Project Proposal** 


## 🔶 Methodology

### 🔹 (Non-Private) Optimizers to Try
As part of this project, we aim to work with three optimizers, namely Stochastic Gradient Descent (SGD), Adam, and RMSprop. Initial studies on these optimizers have shown advantages in terms of adaptability, simplicity, and stability in training:

- **Adam**: Combines momentum and adaptive learning rates, making it robust across many domains.
- **RMSprop**: Appropriately scales the learning rate to prevent vanishing/exploding gradients.
- **SGD**: Simple and widely used in neural network training, but may require tuning of learning rates.

### 🔹 Main Components of the Optimizer
The DP optimizer will be developed based on the following components:

- **Gradient Clipping**: To ensure privacy, we will clip gradients to limit the influence of any single data point.
- **Error-Feedback Mechanism**: This mechanism improves convergence by tackling the noise introduced by gradient clipping.
- **Noise Addition**: The optimizer should satisfy (ε, δ)-differential privacy, so noise will be added to gradients.

### 🔹 Rationale Behind Algorithm Choices
The optimizers we are testing offer performance and adaptability while better handling noise and efficient learning. With modifications appropriate for privacy, the DP optimizer is expected to match or outperform DP-SGD.

### 🔹 Non-Private Algorithm Steps

#### Notations:
- **θ**: Vector representing the weights (parameters) of the model
- **g<sub>i</sub>**: Gradient of the loss function with respect to θ<sub>i</sub> (i.e., ∂L/∂θ<sub>i</sub>)
- **η**: Learning rate
- **λ**: Weight decay coefficient
- **μ**: Momentum coefficient
- **m<sub>i</sub>**: Momentum buffer for each parameter θ<sub>i</sub>
- **d**: Damping factor for momentum

#### Steps:
1. **L2 Regularization**:  
   Regularization helps prevent overfitting by handling large weight values. The gradient is adjusted as follows:  
   > g<sub>i</sub> ← g<sub>i</sub> + λθ<sub>i</sub>

   <br> This term pushes the weights θ<sub>i</sub> toward zero, promoting smaller parameter values.

2. **Momentum**:  
   Momentum helps accelerate the gradient vectors in the correct direction and dampens oscillations:  
   > m<sub>i</sub> ← μm<sub>i</sub> + (1− d)g<sub>i</sub> 
   

   The gradients are replaced by the momentum-adjusted value:  
   > g<sub>i</sub> ← m<sub>i</sub>
   

   This effectively smooths the updates by incorporating previous gradient information.

4. **Gradient Descent Parameter Update**:  
   Once the gradient g<sub>i</sub> has been adjusted for weight decay and momentum, it is used to update the parameters θ<sub>i</sub>:  
   > θ<sub>i</sub> ← θ<sub>i</sub> − ηg<sub>i</sub>

## 🔶 Experimental Setup

### 🔹 System Description
The model will be developed using PyTorch and the differential privacy library [Opacus](https://ai.meta.com/blog/introducing-opacus-a-high-speed-library-for-training-pytorch-models-with-differential-privacy/). It will be tested on PyTorch’s CIFAR-10 and CIFAR-100 datasets. The model will be trained on the university’s server equipped with GPUs (csci-cuda.cs.uga.edu) to handle the computational load.

### 🔹 Parameters to Measure
The performance of the model will be evaluated based on the following parameters:
- **Training loss and accuracy**
- **Differential privacy bounds (ε, δ)**
- **Model generalization (test accuracy)**

### 🔹 Design of Experiments
To demonstrate the improvement in the developed model, we will compare the convergence rate, ultimate accuracy, and privacy budget usage of DP-SGD and DiceSGD.

## 🔶 References
1. Jian Du, Song Li, Xiangyi Chen, Siheng Chen, and Mingyi Hong. [*Dynamic Differential-Privacy Preserving SGD.* arXiv preprint arXiv:2111.00173 (2022).](https://arxiv.org/abs/2111.00173)

2. Meta. [*Introducing Opacus: A high-speed library for training PyTorch models with differential privacy.*](https://ai.meta.com/blog/introducing-opacus-a-high-speed-library-for-training-pytorch-models-with-differential-privacy/) August 2020.

