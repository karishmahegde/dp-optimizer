# Differentially-Private Image Classifier  <br> \[CSCI8960: Privacy Preserving Data Analysis\]

## Abstract
This project aims to develop a highly efficient, scalable, and differentially private
optimizer. The main objective is to compete with the current state-of-the-art
accuracy of 70.7% while maintaining (ϵ = 3.0,δ = 10−5) differential privacy
guarantees. The project focuses on optimizing privacy-preserving techniques such
as Differentially Private Stochastic Gradient Descent (DP-SGD) using techniques like Adaptive
Gradient Clipping, Weight Standardization and dynamic learning rate adjustment to improve both accuracy and
computational performance. With a focus on reproducibility, the outcomes are
compared to privacy and accuracy requirements.

## 🔶 Best Accuracy
62.39%

## 🔶 Introduction
As organizations increasingly employ machine learning models, ensuring the privacy of individuals’
data used to train these models has become essential. Differential privacy (DP) offers a robust solution
to this challenge that ensures an adversary cannot predict data or the presence of an individual in
training data from released models. The central goal of this project is to develop a differentially
private optimizer that balances accuracy and privacy, addressing the limitations of current approaches.
The project focuses on developing models leveraging the DP-SGD (Differentially Private Stochastic
Gradient Descent) method through Opacus. As DP-SGD presents trade-offs between accuracy and
privacy, this work seeks to refine DP-SGD by introducing key optimizations, including Adaptive
Gradient Clipping, Weight Standardization and Adaptive Gradient Clipping. The project uses ResNet-20 and WRN-16-4 (Wide
Residual Networks) architectures for image classification on the CIFAR-10 dataset. Both models have
been modified to incorporate the proposed privacy-preserving techniques, with a focus on achieving
high accuracy while maintaining privacy budgets.

## 🔶 Models
### 🔷 ResNet-20: The Initial Model
The concept of Residual Learning was first presented in the 2015 landmark paper by K. He et al.[1]
to address the vanishing gradient problem that occurs when very deep networks perform worse than
their shallower counterparts. 

The main concept is to reframe deep network layers as learning residual functions with respect
to the layer inputs instead than learning unreferenced functions directly. This is accomplished
through "shortcuts" between each layer, in which the input of a block is kept and added to the
transformation’s output inside the block. As the network develops deeper, this aids in the retention of
crucial information.
ResNet-20 is a variation of the Residual Network (ResNet) architecture that belongs to a family of
models including more complex models like ResNet-50 and ResNet-101, and is especially made for
jobs like image classification. Our primary intuition to begin with ResNet-20 as a baseline model was due to the popularity of
ResNets in image classification tasks. The 20 layer model is most feasible within our resource
limits, while still leveraging the advantages of ResNet architectures. 

### 🔷 WRN-16-4
Wide Residual Networks (Wide-ResNet/WRN) are a variation of ResNets in which the residual
networks’ width is increased and their depth is decreased with the help of Wide residual blocks.
The idea was first proposed by S. Zagoruyko et al.[9] to combat network training issues to increase
accuracy. WRNs have fewer layers but more parameters per layer since they trade off extreme depth for greater width. Compared to highly deep ResNets, this is computationally more efficient and
frequently results in better performance.

The WRN-16-4 is not directly available as a pre-built model in PyTorch’s standard torchvision
models library. PyTorch offers WRN-50-2 and WRN-101-2 but were too large for processing. Hence,
we implemented WRN-16-4 by modifying the existing ResNet architecture. We brought over the
Group normalization, adaptive gradient clipping and weight standardization implementations from
the ResNet-20 model.

## 🔶 Results & Training Details

The models will be developed using PyTorch and the differential privacy library [Opacus](https://ai.meta.com/blog/introducing-opacus-a-high-speed-library-for-training-pytorch-models-with-differential-privacy/). It was trained and tested on Google Colaboratory using Tesla T4 GPU.

### 🔷 Optimizer Details

The Differentially Private PyTorch library that is being used favours gradient
calculation by the Stochastic Gradient Descent (SGD) method. In each iteration, after computing the
gradient in the back propagation, some noise is added to the value. This masks the original data from
the model and avoids memorization of examples. This is especially important for outliers, that are at
a higher chance of being leaked, as they have a greater influence on gradient calculation.

Taking into account the computation, Opacus handles per-sample gradient by implementing an
effective technique. This involves the algorithm re-uses the already computed gradients and further
processes them to obtain per-sample gradients In a traditional ML model, the per-sample gradient
would require O(mnp2) operations, whereas Opacus’s technique does this with O(mnp) operations[11].
It involves computing matrices Z (pre-activation) and H (activation) for the minibatch, the algorithm
computes the per-sample gradient norms without to re-running the backpropagation multiple times.

The hyperparameter values for the setup obtaining best accuracy are as follows:
- **Privacy Budget (ϵ):** 3.0  
- **Privacy Loss Probability (δ):** 1 × 10⁻⁵  
- **Batch Size:**  
  - **Logical Batch Size:** 4096  
  - **Physical Batch Size:** 128  
- **Learning Rate:** 1 × 10⁻³
- **Epochs:** 25-50
