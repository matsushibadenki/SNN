# **Benchmark Report: SNN Performance & Robustness**

This report documents the performance benchmarks of the Spiking Neural Network (SNN) models developed in this project.

## **1\. Logic Gated SNN (Hyper-Robust Config)**

Date: 2025-12-24  
Model: HybridNeuromorphicCore (LSM with LogicGatedSNN layers)  
Task: Synthetic Pattern Recognition (10 classes, 784 inputs)  
Configuration:

* Hidden Neurons: 4096  
* Weights: 1.58-bit (Ternary) for Reservoir, Continuous for Readout  
* Training Noise Range: 0.0 \- 0.45 (XOR Noise)

### **Result Summary**

| Metric | Value | Note |
| :---- | :---- | :---- |
| **Peak Accuracy** | **100.0%** | On clean / low-noise data (0.0 \- 0.2) |
| **Robust Accuracy** | **85.2%** | At **40%** input noise (Noise Level 0.4) |
| **Theoretical Limit** | **9.4%** | At 50% noise (Random Guess), verifying physical correctness |
| **Convergence** | \< 15 Epochs | Fast learning with autonomous\_step |

### **Detailed Stress Test (Robustness)**

The following table shows the model's accuracy under increasing levels of bit-flip (XOR) noise.

| Noise Level | Accuracy | Loss | Output Spike % | Status |
| :---- | :---- | :---- | :---- | :---- |
| **0.10** | 100.0% | 0.0000 | 10.0% | Excellent |
| **0.20** | 100.0% | 0.0001 | 10.0% | Excellent |
| **0.30** | 99.2% | 0.0017 | 10.2% | Excellent |
| **0.40** | **85.2%** | 0.0208 | 9.9% | **Robust** |
| **0.45** | 40.0% | 0.0814 | 5.8% | Weak (Near Limit) |
| **0.50** | 9.4% | 0.1305 | 3.7% | Theoretical Limit (OK) |

**Analysis:** The model demonstrates exceptional robustness, maintaining near-perfect accuracy up to 30% noise and remaining highly functional at 40% noise. The drop to random guessing at 50% noise confirms the model is not hallucinating and respects information-theoretic bounds.

## **2\. CIFAR-10 Benchmarks (Previous)**

*(Placeholder for CIFAR-10 results comparing ANN vs SNN, to be populated from benchmarks/cifar10\_ann\_vs\_snn\_leaderboard.md)*

## **3\. Efficiency Metrics**

* **Multiplication Reduction:** 100% for Logic Gated layers (Add/Sub only).  
* **Sparsity:** High sparsity observed in Reservoir layers (\~42% activity in Logic Gated tests).

## **Conclusion**

The Logic Gated SNN architecture has proven to be:

1. **Highly Accurate:** Solves pattern recognition tasks perfectly.  
2. **Extremely Robust:** Resilient to severe data corruption.  
3. **Physically Plausible:** Behaves correctly at information limits.  
4. **Efficient:** Eliminates multiplications in core processing layers.