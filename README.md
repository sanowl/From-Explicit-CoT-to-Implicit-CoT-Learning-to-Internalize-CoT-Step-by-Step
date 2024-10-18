# Chain-of-Thought (CoT) Training with Stepwise Internalization

## Project Overview

This project implements a training pipeline for language models using the Chain-of-Thought (CoT) approach with Stepwise Internalization. The objective is to enhance the model's capacity for complex reasoning tasks, with a particular focus on mathematical problem-solving.

## Research Foundation

This implementation is based on the paper: [https://arxiv.org/pdf/2405.14838](https://arxiv.org/pdf/2405.14838)

## Key Components

1. **CoTDataset**: A custom dataset class for handling Chain-of-Thought data.
2. **StepwiseInternalization**: The primary training class implementing the stepwise internalization approach.
3. **AWS SageMaker Integration**: Configuration for cloud-based training on AWS SageMaker.

## Implemented Features

- Data preprocessing for multiplication problems and GSM8K dataset.
- Custom collate function for managing variable-length sequences.
- Stepwise internalization training loop.
- Mixed precision training utilizing PyTorch's autocast and GradScaler.
- Learning rate scheduling with linear warmup.
- Evaluation metrics for generated sequences.
- AWS SageMaker integration for scalable training.

## Model Architecture

The project utilizes GPT-2 (medium by default) as the base language model. The model is fine-tuned on CoT data using the stepwise internalization approach.

## Training Process

1. The model is trained on multiplication problems of varying difficulty.
2. It is further trained on the GSM8K dataset for general math word problems.
3. The training process employs stepwise internalization, gradually removing tokens from the chain-of-thought to encourage the model to internalize reasoning steps.

## Mathematical Framework

The key mathematical concepts underlying this project include:

1. **Stepwise Internalization**: 
   The number of tokens removed at step $$t$$ is given by:

   $$R(t) = \left\lfloor\frac{\delta t}{T}\right\rfloor$$

   Where:
   - $$\delta$$ is the token removal rate
   - $$T$$ is the total number of training steps

2. **Smoothing Function**:
   To smooth the token removal process, we add a random offset:

   $$O = \min\{k : U < e^{-\lambda k}\}$$

   Where:
   - $$U$$ is a uniform random variable on $$[0, 1]$$
   - $$\lambda$$ is the smoothing parameter

3. **Loss Function**:
   The model is trained to minimize the negative log-likelihood:

   $$\mathcal{L} = -\sum_{i=1}^{N} \log P(y_i|x_i, \theta)$$

   Where:
   - $$x_i$$ is the input sequence
   - $$y_i$$ is the target output
   - $$\theta$$ represents the model parameters

## AWS SageMaker Setup

The project includes scripts for configuring and executing training jobs on AWS SageMaker, enabling scalable and managed training in the cloud environment.

## Usage Instructions

To train the model:

1. Configure an AWS SageMaker environment.
2. Upload the training script and data to an S3 bucket.
3. Execute the `setup_sagemaker_training()` function to initiate a training job.

For local training or testing, use the following command:

```
python cot_training_script.py --epochs 10 --batch_size 32 --learning_rate 5e-5 --model_name gpt2-medium
```

