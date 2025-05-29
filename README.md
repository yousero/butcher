# Butcher Chess Engine: Neural Network-Based Chess AI
Butcher is an open-source chess engine inspired by LC0, designed to predict the best next move using a deep neural network without Monte Carlo Tree Search (MCTS). Unlike traditional engines, Butcher focuses on direct policy prediction through neural networks for efficient move selection.

## Key Features

### Neural Network Architecture
- Deep convolutional neural network with residual blocks
- Input: 8x8x18 tensor representation of chess positions
- Output: Policy vector for move prediction
- Built-in regularization with L2 weight decay and dropout
- Batch normalization for stable training

### Training Capabilities
- Training on chess puzzles for supervised learning
- Self-play training loop for reinforcement learning
- Configurable training parameters (batch size, learning rate, epochs)
- Automatic model checkpointing and saving
- Support for resuming training from saved models
