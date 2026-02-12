# Chess Reasoning LLM Fine-Tuning Pipeline

This notebook implements a two-stage fine-tuning pipeline to train an **Unsloth-optimized LLM** (DeepSeek-R1-Distill-Qwen) for **chess move prediction and structured reasoning**.

## Overview

The training workflow consists of:

1. **Supervised Fine-Tuning (SFT)**  
   Establishes a structured *Chain-of-Thought (CoT)* response format for chess reasoning.

2. **Group Relative Policy Optimization (GRPO)**  
   Refines model performance using the **Stockfish engine** as a reward signal, evaluating:
   - Move legality  
   - Move optimality  

## Features

- ♟️ FEN string parsing into text-based board representations  
- 📊 Dynamic dataset generation  
- 🧠 Structured reasoning output format  
- ⚙️ Stockfish-based reward modeling  
- 📈 Final evaluation script benchmarking model accuracy against Stockfish  

## Objective

Train a reasoning-capable LLM that produces legally valid and strategically sound chess moves while maintaining interpretable step-by-step reasoning.
