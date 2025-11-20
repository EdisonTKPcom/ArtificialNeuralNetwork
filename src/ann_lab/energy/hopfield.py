"""
Hopfield Network for associative memory.

Stores patterns and retrieves them from partial or noisy inputs.
"""

import torch
import torch.nn as nn
from ..core.base_model import BaseModel


class HopfieldNetwork(BaseModel):
    """
    Hopfield Network for pattern storage and retrieval.
    
    Recurrent neural network that stores patterns as stable states.
    Can retrieve complete patterns from partial or noisy inputs.
    
    Good for: associative memory, pattern completion, error correction.
    
    Args:
        num_neurons: Number of neurons (size of patterns)
    """
    
    def __init__(self, num_neurons: int):
        super().__init__()
        
        self.num_neurons = num_neurons
        
        # Weight matrix (symmetric, zero diagonal)
        self.weights = nn.Parameter(
            torch.zeros(num_neurons, num_neurons),
            requires_grad=False
        )
        
    def train_patterns(self, patterns: torch.Tensor):
        """
        Store patterns using Hebbian learning.
        
        Weight update: w_ij = (1/N) * sum_p (x_i^p * x_j^p)
        where patterns are assumed to be {-1, +1}
        
        Args:
            patterns: Patterns to store (num_patterns, num_neurons)
                     Values should be -1 or +1
        """
        num_patterns = patterns.size(0)
        
        # Hebbian learning rule
        self.weights.data = torch.mm(patterns.t(), patterns) / num_patterns
        
        # Zero out diagonal (no self-connections)
        self.weights.data.fill_diagonal_(0)
        
    def energy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute energy of a state.
        
        E = -0.5 * sum_ij (w_ij * s_i * s_j)
        
        Args:
            state: Current state (num_neurons,) or (batch, num_neurons)
            
        Returns:
            energy: Energy value(s)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # E = -0.5 * state @ weights @ state.T
        energy = -0.5 * torch.sum(state @ self.weights * state, dim=1)
        
        return energy
    
    def update_async(self, state: torch.Tensor, num_iterations: int = 100) -> torch.Tensor:
        """
        Asynchronous update (one neuron at a time).
        
        Args:
            state: Initial state (num_neurons,), values in {-1, +1}
            num_iterations: Number of update iterations
            
        Returns:
            state: Converged state
        """
        state = state.clone()
        
        for _ in range(num_iterations):
            # Choose random neuron
            idx = torch.randint(0, self.num_neurons, (1,)).item()
            
            # Compute activation
            activation = torch.sum(self.weights[idx] * state)
            
            # Update neuron (sign function)
            state[idx] = torch.sign(activation) if activation != 0 else state[idx]
        
        return state
    
    def update_sync(self, state: torch.Tensor) -> torch.Tensor:
        """
        Synchronous update (all neurons at once).
        
        Args:
            state: Current state (num_neurons,) or (batch, num_neurons)
            
        Returns:
            new_state: Updated state
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        # Compute activations for all neurons
        activations = torch.mm(state, self.weights)
        
        # Apply sign function
        new_state = torch.sign(activations)
        new_state[new_state == 0] = 1  # Handle zero case
        
        if squeeze:
            new_state = new_state.squeeze(0)
        
        return new_state
    
    def recall(
        self,
        pattern: torch.Tensor,
        max_iterations: int = 100,
        method: str = 'async'
    ) -> torch.Tensor:
        """
        Retrieve stored pattern from initial state.
        
        Args:
            pattern: Initial (possibly noisy) pattern (num_neurons,)
            max_iterations: Maximum iterations
            method: 'async' or 'sync'
            
        Returns:
            retrieved: Retrieved pattern
        """
        state = pattern.clone()
        
        if method == 'async':
            state = self.update_async(state, max_iterations)
        else:  # sync
            for _ in range(max_iterations):
                new_state = self.update_sync(state)
                if torch.all(new_state == state):
                    break
                state = new_state
        
        return state
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Recall pattern (synchronous update until convergence).
        
        Args:
            x: Input pattern(s) (batch, num_neurons) or (num_neurons,)
            
        Returns:
            output: Retrieved pattern(s)
        """
        return self.recall(x, method='sync')
