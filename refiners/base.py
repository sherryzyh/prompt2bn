"""
BNSynth Base Refiner Module

This module defines the base interface for Bayesian Network structure refiners
used in BNSynth. Refiners take existing BN structures and improve them using:
- LLM-enhanced methods: Intelligent refinement with LLM reasoning (e.g., ReActBN)
- Traditional methods: Statistical optimization algorithms (e.g., Hill Climbing, PC, MMHC)

All refiners are data-dependent, requiring observation data for structure optimization.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd


class BaseRefiner(ABC):
    """
    Abstract base class for Bayesian Network structure refiners in BNSynth.
    
    Refiners take existing BN structures (from generators or initial graphs) and
    improve them using data-dependent optimization. Supports:
    - LLM-enhanced refinement: Uses LLM reasoning for intelligent structure optimization
    - Traditional refinement: Statistical algorithms for data-driven structure learning
    
    All refiners require observation data for structure optimization.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the refiner.
        
        Args:
            logger: Logger instance for output
        """
        self.logger = logger or logging.getLogger(__name__)
        
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the refiner."""
        pass
    
    @abstractmethod
    def run(
        self,
        desc_variables: str,
        dag_variables: list,
        dag: pd.DataFrame,
        observation: pd.DataFrame,
        init_generation: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Refine an existing Bayesian Network structure.
        
        Args:
            desc_variables: Variable descriptions
            dag_variables: List of variable names
            dag: True DAG structure (for evaluation)
            observation: Observed data for refinement
            init_generation: Initial BN structure to refine
            **kwargs: Additional refiner-specific parameters
            
        Returns:
            Dictionary containing:
            - 'Matrix': Adjacency matrix of refined structure
            - 'Graph': NetworkX graph representation
            - 'Score': Final score achieved
            - 'Refiner': Refiner name
            - Additional refiner-specific results
        """
        pass
    
    def _validate_inputs(
        self, 
        observation: pd.DataFrame, 
        init_generation: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Validate input data and initial generation.
        
        Args:
            observation: Input data to validate
            init_generation: Initial structure to validate
            
        Raises:
            ValueError: If input data is invalid
        """
        if observation is None or observation.empty:
            raise ValueError("Observation data cannot be None or empty")
        
        if len(observation.columns) < 2:
            raise ValueError("At least 2 variables are required for structure refinement")
        
        if init_generation is None:
            self.logger.warning("No initial generation provided, starting from empty structure")
        
        self.logger.info(f"Input validation passed: {len(observation.columns)} variables, {len(observation)} samples")
    
    def _log_results(self, results: Dict[str, Any]) -> None:
        """
        Log refiner results.
        
        Args:
            results: Results dictionary from run()
        """
        self.logger.info(f"Refiner {self.name} completed")
        self.logger.info(f"Final score: {results.get('Score', 'N/A')}")
        
        if 'Matrix' in results:
            matrix = results['Matrix']
            if hasattr(matrix, 'shape'):
                self.logger.info(f"Refined structure: {matrix.shape[0]}x{matrix.shape[1]} adjacency matrix")
        
        if 'Graph' in results:
            graph = results['Graph']
            if hasattr(graph, 'edges'):
                self.logger.info(f"Number of edges in refined structure: {len(graph.edges)}")
