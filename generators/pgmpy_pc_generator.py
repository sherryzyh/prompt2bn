"""
PGMPY PC generator for Bayesian network structure generation.

This module implements the PC algorithm using pgmpy for generating
Bayesian network structures from data.
"""

import logging
from typing import Optional, Dict, Any
import pandas as pd
from pgmpy.estimators import PC

from .base import BaseGenerator


class PgmpyPCGenerator(BaseGenerator):
    """
    PC generator using pgmpy's PC estimator.
    
    This generator uses pgmpy's PC algorithm to generate Bayesian network
    structures. PC is a constraint-based algorithm that learns the skeleton
    and then orients the edges.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the PC generator.
        
        Args:
            logger: Logger instance for output
        """
        super().__init__(model=None, logger=logger)
    
    @property
    def name(self) -> str:
        """Return the name of the generator."""
        return "PgmpyPCGenerator"
    
    def run(
        self,
        desc_variables: str,
        dag_variables: list,
        observation: Optional[pd.DataFrame] = None,
        generations: Optional[list] = None,
        **kwargs
    ) -> tuple[int, Dict[str, Any]]:
        """
        Generate a Bayesian Network structure using PC.
        
        Args:
            desc_variables: Variable descriptions (unused, for compatibility)
            dag_variables: List of variable names
            observation: Observed data for structure learning
            generations: Previous generations (unused, for compatibility)
            **kwargs: Additional generator-specific parameters
            
        Returns:
            Tuple of (validation_status, results_dict) where:
            - validation_status: 1 if valid DAG, 0 otherwise
            - results_dict: Dictionary containing generated structure
        """
        if observation is None:
            raise ValueError("Observation data is required for traditional generators")
        
        self._validate_inputs(observation)
        
        try:
            # Set up the PC estimator
            pc = PC(observation)
            
            # Perform PC estimation
            learned_model = pc.estimate()
            
            # Convert to adjacency matrix
            matrix = self._dag_to_adjacency_matrix(learned_model, dag_variables)
            
            # Create a generation dict for compatibility
            generation = self._dag_to_generation_dict(learned_model, dag_variables)
            
            results = {
                'Generation': generation,
                'Matrix': matrix
            }
            
            self._log_results(results)
            return 1, results  # Always return 1 for valid DAG
            
        except Exception as e:
            self.logger.error(f"Error in PC generation: {e}")
            return 0, {'Generation': None, 'Matrix': None}
    