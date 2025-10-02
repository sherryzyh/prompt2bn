"""
PGMPY Hill Climbing generator for Bayesian network structure generation.

This module implements the traditional hill climbing algorithm using pgmpy's
HillClimbSearch with BIC scoring for generating Bayesian network structures from data.
"""

import logging
from typing import Optional, Dict, Any
import pandas as pd
from pgmpy.estimators import HillClimbSearch, BIC

from .base import BaseGenerator


class PgmpyHillClimbingGenerator(BaseGenerator):
    """
    Traditional hill climbing generator using pgmpy's HillClimbSearch.
    
    This generator uses pgmpy's built-in hill climbing with BIC scoring
    to generate Bayesian network structures from data.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None, max_iter: int = 20):
        """
        Initialize the hill climbing generator.
        
        Args:
            logger: Logger instance for output
            max_iter: Maximum number of iterations for hill climbing
        """
        super().__init__(model=None, logger=logger)
        self.max_iter = max_iter
    
    @property
    def name(self) -> str:
        """Return the name of the generator."""
        return "PgmpyHillClimbingGenerator"
    
    def run(
        self,
        desc_variables: str,
        dag_variables: list,
        observation: Optional[pd.DataFrame] = None,
        generations: Optional[list] = None,
        **kwargs
    ) -> tuple[int, Dict[str, Any]]:
        """
        Generate a Bayesian Network structure using hill climbing.
        
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
            # Set up the hill climbing search
            hc = HillClimbSearch(observation)
            scoring_method = BIC(observation)
            
            # Perform hill climbing search
            learned_model = hc.estimate(
                scoring_method=scoring_method,
                max_indegree=4,
                max_iter=self.max_iter
            )
            
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
            self.logger.error(f"Error in hill climbing generation: {e}")
            return 0, {'Generation': None, 'Matrix': None}
    