"""
PGMPY Hill Climbing refiner for Bayesian network structure refinement.

This module implements the traditional hill climbing algorithm using pgmpy's
HillClimbSearch with BIC scoring for refining existing Bayesian network structures.
"""

import logging
from typing import Optional, Dict, Any
import pandas as pd
from pgmpy.base import DAG
from pgmpy.estimators import HillClimbSearch, BIC
import networkx as nx

from errors.generation_error import InvalidDAGError
from utils.eval_utils import evaluate_generation
from utils.graph_utils import generation_dict_to_discrete_bn
from .base import BaseRefiner


class PgmpyHillClimbingRefiner(BaseRefiner):
    """
    Traditional hill climbing refiner using pgmpy's HillClimbSearch.
    
    This refiner uses pgmpy's built-in hill climbing with BIC scoring
    to refine existing Bayesian network structures from data. It can start
    from an initial structure and improve it.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None, max_iter: int = 20):
        """
        Initialize the hill climbing refiner.
        
        Args:
            logger: Logger instance for output
            max_iter: Maximum number of iterations for hill climbing
        """
        super().__init__(logger)
        self.max_iter = max_iter
    
    @property
    def name(self) -> str:
        """Return the name of the refiner."""
        return "PgmpyHillClimbingRefiner"
    
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
        Refine an existing Bayesian Network structure using hill climbing.
        
        Args:
            desc_variables: Variable descriptions (unused, for compatibility)
            dag_variables: List of variable names
            dag: True DAG structure (for evaluation)
            observation: Observed data for structure refinement
            init_generation: Initial BN structure to refine (optional)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing refined structure and metrics
        """
        self._validate_inputs(observation, init_generation)
        
        try:
            # Create initial structure if provided
            if init_generation is not None:
                initial_dag = generation_dict_to_discrete_bn(init_generation, dag_variables)
                self.logger.info("Starting refinement from provided initial structure")
            else:
                initial_dag = None
                self.logger.info("Starting refinement from empty structure")
            
            # Set up the hill climbing search
            hc = HillClimbSearch(observation)
            scoring_method = BIC(observation)
            
            # Perform hill climbing search
            if initial_dag is not None:
                # Start from the provided initial structure
                learned_model = hc.estimate(
                    scoring_method=scoring_method,
                    max_indegree=4,
                    max_iter=self.max_iter,
                    start_dag=initial_dag
                )
            else:
                # Start from empty structure
                learned_model = hc.estimate(
                    scoring_method=scoring_method,
                    max_indegree=4,
                    max_iter=self.max_iter
                )
            
            # Convert to adjacency matrix
            matrix = self._dag_to_adjacency_matrix(learned_model, dag_variables)
            
            # Create NetworkX graph
            graph = nx.DiGraph()
            graph.add_nodes_from(dag_variables)
            for edge in learned_model.edges():
                graph.add_edge(edge[0], edge[1])
            
            # Calculate final score
            final_score = scoring_method.score(learned_model)
            
            # Calculate initial metrics if we had an initial structure
            init_nhd = None
            init_shd = None
            if init_generation is not None:
                init_matrix = self._dag_to_adjacency_matrix(initial_dag, dag_variables)
                _, _, _, _, init_nhd, init_shd = evaluate_generation(
                    dag_input=dag,
                    pred_input=init_matrix,
                    logger=self.logger,
                )
            
            # Calculate final metrics
            _, _, f1_score, accuracy, final_nhd, final_shd = evaluate_generation(
                dag_input=dag,
                pred_input=matrix,
                logger=self.logger,
            )
            
            results = {
                'Matrix': matrix,
                'Graph': graph,
                'Score': final_score,
                'Refiner': self.name,
                'init_nhd': init_nhd,
                'init_shd': init_shd,
                'final_nhd': final_nhd,
                'final_shd': final_shd,
                'MoveCount': 0,  # Hill climbing doesn't track individual moves
                'MetricsHistory': [],
                'ActionHistory': []
            }
            
            self._log_results(results)
            return results
            
        except Exception as e:
            self.logger.error(f"Error in hill climbing refinement: {e}")
            raise
    
    def _dag_to_adjacency_matrix(self, dag: DAG, dag_variables: list) -> pd.DataFrame:
        """Convert a DAG to adjacency matrix format."""
        matrix = pd.DataFrame(0, index=dag_variables, columns=dag_variables)
        
        for edge in dag.edges():
            parent, child = edge
            if parent in dag_variables and child in dag_variables:
                matrix.loc[parent, child] = 1
        
        return matrix
