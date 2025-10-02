"""
BNSynth Base Generator Module

This module defines the base interface for Bayesian Network structure generators
used in BNSynth. Supports both:
- LLM-based generators: Data-free structure generation using language models (e.g., PromptBN)
- Traditional generators: Data-dependent structure learning using statistical methods (e.g., Hill Climbing, PC, MMHC)
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd

from llm import LLMClient
from errors.generation_error import ParseResponseError, MaxRetriesError


class BaseGenerator(ABC):
    """
    Abstract base class for Bayesian Network structure generators in BNSynth.
    
    This base class supports both:
    - LLM-based generators: Data-free structure generation (e.g., PromptBN)
    - Traditional generators: Data-dependent structure learning (e.g., Hill Climbing, PC, MMHC)
    
    LLM-based generators can create structures without observation data, while
    traditional generators require observation data for statistical structure learning.
    """
    
    def __init__(
        self, 
        model: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize the generator.
        
        Args:
            model: LLM model name (required for LLM-based generators)
            logger: Logger instance for output
        """
        self.model = model
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize LLM client only if model is provided
        if model is not None:
            self.client = LLMClient.from_model(model, self.logger)
            self.messages = [
                {"role": "system", "content": "You are a helpful assistant that constructs Bayesian Networks."},
            ]
            self.prompt = None
        else:
            self.client = None
            self.messages = None
            self.prompt = None
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the generator."""
        pass
    
    @property
    def is_llm_based(self) -> bool:
        """Return True if this is an LLM-based generator."""
        return self.model is not None
    
    @abstractmethod
    def run(
        self,
        desc_variables: str,
        dag_variables: list,
        observation: Optional[pd.DataFrame] = None,
        generations: Optional[list] = None,
        **kwargs
    ) -> tuple[int, Dict[str, Any]]:
        """
        Generate a Bayesian Network structure.
        
        Args:
            desc_variables: Variable descriptions
            dag_variables: List of variable names
            observation: Observed data for structure learning (required for traditional generators)
            generations: Previous generations (unused, for compatibility)
            **kwargs: Additional generator-specific parameters
            
        Returns:
            Tuple of (validation_status, results_dict) where:
            - validation_status: 1 if valid DAG, 0 otherwise
            - results_dict: Dictionary containing:
                - 'Generation': Generated structure representation
                - 'Matrix': Adjacency matrix of learned structure
                - Additional generator-specific results
        """
        pass
    
    # LLM-specific methods (only used by LLM-based generators)
    def prepare_messages(self, desc_variables: str, dag_variables: list) -> list:
        """
        Prepare messages for LLM generation.
        Only implemented by LLM-based generators.
        """
        if not self.is_llm_based:
            raise NotImplementedError("prepare_messages() only available for LLM-based generators")
        raise NotImplementedError("Subclasses must implement .prepare_messages()")
    
    def parse_response(self, response: str) -> Any:
        """
        Parse LLM response.
        Only implemented by LLM-based generators.
        """
        if not self.is_llm_based:
            raise NotImplementedError("parse_response() only available for LLM-based generators")
        raise NotImplementedError("Subclasses must implement .parse_response()")
    
    def generate(self, max_retries: int = 3) -> Any:
        """
        Generate a Bayesian Network structure using the language model.
        Only used by LLM-based generators.
        """
        if not self.is_llm_based:
            raise NotImplementedError("generate() only available for LLM-based generators")
        
        for attempt in range(1, max_retries + 1):
            self.logger.debug(f"Generating... ({attempt}/{max_retries})")
            response = self.client.chat(messages=self.messages)
            self.logger.debug(f"Raw response:\n{response}")
            try:
                generation = self.parse_response(response)
                self.logger.debug(f"JSON output:\n{generation}")
                return generation
            except ParseResponseError as e:
                self.logger.exception(f"Parse Response Error on attempt {attempt}: {e}")
                raise e
            except Exception as e:
                self.logger.exception(f"Exception on attempt {attempt}: {e}; Retrying...")
                continue
        raise MaxRetriesError("Max retries reached")
    
    def construct_matrix(self, generation: Any, dag_variables: list) -> pd.DataFrame:
        """
        Construct adjacency matrix from generation.
        Only implemented by LLM-based generators.
        """
        if not self.is_llm_based:
            raise NotImplementedError("construct_matrix() only available for LLM-based generators")
        raise NotImplementedError("Subclasses must implement .construct_matrix()")
    
    # Utility methods (available to all generators)
    def _validate_inputs(self, observation: Optional[pd.DataFrame] = None) -> None:
        """
        Validate input data.
        
        Args:
            observation: Input data to validate (required for traditional generators)
            
        Raises:
            ValueError: If input data is invalid
        """
        if not self.is_llm_based and (observation is None or observation.empty):
            raise ValueError("Observation data is required for traditional generators")
        
        if observation is not None and len(observation.columns) < 2:
            raise ValueError("At least 2 variables are required for structure learning")
        
        if observation is not None:
            self.logger.info("Input validation passed: %d variables, %d samples", len(observation.columns), len(observation))
    
    def _log_results(self, results: Dict[str, Any]) -> None:
        """
        Log generator results.
        
        Args:
            results: Results dictionary from run()
        """
        self.logger.info("Generator %s completed", self.name)
        
        if 'Matrix' in results:
            matrix = results['Matrix']
            if hasattr(matrix, 'shape'):
                self.logger.info("Generated structure: %dx%d adjacency matrix", matrix.shape[0], matrix.shape[1])
        
        if 'Generation' in results:
            self.logger.info("Structure generation completed successfully")
    
    def _dag_to_adjacency_matrix(self, dag, dag_variables: list) -> pd.DataFrame:
        """
        Convert a DAG to adjacency matrix format.
        Utility method for traditional generators.
        """
        matrix = pd.DataFrame(0, index=dag_variables, columns=dag_variables)
        
        for edge in dag.edges():
            parent, child = edge
            if parent in dag_variables and child in dag_variables:
                matrix.loc[parent, child] = 1
        
        return matrix
    
    def _dag_to_generation_dict(self, dag, dag_variables: list) -> Dict[str, Any]:
        """
        Convert a DAG to generation dict format for compatibility.
        Utility method for traditional generators.
        """
        nodes = []
        edges = []
        
        for node in dag_variables:
            parents = [p for p in dag.predecessors(node)]
            nodes.append({
                "node_id": dag_variables.index(node),
                "node_name": node,
                "parents": parents,
                "description": f"Variable {node}",
                "distribution": "discrete",
                "conditional_probability_table": "To be estimated"
            })
        
        for edge in dag.edges():
            edges.append({
                "from": edge[0],
                "to": edge[1],
                "justification": "Learned from data using %s" % self.name
            })
        
        return {
            "bn": {
                "nodes": nodes,
                "edges": edges,
                "network_summary": "Bayesian network learned using %s with %d variables and %d edges" % (self.name, len(nodes), len(edges))
            }
        }