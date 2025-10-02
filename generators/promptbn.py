from typing import Any, Optional
import json

from utils.graph_utils import construct_matrix_from_nodes
from .base import BaseGenerator
from errors.generation_error import ParseResponseError, MaxRetriesError

BASELINE_PROMPT = """
You are an expert in building Bayesian Networks. You will receive variables in table format with columns [node, var_name, var_description, var_distribution]. Your task is to construct a Bayesian Network as a Directed Acyclic Graph (DAG) based on the table following the instructions below.

[Instructions]
1. Parse the table to understand:
   - The node id (node).
   - The variable name (var_name).
   - The semantic meaning of each variable (var_description).
   - The distribution type (var_distribution).

2. Construct a Bayesian Network as a Directed Acyclic Graph (DAG) by:
   - Proposing parents (if any) for each node based on relevant domain knowledge or any clues from the table.
   - Ensuring no cycles exist.

3. Provide a strict JSON object with the following structure:
   {{
     "bn": {{
       "nodes": [
         {{
           "node_id": <integer>,
           "node_name": <string>,
           "parents": [<string>, ...],
           "description": <string>,
           "distribution": <string>,
           "conditional_probability_table": <string>
         }},
         ...
       ],
       "edges": [
         {{
           "from": <string>,
           "to": <string>,
           "justification": <string>
         }},
         ...
       ],
       "network_summary": <string>
     }}
   }}


[Output Format]
1. In "nodes":
   - "node_id": A unique ID or index for each node.
   - "node_name": The variable name from var_name column.
   - "parents": Array of node_name values for parent nodes.
   - "description": Short text from var_description.
   - "distribution": Data from var_distribution column.
   - "conditional_probability_table": e.g. "P(tub | asia)".

2. In "edges":
   - "from" and "to": References to the "node_name" fields, indicating parent-child relationships.
   - "justification": A concise reason for why this relationship exists.

3. "network_summary": A concise explanation of how the Bayesian Network structure was derived.

4. Output ONLY the valid JSON object, with no additional commentary or text.

[Variables]
{desc_variables}
"""

class PromptBNGenerator(BaseGenerator):
    """
    PromptBN Generator for BNSynth.
    
    LLM-based Bayesian Network structure generator that creates networks from
    variable descriptions without requiring observation data (data-free generation).
    Uses prompt engineering to leverage LLM's domain knowledge for structure creation.
    """

    def __init__(self, model: Any, logger: Optional[Any] = None) -> None:
        super().__init__(model, logger)
        self.prompt = BASELINE_PROMPT
        self.messages = []
        self.logger.debug(f"[Model] {self.model}")
        self.logger.debug(f"[Client] {type(self.client)}")
    
    @property
    def name(self) -> str:
        """Return the name of the generator."""
        return "PromptBNGenerator"

    def prepare_user_prompt(self, desc_variables: Any, dag_variables: Any) -> str:
        """
        Format the user prompt with variable descriptions and DAG variables.
        """
        prompt = self.prompt.format(desc_variables=desc_variables)
        return prompt

    def prepare_messages(self, desc_variables: Any, dag_variables: Any) -> list:
        """
        Prepare the message list for the language model.
        """
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant that constructs Bayesian Networks."},
            {"role": "user", "content": self.prepare_user_prompt(desc_variables, dag_variables)}
        ]
        self.logger.debug(f"[Prompt] {self.messages[-1]['content']}")
        return self.messages

    def construct_matrix(self, generation, dag_variables):
        return construct_matrix_from_nodes(generation, dag_variables)

    def parse_response(self, response: str) -> Any:
        """
        Parse the response from the language model.
        """
        # Remove wrapping triple backticks (with or without 'json')
        content = response.replace("```json\n", "").replace("\n```", "").strip()
        try:
            return json.loads(content, strict=False)
        except json.JSONDecodeError as e:
            self.logger.debug(f"Unjsonified Raw Generation:\n```json\n{response}\n```")
            self.logger.exception(f"JSON Decode Error: {e}")
            raise ParseResponseError(f"JSON Decode Error: {e}") from e
    
    def run(self, desc_variables: str, dag_variables: list, observation=None, generations=None):
        """
        Generate a Bayesian Network structure using LLM.
        
        Args:
            desc_variables: Variable descriptions
            dag_variables: List of variable names
            observation: Observed data (unused for LLM-based generation)
            generations: Previous generations (unused)
            
        Returns:
            Tuple of (dag_validation, results_dict) where:
            - dag_validation: 1 if valid DAG, 0 otherwise
            - results_dict: Dictionary containing 'Generation' and 'Matrix'
        """
        self.messages = self.prepare_messages(desc_variables, dag_variables)
        try:    
            generation = self.generate()
        except ParseResponseError as e:
            self.logger.exception(f"Parse Response Error: {e}")
            raise e
        except MaxRetriesError as e:
            self.logger.exception(f"Max Retries Error: {e}; Retrying...")
            raise e
        
        try:
            matrix = self.construct_matrix(generation, dag_variables)
        except Exception as e:
            self.logger.exception(f"Invalid DAG Error: {e}")
            raise e
        else:
            self.logger.debug(f"Valid DAG, Raw generation:\n{generation}")
            self.logger.debug(f"Constructed Matrix:\n{matrix}")
        
        results = {
            "Generation": generation,
            "Matrix": matrix
        }
        return 1, results