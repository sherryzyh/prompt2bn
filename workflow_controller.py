"""
BNSynth Workflow Controller for Bayesian Network Synthesis

This module provides the main workflow controller for BNSynth that orchestrates 
generation and refinement workflows. It supports:
- Generation: Create BN structures using LLM-based or traditional generators (data-free or data-dependent)
- Refinement: Refine existing structures using LLM-enhanced or traditional refiners (data-dependent)
- Pipeline: Generate then refine in sequence

The controller enables flexible BN synthesis combining:
- LLM-enforced methods (PromptBN, ReActBN) with traditional algorithms (Hill Climbing)
- Data-dependent structural learning (using observation data) with data-free generation
- Initial graph refinement with de novo structure generation
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import os

from utils.data_utils import load_data, load_observation, load_generations
from utils.eval_utils import evaluate_generation
from utils.graph_utils import construct_matrix_from_bn_dict
from errors.generation_error import MaxRetriesError, InvalidDAGError, ParseResponseError
from errors.pipeline_error import MissingGenerationsError
from utils.path_utils import get_experiment_path


@dataclass
class WorkflowResult:
    """Result container for workflow execution."""
    workflow_type: str
    success: bool
    dataset: str
    generation_results: Optional[pd.DataFrame] = None
    refinement_results: Optional[pd.DataFrame] = None
    error: Optional[str] = None


class WorkflowController:
    """
    Main workflow controller for BNSynth: Bayesian Network Synthesis Framework.
    
    This controller orchestrates three types of workflows:
    1. Generation: Create BN structures using LLM-based (data-free) or traditional (data-dependent) generators
    2. Refinement: Refine existing structures using LLM-enhanced or traditional refiners (data-dependent)
    3. Pipeline: Execute generation followed by refinement in sequence
    
    Supports flexible combination of:
    - LLM-enforced methods (PromptBN, ReActBN) and traditional algorithms (Hill Climbing)
    - Data-dependent structural learning and data-free generation
    - Initial graph refinement and de novo structure generation
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        exp_data_dir: str,
    ):
        """
        Initialize the workflow controller.
        
        Args:
            config: Complete configuration dictionary
            logger: Logger instance for output
            exp_data_dir: Experiment data directory
        """
        self.config = config
        self.logger = logger
        self.exp_data_dir = exp_data_dir
        self.workflow_type = config['workflow']
        
    def _load_data_for_dataset(self, dataset):
        """Load DAG, variables, descriptions, and observation for a specific dataset."""
        dag, dag_variables, desc_variables = load_data(
            self.config['data']['input_data']['source'],
            dataset,
            self.logger,
            self.config,
        )
        observation = None
        if self.config.get('observation', 0) > 0:
            observation = load_observation(
                self.config['data']['input_data']['source'],
                dataset,
                self.config['observation'],
                self.config,
            )
            self.logger.info(f"[Data] {len(observation)} observations loaded for dataset {dataset}")
        return dag, dag_variables, desc_variables, observation
    
    def _load_generator(self) -> Any:
        """Load and instantiate the generator for generation workflow."""
        generator_config = self.config['generation']
        generator_name = generator_config['generator']
        
        if generator_name == 'promptbn':
            from generators.promptbn import PromptBNGenerator as Generator
            generator = Generator(
                model=generator_config['model'],
                logger=self.logger
            )
        elif generator_name == 'pgmpy_hill_climbing':
            from generators.pgmpy_hill_climbing_generator import PgmpyHillClimbingGenerator as Generator
            generator = Generator(logger=self.logger)
        elif generator_name == 'pgmpy_mmhc':
            from generators.pgmpy_mmhc_generator import PgmpyMMHCGenerator as Generator
            generator = Generator(logger=self.logger)
        elif generator_name == 'pgmpy_pc':
            from generators.pgmpy_pc_generator import PgmpyPCGenerator as Generator
            generator = Generator(logger=self.logger)
        else:
            raise ValueError(f"Unknown generator: {generator_name}")
        
        self.logger.info(f"[Generator] {generator_name} generator loaded")
        return generator
    
    def _load_refiner(self) -> Any:
        """Load and instantiate the refiner for refinement workflow."""
        refinement_config = self.config['refinement']
        # Map algorithm names to their refiner classes
        # (config['refinement']['llm']['algorithm'] is kept for backward compatibility with config files)
        algorithm = refinement_config['refiner']
        if algorithm == 'reactbn':
            from refiners.react_bn_agent import ReActBNAgent as Refiner
            refiner = Refiner(refinement_config['model'], self.logger)
        elif algorithm == 'hill_climbing':
            from refiners.pgmpy_hill_climbing_refiner import PgmpyHillClimbingRefiner as Refiner
            refiner = Refiner(self.logger)
        else:
            raise ValueError(f"Unknown refiner: {algorithm}. Available refiners: 'reactbn', 'hill_climbing'")
        self.logger.info("[Refiner] %s loaded", algorithm)
        return refiner
    
    def execute_generation_workflow(
        self,
        dataset,
        generator,
    ) -> WorkflowResult:
        """
        Execute the generation workflow for a specific dataset using a provided generator.
        Raises an error if dataset is None.
        """
        if dataset is None:
            raise ValueError("A dataset must be provided to execute_generation_workflow.")
        if generator is None:
            raise ValueError("A generator must be provided to execute_generation_workflow.")
        dag, dag_variables, desc_variables, observation = self._load_data_for_dataset(dataset)
        self.logger.info(f"[Data] DAG, variables, and descriptions loaded for dataset {dataset}")
        
        try:
            result_df = self._run_generation_experiment(
                generator=generator,
                dataset=dataset,
                desc_variables=desc_variables,
                dag_variables=dag_variables,
                observation=observation,
                dag=dag,
            )
            
            self.logger.info("Generation workflow completed successfully")
            return WorkflowResult(
                workflow_type="generation",
                success=True,
                dataset=dataset,
                generation_results=result_df
            )
            
        except (InvalidDAGError, ParseResponseError, MaxRetriesError, MissingGenerationsError) as e:
            self.logger.error("Generation workflow failed: %s", e, exc_info=True)
            return WorkflowResult(
                workflow_type="generation",
                success=False,
                dataset=dataset,
                error=str(e)
            )
    
    def execute_refinement_workflow(
        self,
        dataset,
        refiner,
    ) -> WorkflowResult:
        """
        Execute the refinement workflow for a specific dataset using a provided refiner.
        Raises an error if dataset is None.
        """
        if dataset is None:
            raise ValueError("A dataset must be provided to execute_refinement_workflow.")
        if refiner is None:
            raise ValueError("A refiner must be provided to execute_refinement_workflow.")
        
        dag, dag_variables, desc_variables, observation = self._load_data_for_dataset(dataset)
        self.logger.info(f"[Data] DAG, variables, and descriptions loaded for dataset {dataset}")
        
        try:
            refinement_config = self.config['refinement']
            source_experiment = refinement_config['data'].get('source_experiment')
            generations = load_generations(
                dataset,
                init_generator=refinement_config['data']['init_generator'],
                init_model=refinement_config['data']['init_model'],
                config=self.config,
                source_experiment=source_experiment,
            )
            self.logger.info(f"[Refinement] {len(generations)} initial generations loaded")
            
            result_df = self._run_refinement_experiment(
                refiner=refiner,
                dataset=dataset,
                generations=generations,
                observation=observation,
                desc_variables=desc_variables,
                dag_variables=dag_variables,
                dag=dag,
            )
            
            self.logger.info("Refinement workflow completed successfully")
            return WorkflowResult(
                workflow_type="refinement",
                success=True,
                dataset=dataset,
                refinement_results=result_df
            )
            
        except (InvalidDAGError, ParseResponseError, MaxRetriesError, MissingGenerationsError) as e:
            self.logger.error("Refinement workflow failed: %s", e, exc_info=True)
            return WorkflowResult(
                workflow_type="refinement",
                success=False,
                dataset=dataset,
                error=str(e)
            )
    
    def execute_pipeline_workflow(
        self,
        dataset,
        generator,
        refiner
    ) -> WorkflowResult:
        """
        Execute the pipeline workflow for a specific dataset using provided generator and refiner.
        Raises an error if dataset is None.
        """
        if dataset is None:
            raise ValueError("A dataset must be provided to execute_pipeline_workflow.")
        self.logger.info(f"[Data] DAG, variables, and descriptions loaded for dataset {dataset}")
        
        self.logger.info("=== Starting Pipeline Workflow ===")
        
        try:
            # Step 1: Execute generation
            self.logger.info("Step 1: Executing generation workflow")
            generation_result = self.execute_generation_workflow(dataset, generator)
            
            if not generation_result.success:
                self.logger.error("Generation workflow failed, aborting pipeline")
                return WorkflowResult(
                    workflow_type="pipeline",
                    success=False,
                    dataset=dataset,
                    error=f"Generation failed: {generation_result.error}"
                )
            
            # Step 2: Execute refinement using generated structures
            self.logger.info("Step 2: Executing refinement workflow")
            refinement_result = self.execute_refinement_workflow(dataset, refiner)
            
            if not refinement_result.success:
                self.logger.error("Refinement workflow failed")
                return WorkflowResult(
                    workflow_type="pipeline",
                    success=False,
                    dataset=dataset,
                    error=f"Refinement failed: {refinement_result.error}"
                )
            
            self.logger.info("Pipeline workflow completed successfully")
            return WorkflowResult(
                workflow_type="pipeline",
                success=True,
                dataset=dataset,
                refinement_results=refinement_result.refinement_results,
            )
            
        except (InvalidDAGError, ParseResponseError, MaxRetriesError, MissingGenerationsError) as e:
            self.logger.error("Pipeline workflow failed: %s", e, exc_info=True)
            return WorkflowResult(
                workflow_type="pipeline",
                success=False,
                dataset=dataset,
                error=str(e)
            )
    
    def _run_generation_experiment(
        self,
        generator,
        dataset,
        desc_variables,
        dag_variables,
        observation,
        dag,
    ) -> pd.DataFrame:
        """Run the generation experiment (adapted from run_generation.py)."""
        import time
        from utils.io_utils import write_error_result, write_result
        count = 0
        count_from_last_validated = 0
        validated_count = 0
        generations = []
        results = {
            "dataset": [],
            "run": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "accuracy": [],
            "nhd": [],
            "shd": [],
            "latency": []
        }
        
        generation_config = self.config['generation']
        repeated_run = generation_config.get('repeated_run', 3)
        
        while True:
            starttime = time.time()
            count += 1
            
            try:
                prev_generations = None
                dag_validation, llm_results = generator.run(
                    desc_variables,
                    dag_variables,
                    observation=observation,
                    generations=prev_generations,
                )
            except InvalidDAGError as e:
                self.logger.exception(f"Invalid DAG: {e} in the {count}th run.")
                if count - count_from_last_validated >= 5:
                    self.logger.exception("Encountered 5 consecutive errors. Stop the experiment.")
                    write_error_result(
                        self.config,
                        dataset,
                        self.exp_data_dir,
                        "InvalidDAG",
                        self.logger,
                    )
                    raise
                continue
            except ParseResponseError:
                self.logger.exception("Unparsable Raw Generation in %dth run", count)
                if count - count_from_last_validated >= 5:
                    self.logger.exception("Encountered 5 consecutive errors. Stop the experiment.")
                    write_error_result(
                        self.config,
                        dataset,
                        self.exp_data_dir,
                        "UnparsableResponse",
                        self.logger,
                    )
                    raise
                continue
            except MaxRetriesError:
                self.logger.exception(
                    f"Forced stop the experiment (PromptBN generator, {generation_config['llm']['model']}, {dataset}),"
                    f"reached max retries."
                )
                write_error_result(
                    self.config,
                    dataset,
                    self.exp_data_dir,
                    "MaxRetries",
                    self.logger,
                )
                raise
            except MissingGenerationsError as e:
                self.logger.exception(f"ValueError: {e}")
                write_error_result(
                    self.config,
                    dataset,
                    self.exp_data_dir,
                    "MissingGenerations",
                    self.logger,
                )
                raise

            endtime = time.time()
            if dag_validation == 1:
                validated_count += 1
                count_from_last_validated = count
                self.logger.info(f"Current Run: {count}, Validated Run: {validated_count}")
                generations.append(llm_results['Generation'])
                precision, recall, f1_score, accuracy, nhd, shd = evaluate_generation(
                    dag_input=dag,
                    pred_input=llm_results['Matrix'],
                    logger=self.logger,
                )
                self.logger.info(f"nhd: {nhd:.4f}; shd: {shd:.4f}; f1_score: {f1_score:.4f}; accuracy: {accuracy:.4f}")
                results["dataset"].append(dataset)
                results["run"].append(validated_count)
                results["precision"].append(precision)
                results["recall"].append(recall)
                results["f1_score"].append(f1_score)
                results["accuracy"].append(accuracy)
                results["nhd"].append(nhd)
                results["shd"].append(shd)
                results['latency'].append(f"{int(endtime-starttime)}")
                
            if validated_count >= repeated_run:
                self.logger.info(f"Hit run requirement. Total Run Count: {count}")
                break
        
        result_df = pd.DataFrame(results)
        result_df['total_run'] = count
        result_df['validated_run'] = validated_count
        result_df['dag_ratio'] = validated_count/count
        result_df['generator'] = 'promptbn'
        result_df['sample'] = self.config['observation']
        
        write_result(
            self.config,
            dataset,
            self.exp_data_dir,
            result_df,
            generations,
            self.logger,
        )
        
        return result_df
    
    def _run_refinement_experiment(
        self,
        refiner,
        dataset,
        generations,
        observation,
        desc_variables,
        dag_variables,
        dag,
    ) -> pd.DataFrame:
        """Run the refinement experiment (adapted from run_refine.py)."""
        import numpy as np
        from utils.data_utils import save_history, save_matrix_graph
        from utils.io_utils import write_result
        
        exp_results = []
        
        for idx, init_generation in enumerate(generations):
            run_idx = idx + 1
            
            if self.config.get('observation', 0) == 0:
                observation = None
            else:
                observation = load_observation(
                    source=self.config['data']['input_data']['source'],
                    dataset=dataset,
                    sample_size=self.config['observation'],
                    config=self.config,
                )
                
            self.logger.info("[Data] %d Observations loaded with random seed %d", len(observation), run_idx)
            
            if init_generation is None:
                init_shd = np.inf
                init_nhd = np.inf
            else:
                init_generation_df = construct_matrix_from_bn_dict(
                    bn_dict=init_generation,
                    dag_variables=dag_variables,
                )
                _, _, _, _, init_nhd, init_shd = evaluate_generation(
                    dag_input=dag,
                    pred_input=init_generation_df,
                    logger=self.logger,
                )
            
            self.logger.info("run: %d, initial SHD: %.4f, NHD: %.4f", run_idx, init_shd, init_nhd)
            
            result_record = self._run_single_refinement(
                refiner=refiner,
                observation=observation,
                init_generation=init_generation,
                desc_variables=desc_variables,
                dag_variables=dag_variables,
                dag=dag,
            )
            
            self.logger.info(
                "run: %d, SHD: %.4f -> %.4f, NHD: %.4f -> %.4f",
                run_idx,
                init_shd, result_record['final_shd'], init_nhd, result_record['final_nhd']
            )
            
            exp_results.append({
                "run": run_idx,
                "init_nhd": init_nhd,
                "init_shd": init_shd,
                "final_nhd": result_record['final_nhd'],
                "final_shd": result_record['final_shd'],
                "final_score": result_record['final_score'],
                "move_count": result_record['move_count'],
                "latency": result_record['latency'],
                "try_count": result_record['try_count'],
            })

            save_history(
                metrics_history=result_record['MetricsHistory'],
                action_history=result_record['ActionHistory'],
                config=self.config,
                run_idx=run_idx,
            )
            save_matrix_graph(
                matrix=result_record['Matrix'],
                graph=result_record['Graph'],
                node_order=dag_variables,
                config=self.config,
                run_idx=run_idx,
            )

        result_df = pd.DataFrame(exp_results)
        result_df['dataset'] = dataset
        # When writing the output DataFrame for refinement, use 'refiner' column
        result_df['refiner'] = self.config['refinement']['refiner']
        result_df['init_generator'] = self.config['refinement']['data']['init_generator']
        result_df['init_model'] = self.config['refinement']['data']['init_model']
        result_df['source_experiment'] = self.config['refinement']['data']['source_experiment'] if self.workflow_type == 'refinement' else "self"
        result_df['model'] = self.config['refinement']['model']
        result_df['sample'] = self.config['observation']
        
        write_result(
            config=self.config,
            dataset=dataset,
            exp_data_root=self.exp_data_dir,
            result_df=result_df,
            bn_generations=generations,
            logger=self.logger,
        )
        
        return result_df
    
    def _run_single_refinement(
        self,
        refiner,
        observation,
        init_generation,
        desc_variables,
        dag_variables,
        dag,
    ) -> Dict[str, Any]:
        """Run a single refinement experiment."""
        import time
        # Import already available from module level
        
        starttime = time.time()
        max_retries = 3
        
        for try_count in range(max_retries):
            try:
                refiner_result = refiner.run(
                    desc_variables=desc_variables,
                    dag_variables=dag_variables,
                    dag=dag,
                    observation=observation,
                    init_generation=init_generation,
                )
                break
            except InvalidDAGError:
                if try_count >= max_retries - 1:
                    self.logger.exception("Invalid DAG: Encountered max retries. Stop the experiment.")
                    raise
            except MaxRetriesError:
                if try_count >= max_retries - 1:
                    self.logger.exception("Max retries: Forced stop the experiment.")
                    raise

        endtime = time.time()

        _, _, f1_score, accuracy, nhd, shd = evaluate_generation(
            dag_input=dag,
            pred_input=refiner_result['Matrix'],
            logger=self.logger,
        )
        self.logger.info("nhd: %.4f; shd: %.4f; f1_score: %.4f; accuracy: %.4f", nhd, shd, f1_score, accuracy)

        return {
            "try_count": try_count + 1,
            "init_nhd": refiner_result.get("init_nhd"),
            "init_shd": refiner_result.get("init_shd"),
            "final_nhd": refiner_result.get("final_nhd"),
            "final_shd": refiner_result.get("final_shd"),
            "final_score": refiner_result.get("Score"),
            "move_count": refiner_result.get("MoveCount"),
            "latency": f"{int(endtime-starttime)}",
            "Matrix": refiner_result.get("Matrix"),
            "Graph": refiner_result.get("Graph"),
            "ActionHistory": refiner_result.get("ActionHistory"),
            "MetricsHistory": refiner_result.get("MetricsHistory")
        }
    
    def execute(self, dataset, worker=None) -> WorkflowResult:
        """
        Execute the configured workflow for a specific dataset, using the provided generator/refiner/tuple as needed.
        """
        if dataset is None:
            raise ValueError("A dataset must be provided to execute the workflow.")
        if self.workflow_type == "generation":
            return self.execute_generation_workflow(dataset, worker)
        elif self.workflow_type == "refinement":
            return self.execute_refinement_workflow(dataset, worker)
        elif self.workflow_type == "pipeline":
            generator, refiner = worker
            return self.execute_pipeline_workflow(dataset, generator, refiner)
        else:
            raise ValueError(f"Unknown workflow type: {self.workflow_type}")

    def analyze_results(self):
        from utils.analyze_utils import analyze_generation, analyze_refinement
        """Run the appropriate analysis for the workflow type."""
        exp_root = self.config['data']['experiment_data']['experiment_name']
        results_dir = os.path.join(
            self.config['data']['experiment_data']['root'],
            exp_root,
            self.config['data']['experiment_data']['results']
        )
        statistics_dir = os.path.join(
            self.config['data']['experiment_data']['root'],
            exp_root,
            self.config['data']['experiment_data']['statistics']
        )
        out_path = os.path.join(statistics_dir, f"{exp_root}_statistics.csv")
        datasets = self.config['data']['input_data']['dataset']
        if not isinstance(datasets, list):
            datasets = [datasets]
        if self.workflow_type == 'generation':
            for dataset in datasets:
                self.logger.info(f"Running generation analysis for dataset: {dataset}...")
                analyze_generation(results_dir, statistics_dir, out_path)
        elif self.workflow_type == 'refinement':
            for dataset in datasets:
                self.logger.info(f"Running refinement analysis for dataset: {dataset}...")
                analyze_refinement(results_dir, statistics_dir, out_path)
        elif self.workflow_type == 'pipeline':
            for dataset in datasets:
                self.logger.info(f"Pipeline workflow: running both generation and refinement analysis for dataset: {dataset}...")
                analyze_generation(results_dir, statistics_dir, out_path.replace('_statistics.csv', f'_generation_statistics_{dataset}.csv'))
                analyze_refinement(results_dir, statistics_dir, out_path.replace('_statistics.csv', f'_refinement_statistics_{dataset}.csv'))
        else:
            self.logger.warning(f"Unknown workflow type for analysis: {self.workflow_type}")
            return None

    def run_full_experiment(self, max_workers=4):
        datasets = self.config['data']['input_data']['dataset']
        if not isinstance(datasets, list):
            datasets = [datasets]
        max_workers = min(max_workers, len(datasets))
        # Load generator/refiner once
        if self.workflow_type == 'generation':
            worker = self._load_generator()
        elif self.workflow_type == 'refinement':
            worker = self._load_refiner()
        elif self.workflow_type == 'pipeline':
            worker = (self._load_generator(), self._load_refiner())
        else:
            raise ValueError(f"Unknown workflow type: {self.workflow_type}")
        def run_for_dataset(dataset):
            return self.execute(dataset, worker)
        if len(datasets) == 1:
            run_for_dataset(datasets[0])
            self.logger.info("Completed workflow for dataset: %s", datasets[0])
        else:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                futures = [
                    executor.submit(run_for_dataset, dataset)
                    for dataset in datasets
                ]
                for future in concurrent.futures.as_completed(futures):
                    future.result()  # Will raise exceptions if any
        self.analyze_results()
