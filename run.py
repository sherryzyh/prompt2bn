#!/usr/bin/env python3
"""
BNSynth: Bayesian Network Synthesis Framework Runner

This script provides the main entry point for BNSynth workflows supporting:
- BN construction/generation (LLM-based and traditional methods)
- BN refinement based on initial graphs (LLM-enhanced and traditional methods)
- Data-dependent and data-free structural learning
- Pipeline workflows combining generation and refinement

Supported workflows:
- Generation only: Create BN structures using generators (data-free or data-dependent)
- Refinement only: Refine existing structures using refiners (data-dependent)
- Pipeline: Generate then refine in sequence

Usage:
    python run.py --config configs/my_config.yaml
    python run.py --config configs/generation_only.yaml
    python run.py --config configs/refinement_only.yaml
"""

import argparse
import logging
import os
import yaml
import datetime
from typing import Dict, Any

# Import will be resolved at runtime
from workflow_controller import WorkflowController
from utils.io_utils import get_experiment_path, init_experiment
# Removed: from analyze import analyze_experiment


def init_logging(config: dict) -> logging.Logger:
    """Initialize logging to a file and return the logger."""
    exp_data_dir = get_experiment_path(config)
    os.makedirs(exp_data_dir, exist_ok=True)
    log_file_path = os.path.join(
        exp_data_dir,
        "logs",
        f"{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}.log"
    )

    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_dir = os.path.dirname(log_file_path)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        filename=log_file_path,
        level=logging.DEBUG,
        encoding="utf-8",
        filemode="a",
        style="{",
        format="{asctime} - {levelname}:{filename}:{lineno}:{message}",
        datefmt="%Y-%m-%d %H:%M",
    )

    # Add console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "{asctime} - {levelname}:{filename}:{lineno}:{message}", 
        style="{", 
        datefmt="%Y-%m-%d %H:%M"
    )
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    return logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="BNSynth - Bayesian Network Synthesis Framework: Execute generation and/or refinement workflows"
    )
    
    parser.add_argument(
        "--config", "-c", 
        type=str, 
        required=True,
        help="Configuration file path (YAML format)"
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Show what would be executed without running"
    )
    
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def validate_config(config: Dict[str, Any]) -> None:
    """Validate the configuration structure."""
    required_sections = ['data']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate workflow configuration
    if 'workflow' not in config:
        raise ValueError("Missing required 'workflow' configuration section")
    
    workflow = config['workflow']
    valid_workflows = ['generation', 'refinement', 'pipeline']
    if workflow not in valid_workflows:
        raise ValueError(f"Invalid workflow '{workflow}'. Must be one of: {valid_workflows}")
    
    # Validate workflow-specific configurations
    if workflow in ['generation', 'pipeline']:
        if 'generation' not in config:
            raise ValueError(f"Missing 'generation' configuration section for {workflow} workflow")
    
    if workflow in ['refinement', 'pipeline']:
        if 'refinement' not in config:
            raise ValueError(f"Missing 'refinement' configuration section for {workflow} workflow")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError("Configuration file is empty or invalid")
    
    validate_config(config)
    return config


def main():
    """Main entry point for BNSynth framework."""
    args = parse_arguments()
    try:
        config = load_config(args.config)
        datasets = config['data']['input_data']['dataset']
        if not isinstance(datasets, list):
            datasets = [datasets]
        max_workers = min(4, len(datasets))  # Adjust as needed
        logger = init_logging(config)
        exp_data_dir = init_experiment(config, logger)
        
        controller = WorkflowController(
            config,
            logger,
            exp_data_dir,
        )
        controller.run_full_experiment(max_workers=max_workers)
        controller.analyze_results()
        print("[INFO] Batch analysis completed for all datasets.")
    except Exception as e:
        logging.error("Fatal error in BNSynth runner: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
