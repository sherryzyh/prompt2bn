# BNSynth: Bayesian Network Synthesis Framework

A toolkit for fast, flexible Bayesian network structure generation and refinement using both LLM-based and traditional statistical methods.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- R (for SHD calculation)
- LLM API key

*Supported LLM Providers:*
- OpenAI (GPT models)
- Gemini
- DeepSeek

> You can use any of these providers—just set the appropriate API key in your `.env` file. Only one is required.

### Installation
```bash
# Clone and enter the repo
git clone https://github.com/yourusername/bnsynth.git
cd bnsynth

# (Recommended) Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
./install.sh
```

### Configure API Key
```bash
# Set your API key for your preferred LLM provider (OpenAI, Gemini, or DeepSeek)
echo "OPENAI_API_KEY=your-actual-key-here" > .env
# or
# echo "GEMINI_API_KEY=your-gemini-key-here" > .env
# or
# echo "DEEPSEEK_API_KEY=your-deepseek-key-here" > .env
```

### Install R & Required Packages
```bash
# macOS:   brew install r
# Ubuntu:  sudo apt-get install r-base
# Windows: Download from r-project.org

# In R console:
if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager")
BiocManager::install(c("graph","RBGL","pcalg"))
```

### Run Your First Workflow

You can get started immediately using the **sample data** and **sample configuration files** provided in this repository.

1. **Use the sample data:**
   - A sample dataset is already included under `data/BNlearnR/` (e.g., the "asia" dataset).
2. **Use the sample configuration:**
   - Sample config files are available in the `configs/` directory (e.g., `sample_generation_only.yaml`, `sample_refinement_only.yaml`).
3. **Run a workflow:**
   - For generation:
     ```bash
     python run.py --config configs/sample_generation_only.yaml
     ```
   - For refinement: (after you run the generation workflow)
     ```bash
     python run.py --config configs/sample_refinement_only.yaml
     ```
4. **Check results:**
   - Outputs will be saved in the `experiments/` directory as specified in your config.

That's it! You can now explore results or try customizing your workflow further.

---

## 🏗️ Project Structure
```
bnsynth/
├── run.py                 # Main entry point
├── workflow_controller.py # Workflow orchestration
├── llm.py                 # LLM client wrapper
├── generators/            # Structure generators
├── refiners/              # Structure refiners
├── utils/                 # Utility functions
├── configs/               # [Sample]YAML experiment configs
├── data/                  # [Sample]Input datasets
└── README.md
```

---

## 📂 Input Data Structure

Place your datasets under `data/BNlearnR/{dataset_name}/` (or `data/BnRep/{dataset_name}/`) based on your data source. Example for the "asia" dataset (BNlearnR):

```
data/
└── BNlearnR/
    └── asia/
        ├── asia_dag.csv           # True DAG structure
        ├── asia_bnlearn.csv       # Variable metadata
        └── samples/
            ├── asia_5000_1.csv   # Observation samples
            ├── asia_5000_2.csv
            └── ...
```

- **DAG CSV**: Adjacency matrix or edge list
- **Metadata CSV**: Variable names/descriptions
- **Samples**: Observational data (one or more files)

---

## ⚡ Basic Usage

### Supported Workflows
- **Generation only**: Create BN structures from scratch
- **Refinement only**: Refine existing BN structures
- **Unified pipeline (generation + refinement)**: *Coming soon!*

> For now, you can run generation and then run refinement separately for the same dataset to achieve a full pipeline.

### Run Generation Workflow
```bash
python run.py --config configs/sample_generation_only.yaml
```

### Run Refinement Workflow
```bash
python run.py --config configs/sample_refinement_only.yaml
```

### Use Your Own Data
1. Place your files as shown above
2. Update the config YAML with your dataset name and paths
3. Run as above

---

## ⚙️ Minimal Configuration Example
```yaml
workflow: "generation"  # or "refinement"
data:
  input_data:
    root: "data"
    dataset: "asia"
    source: "bnlearn"
  experiment_data:
    root: "experiments"
    experiment_name: "my_experiment"
    generation: "generations"
    logs: "logs"
    results: "results"
    statistics: "statistics"
observation: 100

generation:
  repeated_run: 3
  generator: "promptbn"
  model: "o3-mini"

# or "refinement"
# refinement:
#   data:
#     init_generator: "promptbn"
#     init_model: "o3-mini"
#     source_experiment: "previous_gen_experiment"
#   refiner: "reactbn"
#   model: "o3-mini"

```

---

## 🐛 Troubleshooting
- **R not found**: Install R and required packages as above
- **API key errors**: Ensure `.env` exists and contains your key (OpenAI, Gemini, or DeepSeek)
- **Data loading errors**: Check your data directory and file names
- **Import errors**: Activate your virtual environment

---

## 📄 License & Citation

This project is licensed under the MIT License.

If you use BNSynth in your research, please cite:
```bibtex
@software{bnsynth,
  title={BNSynth: Bayesian Network Synthesis Framework},
  author={Your Name},
  year={2025},
  url={https://github.com/sherryzyh/bnsynth}
}
```

---

For more details, advanced configuration, and troubleshooting, see the full documentation or open an issue on GitHub.

---

## 📬 Contact

For questions, support, or feedback, please contact: yinghuan.flash@gmail.com
