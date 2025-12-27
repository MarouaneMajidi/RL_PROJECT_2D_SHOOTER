# DDA Agents Evaluation and Analysis

This folder contains all the code and results for evaluating and comparing DDA (Dynamic Difficulty Adjustment) agents.

## Contents

- **`evaluate_dda_agents.py`** - Script to run evaluation on all DDA agents (100 games per agent)
- **`analyze_dda_evaluation.ipynb`** - Jupyter notebook for analyzing results and finding the best agent
- **`data/evaluation/`** - Contains the evaluation results:
  - `dda_evaluation_agents.csv` - Game-level metrics for each agent
  - `dda_evaluation_intervals.csv` - Interval-level metrics (5-second intervals)

## Usage

### 1. Run Evaluation (if not already done)

```bash
python evaluate_dda_agents.py
```

This will:
- Load the player agent from `../checkpoints/player_ppo/best_model_marouane.pth`
- Find all DDA models in `../checkpoints/dda/` and `../checkpoints_cnn/dda_cnn/`
- Run 100 games per DDA agent
- Save results to `data/evaluation/`

### 2. Analyze Results

Open and run the Jupyter notebook:

```bash
jupyter notebook analyze_dda_evaluation.ipynb
```

The notebook will:
- Load the evaluation results
- Generate comparison visualizations
- Calculate composite scores
- Identify the best performing DDA agent
- Provide detailed statistics and recommendations

## Output Files

After running the notebook, you'll get:
- `dda_survival_time_comparison.png` - Survival time comparisons
- `dda_flow_zone_comparison.png` - Flow zone metrics
- `dda_performance_comparison.png` - Performance metrics
- `dda_composite_scores.png` - Composite score comparison
- `dda_action_distribution.png` - Action distribution analysis
- `dda_correlation_matrix.png` - Correlation heatmap

## Notes

- The evaluation script automatically detects both RL-based and CNN-based DDA agents
- Results are saved in CSV format for easy analysis
- The notebook provides comprehensive statistical analysis and visualizations

