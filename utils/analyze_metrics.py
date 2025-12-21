"""
Analyze collected DDA evaluation metrics.

Reads all collected metrics and generates summary statistics and comparisons.
"""

import os
import json
import csv
import glob
from typing import Dict, List
import statistics


def load_player_data(player_id: str, data_dir: str = "data/evaluation") -> Dict:
    """Load all data for a specific player."""
    player_data = {
        'player_id': player_id,
        'match1': {},
        'match2': {},
        'answers': {},
        'preference': None,
    }
    
    # Load match 1 metrics
    match1_files = glob.glob(os.path.join(data_dir, f"{player_id}_match1_*_metrics.json"))
    if match1_files:
        with open(match1_files[0], 'r') as f:
            player_data['match1']['metrics'] = json.load(f)
    
    # Load match 2 metrics
    match2_files = glob.glob(os.path.join(data_dir, f"{player_id}_match2_*_metrics.json"))
    if match2_files:
        with open(match2_files[0], 'r') as f:
            player_data['match2']['metrics'] = json.load(f)
    
    # Load answers
    match1_answers = glob.glob(os.path.join(data_dir, f"{player_id}_match1_answers.json"))
    if match1_answers:
        with open(match1_answers[0], 'r') as f:
            player_data['answers']['match1'] = json.load(f)
    
    match2_answers = glob.glob(os.path.join(data_dir, f"{player_id}_match2_answers.json"))
    if match2_answers:
        with open(match2_answers[0], 'r') as f:
            player_data['answers']['match2'] = json.load(f)
    
    # Load preference
    preference_files = glob.glob(os.path.join(data_dir, f"{player_id}_preference.json"))
    if preference_files:
        with open(preference_files[0], 'r') as f:
            player_data['preference'] = json.load(f)
    
    return player_data


def get_all_players(data_dir: str = "data/evaluation") -> List[str]:
    """Get list of all player IDs from collected data."""
    player_ids = set()
    
    # Find all player IDs from metric files
    metric_files = glob.glob(os.path.join(data_dir, "*_match*_metrics.json"))
    for file in metric_files:
        filename = os.path.basename(file)
        parts = filename.split('_')
        if len(parts) >= 2:
            player_id = parts[0]
            player_ids.add(player_id)
    
    return sorted(list(player_ids))


def calculate_comparison_stats(all_players_data: List[Dict]) -> Dict:
    """Calculate comparison statistics between with_dda and without_dda."""
    stats = {}
    
    # Separate data by condition
    with_dda_metrics = {}
    without_dda_metrics = {}
    
    for player_data in all_players_data:
        match1_condition = player_data['answers'].get('match1', {}).get('condition')
        match2_condition = player_data['answers'].get('match2', {}).get('condition')
        
        # Match 1
        if match1_condition == "with_dda" and 'match1' in player_data and 'metrics' in player_data['match1']:
            for key, value in player_data['match1']['metrics'].items():
                if key not in with_dda_metrics:
                    with_dda_metrics[key] = []
                if isinstance(value, (int, float)):
                    with_dda_metrics[key].append(value)
        elif match1_condition == "without_dda" and 'match1' in player_data and 'metrics' in player_data['match1']:
            for key, value in player_data['match1']['metrics'].items():
                if key not in without_dda_metrics:
                    without_dda_metrics[key] = []
                if isinstance(value, (int, float)):
                    without_dda_metrics[key].append(value)
        
        # Match 2
        if match2_condition == "with_dda" and 'match2' in player_data and 'metrics' in player_data['match2']:
            for key, value in player_data['match2']['metrics'].items():
                if key not in with_dda_metrics:
                    with_dda_metrics[key] = []
                if isinstance(value, (int, float)):
                    with_dda_metrics[key].append(value)
        elif match2_condition == "without_dda" and 'match2' in player_data and 'metrics' in player_data['match2']:
            for key, value in player_data['match2']['metrics'].items():
                if key not in without_dda_metrics:
                    without_dda_metrics[key] = []
                if isinstance(value, (int, float)):
                    without_dda_metrics[key].append(value)
    
    # Calculate statistics
    for metric_name in set(list(with_dda_metrics.keys()) + list(without_dda_metrics.keys())):
        with_dda_values = with_dda_metrics.get(metric_name, [])
        without_dda_values = without_dda_metrics.get(metric_name, [])
        
        if with_dda_values and without_dda_values:
            stats[metric_name] = {
                'with_dda': {
                    'mean': statistics.mean(with_dda_values),
                    'median': statistics.median(with_dda_values),
                    'stdev': statistics.stdev(with_dda_values) if len(with_dda_values) > 1 else 0,
                    'count': len(with_dda_values),
                },
                'without_dda': {
                    'mean': statistics.mean(without_dda_values),
                    'median': statistics.median(without_dda_values),
                    'stdev': statistics.stdev(without_dda_values) if len(without_dda_values) > 1 else 0,
                    'count': len(without_dda_values),
                },
            }
            
            # Calculate improvement
            mean_with = stats[metric_name]['with_dda']['mean']
            mean_without = stats[metric_name]['without_dda']['mean']
            if mean_without > 0:
                improvement = ((mean_with - mean_without) / mean_without) * 100
                stats[metric_name]['improvement_percent'] = improvement
            else:
                stats[metric_name]['improvement_percent'] = 0
    
    return stats


def analyze_subjective_answers(all_players_data: List[Dict]) -> Dict:
    """Analyze subjective answers (enjoyment, difficulty, preference)."""
    analysis = {
        'enjoyment': {'with_dda': [], 'without_dda': []},
        'difficulty': {'with_dda': [], 'without_dda': []},
        'preference': {'with_dda': 0, 'without_dda': 0, 'no_preference': 0},
    }
    
    for player_data in all_players_data:
        # Match 1
        match1_condition = player_data['answers'].get('match1', {}).get('condition')
        if match1_condition:
            enjoyment1 = player_data['answers'].get('match1', {}).get('enjoyment')
            difficulty1 = player_data['answers'].get('match1', {}).get('difficulty')
            if enjoyment1:
                analysis['enjoyment'][match1_condition].append(enjoyment1)
            if difficulty1:
                analysis['difficulty'][match1_condition].append(difficulty1)
        
        # Match 2
        match2_condition = player_data['answers'].get('match2', {}).get('condition')
        if match2_condition:
            enjoyment2 = player_data['answers'].get('match2', {}).get('enjoyment')
            difficulty2 = player_data['answers'].get('match2', {}).get('difficulty')
            if enjoyment2:
                analysis['enjoyment'][match2_condition].append(enjoyment2)
            if difficulty2:
                analysis['difficulty'][match2_condition].append(difficulty2)
        
        # Preference
        preference = player_data.get('preference', {}).get('preferred_condition')
        if preference:
            if preference in analysis['preference']:
                analysis['preference'][preference] += 1
    
    # Calculate means
    for condition in ['with_dda', 'without_dda']:
        if analysis['enjoyment'][condition]:
            analysis['enjoyment'][f'{condition}_mean'] = statistics.mean(analysis['enjoyment'][condition])
        if analysis['difficulty'][condition]:
            analysis['difficulty'][f'{condition}_mean'] = statistics.mean(analysis['difficulty'][condition])
    
    return analysis


def generate_report(data_dir: str = "data/evaluation", output_file: str = "evaluation_report.txt"):
    """Generate a comprehensive evaluation report."""
    print("Generating evaluation report...")
    
    # Get all players
    player_ids = get_all_players(data_dir)
    print(f"Found {len(player_ids)} players")
    
    if len(player_ids) == 0:
        print("No player data found!")
        return
    
    # Load all player data
    all_players_data = [load_player_data(pid, data_dir) for pid in player_ids]
    
    # Calculate statistics
    comparison_stats = calculate_comparison_stats(all_players_data)
    subjective_analysis = analyze_subjective_answers(all_players_data)
    
    # Generate report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("DDA EVALUATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"\nTotal Players: {len(player_ids)}")
    report_lines.append(f"Players: {', '.join(player_ids)}")
    report_lines.append("\n" + "=" * 80)
    
    # Key Metrics Comparison
    report_lines.append("\nKEY METRICS COMPARISON")
    report_lines.append("-" * 80)
    
    key_metrics = [
        'survival_time',
        'time_in_flow_zone_percent',
        'average_health',
        'time_in_danger_percent',
        'total_kills',
        'health_variance',
    ]
    
    for metric in key_metrics:
        if metric in comparison_stats:
            stat = comparison_stats[metric]
            report_lines.append(f"\n{metric.upper().replace('_', ' ')}:")
            report_lines.append(f"  With DDA:    {stat['with_dda']['mean']:.2f} (n={stat['with_dda']['count']})")
            report_lines.append(f"  Without DDA: {stat['without_dda']['mean']:.2f} (n={stat['without_dda']['count']})")
            if 'improvement_percent' in stat:
                improvement = stat['improvement_percent']
                sign = "+" if improvement > 0 else ""
                report_lines.append(f"  Improvement: {sign}{improvement:.1f}%")
    
    # Subjective Analysis
    report_lines.append("\n" + "=" * 80)
    report_lines.append("SUBJECTIVE ANALYSIS")
    report_lines.append("-" * 80)
    
    # Enjoyment
    if subjective_analysis['enjoyment']['with_dda'] and subjective_analysis['enjoyment']['without_dda']:
        report_lines.append("\nENJOYMENT (1-5 scale):")
        report_lines.append(f"  With DDA:    {subjective_analysis['enjoyment']['with_dda_mean']:.2f}")
        report_lines.append(f"  Without DDA: {subjective_analysis['enjoyment']['without_dda_mean']:.2f}")
        diff = subjective_analysis['enjoyment']['with_dda_mean'] - subjective_analysis['enjoyment']['without_dda_mean']
        sign = "+" if diff > 0 else ""
        report_lines.append(f"  Difference:  {sign}{diff:.2f}")
    
    # Difficulty
    if subjective_analysis['difficulty']['with_dda'] and subjective_analysis['difficulty']['without_dda']:
        report_lines.append("\nDIFFICULTY (1-5 scale, 3=balanced):")
        report_lines.append(f"  With DDA:    {subjective_analysis['difficulty']['with_dda_mean']:.2f}")
        report_lines.append(f"  Without DDA: {subjective_analysis['difficulty']['without_dda_mean']:.2f}")
        diff = subjective_analysis['difficulty']['with_dda_mean'] - subjective_analysis['difficulty']['without_dda_mean']
        sign = "+" if diff > 0 else ""
        report_lines.append(f"  Difference:  {sign}{diff:.2f}")
    
    # Preference
    report_lines.append("\nPREFERENCE:")
    total_prefs = sum(subjective_analysis['preference'].values())
    if total_prefs > 0:
        for condition, count in subjective_analysis['preference'].items():
            percent = (count / total_prefs) * 100
            report_lines.append(f"  {condition.replace('_', ' ').title()}: {count} ({percent:.1f}%)")
    
    # Write report
    report_text = "\n".join(report_lines)
    with open(output_file, 'w') as f:
        f.write(report_text)
    
    print(f"\nReport saved to {output_file}")
    print("\n" + report_text)
    
    # Also save as JSON for programmatic access
    json_output = {
        'players': player_ids,
        'comparison_stats': comparison_stats,
        'subjective_analysis': subjective_analysis,
    }
    json_file = output_file.replace('.txt', '.json')
    with open(json_file, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"JSON report saved to {json_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze DDA evaluation metrics")
    parser.add_argument("--data-dir", default="data/evaluation", help="Directory containing evaluation data")
    parser.add_argument("--output", default="evaluation_report.txt", help="Output report file")
    
    args = parser.parse_args()
    
    generate_report(args.data_dir, args.output)

