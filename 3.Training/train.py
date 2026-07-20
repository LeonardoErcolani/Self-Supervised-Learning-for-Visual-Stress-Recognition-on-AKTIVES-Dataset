#!/usr/bin/env python3
"""
TRAINING SCRIPT - RUNS ALL PAPER EXPERIMENTS

This script runs the stress-classification experiments on the AKTIVES dataset.
The full list of experiments is defined in `experiments_config.json` (same folder)
and executed sequentially, in the order they appear in the paper.

HOW TO USE:
1. Run all experiments:            python train.py
2. Add / remove / reorder runs:    edit experiments_config.json
3. Results are written to:         Results/

EXPERIMENT TYPES:
- "single":   one feature set (DINOv2 / Face DINOv2 / Upper Body DINOv2 / Emotions / Expression_Pose)
- "combined": DINOv2 + a second feature set (Emotions or Expression_Pose)

TEMPORAL STRATEGIES (selected via the matching CSV):
- "Baseline 10 seconds":  single frame at the 10th second   (*_at_sec_10.csv)
- "Average per interval": mean of the 10 frames             (*_per_interval.csv)
- "Triple intervals":     early / middle / late sub-windows (*_triple_intervals.csv)
- "Overlapping intervals":three overlapping 5s windows      (*_overlapping_intervals.csv)

PCA:
- true:  reduce features to 10 components
- false: use the original feature dimensions

NOTE: All experiments use participant-based splitting to prevent data leakage.
      Run this script from the repository root so the "Processed Data/" paths resolve.
"""

import json
import os
from unified_experiment_runner import run_experiment

# ============================================================================
# EXPERIMENT CONFIGURATIONS - loaded from experiments_config.json
# ============================================================================
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments_config.json")
with open(config_path, "r") as f:
    EXPERIMENTS = json.load(f)

# ============================================================================
# AVAILABLE OPTIONS (for editing experiments_config.json)
# ============================================================================
"""
FEATURE TYPES:
- "DINOv2"           - Full-frame DINOv2 features (feat_*)
- "Face DINOv2"      - Face-only crops (face_feat_*)
- "Upper Body DINOv2"- Upper-body crops (upperbody_feat_*)
- "Emotions"         - FER emotion probabilities (prob_*)
- "Expression_Pose"  - Facial and pose landmark features

EXPERIMENT TYPES:
- "single"           - Use one feature type
- "combined"         - Combine DINOv2 + another feature type

TEMPORAL STRATEGIES:
- "Baseline 10 seconds"   - Single frame at 10th second
- "Average per interval"  - Average features per interval
- "Triple intervals"      - Early / middle / late sub-windows
- "Overlapping intervals" - 5-second overlapping windows

PCA:
- true               - Apply PCA dimensionality reduction (10 components)
- false              - Use original features
"""

if __name__ == "__main__":
    print("🚀 Starting AKTIVES stress-recognition experiments...")
    print(f"📋 Total experiments to run: {len(EXPERIMENTS)}")
    print("\n" + "=" * 60)

    all_results = []
    successful_experiments = 0
    failed_experiments = 0

    for i, exp_config in enumerate(EXPERIMENTS, 1):
        print(f"\n🔬 EXPERIMENT {i}/{len(EXPERIMENTS)}")
        print(f"📋 Name: {exp_config['name']}")
        print(f"🔧 Type: {exp_config['type']}")

        if exp_config['type'] == 'single':
            print(f"📊 Feature Type: {exp_config['feature_type']}")
            print(f"⏱️  Temporal Strategy: {exp_config['temporal_strategy']}")
            print(f"📉 PCA: {exp_config['use_pca']}")
        else:
            print(f"🔗 Combining: DINOv2 + {exp_config['second_feature_type']}")
            print(f"⏱️  Temporal Strategy: {exp_config['temporal_strategy']}")
            print(f"📉 PCA: {exp_config['use_pca']}")

        print("-" * 40)

        try:
            results = run_experiment(exp_config)

            if results:
                print(f"✅ Experiment {i} completed successfully!")
                print(f"📊 Trained {len(results)} models")
                print(f"🏆 Best F1-Score: {max([r['F1-Score'] for r in results]):.4f}")

                for result in results:
                    result['Experiment_Number'] = i
                    result['Total_Experiments'] = len(EXPERIMENTS)
                    all_results.append(result)

                successful_experiments += 1
            else:
                print(f"❌ Experiment {i} failed - no models completed")
                failed_experiments += 1

        except Exception as e:
            print(f"💥 Error in experiment {i}: {e}")
            failed_experiments += 1
            continue

    # Final summary
    print("\n" + "=" * 60)
    print("🎯 FINAL SUMMARY")
    print("=" * 60)
    print(f"✅ Successful experiments: {successful_experiments}")
    print(f"❌ Failed experiments: {failed_experiments}")
    print(f"📊 Total experiments: {len(EXPERIMENTS)}")

    if all_results:
        print(f"🤖 Total models trained: {len(all_results)}")
        best_result = max(all_results, key=lambda x: x['F1-Score'])
        print(f"🏆 Overall best F1-Score: {best_result['F1-Score']:.4f}")
        print(f"🥇 Best experiment: {best_result['Experiment']} ({best_result['Model']})")
        print(f"📁 Results saved to: Results/")
    else:
        print("💥 No experiments completed successfully")
        print("\n🔍 Troubleshooting tips:")
        print("1. Run from the repository root so 'Processed Data/' resolves")
        print("2. Check that the CSV file paths in experiments_config.json exist")
        print("3. Verify the feature_type matches your data")
