#!/usr/bin/env python3
"""
MST++ Reproduction - One-click runner.
Usage:
  conda activate mstpp
  python run.py --mode eval       # Evaluate on validation set
  python run.py --mode train      # Train MST++ from scratch
  python run.py --mode predict --rgb_path <path>  # Predict single image
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()

REQUIRED_DATA = {
    'Train_Spec': 'dataset/Train_Spec/*.mat',
    'Train_RGB': 'dataset/Train_RGB/*.jpg',
}

def check_data():
    """Check if required dataset files exist."""
    missing = []
    for name, pattern in REQUIRED_DATA.items():
        files = list((PROJECT_ROOT / 'dataset' / name).glob('*.mat' if 'Spec' in name else '*.jpg'))
        if not files:
            missing.append(name)
    if missing:
        print(f"ERROR: Missing data: {', '.join(missing)}")
        print("\nPlease download the NTIRE 2022 ARAD_1K dataset and extract:")
        print("  Train_Spec/  -> dataset/Train_Spec/")
        print("  Train_RGB/   -> dataset/Train_RGB/")
        print("  Test_RGB/    -> dataset/Test_RGB/")
        print("\nRun 'python prep_data.py --data_dir <ZIP_DIR>' to auto-extract.\n")
        return False
    return True


def run_eval():
    print("=" * 50)
    print("MST++ Validation Set Evaluation")
    print("=" * 50)
    os.chdir(PROJECT_ROOT / 'test_develop_code')
    cmd = [
        sys.executable, 'test.py',
        '--data_root', str(PROJECT_ROOT / 'dataset/'),
        '--method', 'mst_plus_plus',
        '--pretrained_model_path', './model_zoo/mst_plus_plus.pth',
        '--outf', './exp/mst_plus_plus/',
        '--gpu_id', '0',
    ]
    subprocess.run(cmd, check=True)


def run_train():
    print("=" * 50)
    print("MST++ Training from Scratch (RTX 4090 optimized)")
    print("=" * 50)
    os.chdir(PROJECT_ROOT / 'train_code')
    cmd = [
        sys.executable, 'train.py',
        '--method', 'mst_plus_plus',
        '--batch_size', '32',        # RTX 4090 24GB optimized
        '--end_epoch', '300',
        '--init_lr', '4e-4',
        '--outf', './exp/mst_plus_plus/',
        '--data_root', str(PROJECT_ROOT / 'dataset/'),
        '--patch_size', '128',
        '--stride', '8',
        '--gpu_id', '0',
        '--amp',                     # Mixed precision for 2x speed
        '--grad_clip', '1.0',
        '--num_workers', '4',
    ]
    subprocess.run(cmd, check=True)


def run_predict(rgb_path):
    os.chdir(PROJECT_ROOT / 'predict_code')
    cmd = [
        sys.executable, 'test.py',
        '--rgb_path', rgb_path,
        '--method', 'mst_plus_plus',
        '--pretrained_model_path', './model_zoo/mst_plus_plus.pth',
        '--outf', './exp/mst_plus_plus/',
        '--gpu_id', '0',
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description='MST++ Reproduction Runner')
    parser.add_argument('--mode', choices=['eval', 'train', 'predict'], required=True,
                       help='eval: validation set evaluation | train: train from scratch | predict: single image')
    parser.add_argument('--rgb_path', type=str, default=None,
                       help='Path to RGB image for prediction mode')
    args = parser.parse_args()

    if args.mode in ('eval', 'train'):
        if not check_data():
            sys.exit(1)

    if args.mode == 'eval':
        run_eval()
    elif args.mode == 'train':
        run_train()
    elif args.mode == 'predict':
        if not args.rgb_path:
            args.rgb_path = str(PROJECT_ROOT / 'predict_code/demo/ARAD_1K_0912.jpg')
            print(f'No --rgb_path provided, using demo: {args.rgb_path}')
        run_predict(args.rgb_path)


if __name__ == '__main__':
    main()
