#!/usr/bin/env python3
"""
MST++ Data Preparation - extracts ZIPs, copies files, verifies dataset.

Data download guide:
  百度网盘 (推荐, 提取码 mst1):
    训练光谱: https://pan.baidu.com/s/1NisQ6NjGvVhc0iOLH7OFvg
    训练RGB:  https://pan.baidu.com/s/1k7aSSL5MMipWYszlFaBLkA
    验证光谱: https://pan.baidu.com/s/1CIb5AqLWJxaGilTPtmWl0A
    验证RGB:  https://pan.baidu.com/s/1YakbXgBgnhNmYoxySmZaGw
    测试RGB:  https://pan.baidu.com/s/1RXHK64mUfK_GeeoLzqAmeQ
  Zenodo: https://doi.org/10.5281/zenodo.7839604 (训练光谱 21.4GB)

Usage: python prep_data.py --source <downloaded_data_dir>
"""

import sys, zipfile, shutil, argparse
from pathlib import Path

ROOT = Path(__file__).parent
DS = ROOT / 'dataset'

EXPECTED = {
    'Train_Spec': {f'ARAD_1K_{i:04d}.mat' for i in range(1, 951)},
    'Train_RGB':  {f'ARAD_1K_{i:04d}.jpg' for i in range(1, 951)},
    'Test_RGB':   {f'ARAD_1K_{i:04d}.jpg' for i in range(951, 1001)},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()

    src = Path(args.source)
    for d in EXPECTED:
        (DS / d).mkdir(parents=True, exist_ok=True)

    if not args.verify_only:
        for zp in list(src.glob('*.zip')) + list(src.glob('*.ZIP')):
            n = zp.name.lower()
            tgt = 'Test_RGB' if 'test' in n else 'Train_Spec' if 'spec' in n else 'Train_RGB' if 'rgb' in n else None
            if tgt:
                print(f'Extracting {zp.name} -> {tgt}')
                with zipfile.ZipFile(zp) as zf:
                    zf.extractall(DS / tgt)

        # Copy already-extracted files
        for cand_name, tgt_name in [('Train_Spec','Train_Spec'), ('Train_RGB','Train_RGB'),
                                      ('Train_spectral','Train_Spec'), ('Test_RGB','Test_RGB')]:
            cd = src / cand_name
            if cd.is_dir():
                for f in cd.glob('*.mat') + cd.glob('*.jpg'):
                    shutil.copy2(f, DS / tgt_name / f.name)

    # Verify
    total, found = 0, 0
    for dname, expected in EXPECTED.items():
        dp = DS / dname
        actual = {f.name for f in dp.glob('*') if f.suffix in ('.mat','.jpg')} if dp.exists() else set()
        cnt = len(actual)
        exp = len(expected)
        total += exp; found += cnt
        ok = 'OK' if cnt >= exp else f'MISS {exp - cnt}'
        print(f'  {dname:12s} {cnt:4d}/{exp:<4d} [{ok}]')

    if found >= total:
        print(f'\nData COMPLETE ({found}/{total}). Ready!')
    else:
        print(f'\nINCOMPLETE: {found}/{total}. Download from Baidu Netdisk (code: mst1).')

if __name__ == '__main__':
    main()
