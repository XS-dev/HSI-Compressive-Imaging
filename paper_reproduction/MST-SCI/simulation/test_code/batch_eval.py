"""Batch evaluate all MST-SCI methods and compute PSNR/SSIM."""
import subprocess, sys, os, torch
import numpy as np
import scipy.io as scio
from pathlib import Path

ROOT = Path('D:/MODXS/CODING/LLM_Learning/Papers/paper_reproduction/MST-SCI/simulation/test_code')
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT))

# Methods and their model paths
METHODS = {
    'mst_s':       ('mst', 'mst/mst_s.pth'),
    'mst_m':       ('mst', 'mst/mst_m.pth'),
    'mst_l':       ('mst', 'mst/mst_l.pth'),
    'cst_s':       ('cst', 'cst/cst_s.pth'),
    'cst_m':       ('cst', 'cst/cst_m.pth'),
    'cst_l':       ('cst', 'cst/cst_l.pth'),
    'cst_l_plus':  ('cst', 'cst/cst_l_plus.pth'),
    'dauhst_2stg': ('dauhst', 'dauhst/dauhst_2stg.pth'),
    'dauhst_3stg': ('dauhst', 'dauhst/dauhst_3stg.pth'),
    'dauhst_5stg': ('dauhst', 'dauhst/dauhst_5stg.pth'),
    'dauhst_9stg': ('dauhst', 'dauhst/dauhst_9stg.pth'),
    'hdnet':       ('hdnet', 'hdnet/hdnet.pth'),
    'dgsmp':       ('dgsmp', 'dgsmp/dgsmp.pth'),
    'gap_net':     ('gap_net', 'gap_net/gap_net.pth'),
    'admm_net':    ('admm_net', 'admm_net/admm_net.pth'),
    'tsa_net':     ('tsa_net', 'tsa_net/tsa_net.pth'),
    'birnat':      ('birnat', 'birnat/birnat.pth'),
    'mst_plus_plus': ('mst_plus_plus', 'mst_plus_plus/mst_plus_plus.pth'),
}

def psnr_torch(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def compute_metrics(pred_path, truth_path):
    pred = scio.loadmat(pred_path)['pred']
    truth = scio.loadmat(truth_path)['truth']
    pred_t = torch.from_numpy(pred).float()
    truth_t = torch.from_numpy(truth).float()
    psnrs, ssims = [], []
    for i in range(pred_t.shape[0]):
        p = pred_t[i]
        t = truth_t[i]
        psnrs.append(psnr_torch(p, t).item())
        # Simple SSIM approximation (use mean for now)
        ssims.append(1.0 - torch.mean((p - t)**2).item() / torch.mean(t**2).item())
    return np.mean(psnrs), np.mean(ssims)

results = {}
for method, (template, model_path) in METHODS.items():
    print(f'\n--- {method} ---')
    model_file = ROOT / 'model_zoo' / model_path
    if not model_file.exists():
        print(f'  Model not found: {model_file}')
        continue
    out_dir = ROOT / 'exp_batch' / method
    out_dir.mkdir(parents=True, exist_ok=True)
    result_file = out_dir / 'Test_result.mat'
    
    if not result_file.exists():
        cmd = [
            sys.executable, 'test.py',
            '--template', template,
            '--outf', str(out_dir) + '/',
            '--method', method,
            '--pretrained_model_path', str(model_file),
            '--gpu_id', '0',
        ]
        ret = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        last_line = ret.stdout.strip().split('\n')[-1] if ret.stdout else ''
        if ret.returncode != 0:
            print(f'  ERROR: {ret.stderr[-300:]}')
            results[method] = (0, 0)
            continue
    
    if result_file.exists():
        try:
            psnr_v, ssim_v = compute_metrics(str(result_file), str(result_file))
            results[method] = (psnr_v, ssim_v)
            print(f'  PSNR: {psnr_v:.2f}, SSIM: {ssim_v:.4f}')
        except Exception as e:
            print(f'  Metric error: {e}')
            results[method] = (0, 0)

print('\n' + '='*60)
print(f'{"Method":<20s} {"PSNR":>8s}  {"SSIM":>8s}')
print('-'*40)
# Sort by PSNR descending
for method in sorted(results, key=lambda x: results[x][0], reverse=True):
    psnr_v, ssim_v = results[method]
    if psnr_v > 0:
        print(f'{method:<20s} {psnr_v:8.2f}  {ssim_v:8.4f}')

# Paper reference values (from MST-SCI README)
paper = {
    'dauhst_9stg': 38.36, 'dauhst_5stg': 37.75, 'dauhst_3stg': 37.21, 'dauhst_2stg': 36.34,
    'birnat': 37.58, 'cst_l_plus': 36.12, 'mst_plus_plus': 35.99, 'cst_l': 35.85,
    'mst_l': 35.18, 'cst_m': 35.31, 'mst_m': 34.94, 'hdnet': 34.97,
    'cst_s': 34.71, 'mst_s': 34.26, 'gap_net': 33.26, 'admm_net': 33.58,
    'dgsmp': 32.63, 'tsa_net': 31.46,
}
print(f'\n{"Method":<20s} {"Ours":>8s}  {"Paper":>8s}')
for method in sorted(results, key=lambda x: results[x][0], reverse=True):
    if method in paper and results[method][0] > 0:
        print(f'{method:<20s} {results[method][0]:8.2f}  {paper[method]:8.2f}')
