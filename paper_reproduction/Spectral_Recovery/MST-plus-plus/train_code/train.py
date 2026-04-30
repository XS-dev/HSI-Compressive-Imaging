import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import os
from hsi_dataset import TrainDataset, ValidDataset
from architecture import *
from utils import AverageMeter, initialize_logger, save_checkpoint, record_loss, \
    time2file_name, Loss_MRAE, Loss_RMSE, Loss_PSNR
import datetime

parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox (Enhanced)")
parser.add_argument('--method', type=str, default='mst_plus_plus')
parser.add_argument('--pretrained_model_path', type=str, default=None)
parser.add_argument("--batch_size", type=int, default=32, help="batch size (RTX 4090 default)")
parser.add_argument("--end_epoch", type=int, default=300, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=4e-4, help="initial learning rate")
parser.add_argument("--outf", type=str, default='./exp/mst_plus_plus/', help='path log files')
parser.add_argument("--data_root", type=str, default='../dataset/')
parser.add_argument("--patch_size", type=int, default=128, help="patch size")
parser.add_argument("--stride", type=int, default=8, help="stride")
parser.add_argument("--gpu_id", type=str, default='0', help='GPU ID')
parser.add_argument("--amp", action='store_true', default=True, help='use AMP mixed precision')
parser.add_argument("--no_amp", dest='amp', action='store_false', help='disable AMP')
parser.add_argument("--grad_accum", type=int, default=1, help='gradient accumulation steps')
parser.add_argument("--grad_clip", type=float, default=1.0, help='gradient clipping max norm')
parser.add_argument("--num_workers", type=int, default=4, help='DataLoader workers')
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

# load dataset
print("\nloading dataset ...")
train_data = TrainDataset(data_root=opt.data_root, crop_size=opt.patch_size, bgr2rgb=True, arg=True, stride=opt.stride)
print(f"Iteration per epoch: {len(train_data)}")
val_data = ValidDataset(data_root=opt.data_root, bgr2rgb=True)
print("Validation set samples: ", len(val_data))

# iterations
per_epoch_iteration = 1000
total_iteration = per_epoch_iteration * opt.end_epoch

# loss function
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()

# model
pretrained_model_path = opt.pretrained_model_path
method = opt.method
model = model_generator(method, pretrained_model_path).cuda()
n_params = sum(param.numel() for param in model.parameters())
print(f'Parameters: {n_params:,} ({n_params/1e6:.2f}M)')

# output path
date_time = time2file_name(str(datetime.datetime.now()))
opt.outf = opt.outf + date_time
os.makedirs(opt.outf, exist_ok=True)

if torch.cuda.is_available():
    model.cuda()
    criterion_mrae.cuda()
    criterion_rmse.cuda()
    criterion_psnr.cuda()

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=opt.init_lr, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration, eta_min=1e-6)
scaler = GradScaler('cuda', enabled=opt.amp)

# logging
log_dir = os.path.join(opt.outf, 'train.log')
logger = initialize_logger(log_dir)

# Resume
resume_file = opt.pretrained_model_path
start_iteration = 0
if resume_file is not None:
    if os.path.isfile(resume_file):
        print(f"=> loading checkpoint '{resume_file}'")
        checkpoint = torch.load(resume_file, weights_only=False)
        if 'state_dict' in checkpoint:
            model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}, strict=True)
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'iter' in checkpoint:
                start_iteration = checkpoint['iter']
        else:
            model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()}, strict=False)
        print(f'Loaded. Resuming from iter {start_iteration}')


def validate(val_loader, model):
    model.eval()
    losses_mrae = AverageMeter()
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()
    for input, target in val_loader:
        input, target = input.cuda(), target.cuda()
        with torch.no_grad():
            with autocast('cuda', enabled=opt.amp):
                output = model(input)
            # Center crop for evaluation (ignore border artifacts)
            h, w = target.shape[2], target.shape[3]
            crop = min(h, w) // 4
            loss_mrae = criterion_mrae(output[:, :, crop:-crop, crop:-crop] if crop > 0 else output,
                                       target[:, :, crop:-crop, crop:-crop] if crop > 0 else target)
            loss_rmse = criterion_rmse(output[:, :, crop:-crop, crop:-crop] if crop > 0 else output,
                                       target[:, :, crop:-crop, crop:-crop] if crop > 0 else target)
            loss_psnr = criterion_psnr(output[:, :, crop:-crop, crop:-crop] if crop > 0 else output,
                                       target[:, :, crop:-crop, crop:-crop] if crop > 0 else target)
        losses_mrae.update(loss_mrae.data)
        losses_rmse.update(loss_rmse.data)
        losses_psnr.update(loss_psnr.data)
    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg


def main():
    cudnn.benchmark = True
    iteration = start_iteration
    record_mrae_loss = 1000
    best_psnr = 0

    print(f'Training config: batch={opt.batch_size}, amp={opt.amp}, grad_accum={opt.grad_accum}, '
          f'effective_batch={opt.batch_size * opt.grad_accum}')
    print(f'Total iterations: {total_iteration} ({opt.end_epoch} epochs x {per_epoch_iteration} iters)')

    while iteration < total_iteration:
        model.train()
        losses = AverageMeter()
        train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True,
                                  num_workers=opt.num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False,
                                num_workers=opt.num_workers, pin_memory=True)

        optimizer.zero_grad()
        accum_count = 0

        for i, (images, labels) in enumerate(train_loader):
            if iteration >= total_iteration:
                break

            images, labels = images.cuda(), labels.cuda()
            lr = optimizer.param_groups[0]['lr']

            with autocast('cuda', enabled=opt.amp):
                output = model(images)
                loss = criterion_mrae(output, labels)
                loss = loss / opt.grad_accum  # normalize for accumulation

            scaler.scale(loss).backward()
            accum_count += 1

            if accum_count % opt.grad_accum == 0:
                if opt.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            losses.update(loss.data * opt.grad_accum)
            iteration += 1

            if iteration % 20 == 0:
                print(f'[iter:{iteration}/{total_iteration}], lr={lr:.9f}, train_loss={losses.avg:.9f}')

            if iteration % 1000 == 0:
                mrae_loss, rmse_loss, psnr_loss = validate(val_loader, model)
                print(f'Val MRAE:{mrae_loss:.6f}, RMSE:{rmse_loss:.6f}, PSNR:{psnr_loss:.4f}')

                # Save checkpoint
                improved = mrae_loss < record_mrae_loss
                if improved or iteration % 5000 == 0:
                    save_checkpoint(opt.outf, iteration // 1000, iteration, model, optimizer)
                    if improved:
                        record_mrae_loss = mrae_loss
                    if psnr_loss > best_psnr:
                        best_psnr = psnr_loss
                        # Save best model separately
                        torch.save({'state_dict': model.state_dict()},
                                   os.path.join(opt.outf, 'best.pth'))
                    print(f'  Saved (best PSNR: {best_psnr:.4f})')

                logger.info(f"Iter[{iteration:06d}], Epoch[{iteration // 1000:03d}], "
                            f"lr={lr:.9f}, Train_MRAE={losses.avg:.9f}, "
                            f"Val_MRAE={mrae_loss:.6f}, Val_RMSE={rmse_loss:.6f}, Val_PSNR={psnr_loss:.4f}")

            # Flush any remaining gradients
            if accum_count % opt.grad_accum != 0 and i == len(train_loader) - 1:
                if opt.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

    print(f'\nTraining complete. Best PSNR: {best_psnr:.4f}')
    return 0


if __name__ == '__main__':
    main()
    print(torch.__version__)
