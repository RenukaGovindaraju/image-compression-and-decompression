# /home/metalab/my_research_project/train1.py
# FINAL VERSION WITH ROBUST DATA TRANSFORMS TO HANDLE SMALL IMAGES

import argparse
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging
from torchvision import transforms
from compressai.datasets import ImageFolder
from torch.utils.data import DataLoader

from model1 import MySOTAMambaModel

torch.manual_seed(42)
random.seed(42)

class RateDistortionLoss(nn.Module):
    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
    def forward(self, output, target):
        N, _, H, W = target.size()
        num_pixels = N * H * W
        bpp_y = torch.log(output["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels)
        bpp_z = torch.log(output["likelihoods"]["z"]).sum() / (-math.log(2) * num_pixels)
        bpp_loss = bpp_y + bpp_z
        mse_loss = self.mse(output["x_hat"], target)
        rd_loss = self.lmbda * 255**2 * mse_loss + bpp_loss
        return rd_loss, bpp_loss, mse_loss

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    root_logger = logging.getLogger()
    if root_logger.hasHandlers(): root_logger.handlers.clear()
    file_handler = logging.FileHandler(os.path.join(log_dir, "training.log"))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-v", "--val-dataset", type=str, required=True)
    parser.add_argument("-e", "--epochs", type=int, default=300)
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-4)
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="The physical batch size that fits in GPU memory.")
    parser.add_argument("--lambda", type=float, required=True, dest="lmbda")
    parser.add_argument("--cuda", action='store_true')
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--accumulation-steps", type=int, default=8, help="Number of steps to accumulate gradients.")
    args = parser.parse_args()

    exp_dir = os.path.join("experiments", args.experiment_name)
    chk_dir = os.path.join(exp_dir, "checkpoints")
    log_dir = os.path.join(exp_dir, "logs")
    os.makedirs(chk_dir, exist_ok=True)
    setup_logging(log_dir)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    effective_batch_size = args.batch_size * args.accumulation_steps
    logging.info(f"Experiment: {args.experiment_name} | Lambda: {args.lmbda:.6f}")
    logging.info(f"Physical Batch Size: {args.batch_size} | Accumulation Steps: {args.accumulation_steps} | Effective Batch Size: {effective_batch_size}")
    logging.info(f"Using device: {device}")
    
    # Using a smaller crop size (192) and resizing first to handle small images
    train_transforms = transforms.Compose([
        transforms.Resize(192, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(192),
        transforms.ToTensor()
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(192, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(192),
        transforms.ToTensor()
    ])
    
    train_dataset = ImageFolder(args.dataset, split="", transform=train_transforms)
    val_dataset = ImageFolder(args.val_dataset, split="", transform=val_transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    net = MySOTAMambaModel(N=192, M=320).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.5)
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    start_epoch, best_loss = 0, float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        net.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("loss", float("inf"))
        logging.info(f"Resumed from epoch {start_epoch}, best loss: {best_loss:.4f}")

    for epoch in range(start_epoch, args.epochs):
        net.train()
        logging.info(f"--- Epoch {epoch+1}/{args.epochs}, LR: {optimizer.param_groups[0]['lr']:.2e} ---")
        
        optimizer.zero_grad()
        for i, d in enumerate(train_loader):
            d = d.to(device)
            out_net = net(d)
            rd_loss, bpp_loss, mse_loss = criterion(out_net, d)
            
            rd_loss = rd_loss / args.accumulation_steps
            rd_loss.backward()

            if (i + 1) % args.accumulation_steps == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if (i+1) % 200 == 0:
                logging.info(f"Batch {i+1}/{len(train_loader)} | Loss: {(rd_loss.item() * args.accumulation_steps):.4f} | BPP: {bpp_loss.item():.4f} | MSE: {mse_loss.item():.6f}")

        net.eval()
        val_loss, val_bpp, val_psnr = 0, 0, 0
        with torch.no_grad():
            for d in val_loader:
                d = d.to(device); out_net = net(d)
                rd_loss, bpp_loss, mse_loss = criterion(out_net, d)
                val_loss += rd_loss.item(); val_bpp += bpp_loss.item()
                val_psnr += 10 * math.log10(1./mse_loss.item()) if mse_loss > 0 else 100
        val_loss /= len(val_loader); val_bpp /= len(val_loader); val_psnr /= len(val_loader)
        logging.info(f"Validation | Loss: {val_loss:.4f} | BPP: {val_bpp:.4f} | PSNR: {val_psnr:.3f} dB")
        lr_scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            net.update(force=True)
            filename = f"best_model_e{epoch+1:03d}_loss{val_loss:.4f}.pth.tar"
            torch.save({"epoch": epoch, "state_dict": net.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": lr_scheduler.state_dict(), "loss": val_loss}, os.path.join(chk_dir, filename))
            logging.info(f"🏆 New best model saved: {filename}")
        if (epoch + 1) % args.save_every == 0:
            net.update(force=True)
            torch.save({"epoch": epoch, "state_dict": net.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": lr_scheduler.state_dict(), "loss": val_loss}, os.path.join(chk_dir, f"ckpt_e{epoch+1:03d}.pth.tar"))
            logging.info(f"💾 Periodic checkpoint saved.")

if __name__ == "__main__": main()