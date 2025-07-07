import argparse
import os
import glob
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from model1 import MySOTAMambaModel
from pytorch_msssim import ms_ssim
from tqdm import tqdm

def load_image(filepath):
    img = Image.open(filepath).convert('RGB')
    transform = transforms.ToTensor()
    return transform(img).unsqueeze(0)

def calculate_metrics(x, x_hat, likelihoods):
    EPS = 1e-9
    num_pixels = x.size(2) * x.size(3)

    # BPP
    bpp = sum(
        (-torch.log2(likelihood.clamp(min=EPS)).sum() / num_pixels).item()
        for likelihood in likelihoods.values()
    )

    # PSNR
    mse = F.mse_loss(x_hat, x, reduction='mean')
    psnr = 10 * torch.log10(1.0 / mse).item()

    # MS-SSIM
    ms_ssim_val = ms_ssim(x_hat, x, data_range=1.0, size_average=True).item()

    return bpp, psnr, ms_ssim_val

def save_image(tensor, filename):
    img = tensor.squeeze().detach().cpu().clamp(0, 1)
    img = transforms.ToPILImage()(img)
    img.save(filename)

def main(args):
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'

    # 🔁 Load model
    model = MySOTAMambaModel()
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device).eval()
    print(f"📦 Loaded model from: {args.model}")

    # 📁 Load dataset
    image_paths = sorted(glob.glob(os.path.join(args.dataset, '*.png')))
    if not image_paths:
        print("❌ No PNG images found in dataset path.")
        return

    total_psnr, total_bpp, total_msssim = 0, 0, 0

    print("🔍 Testing on Kodak:")

    for image_path in tqdm(image_paths):
        img_name = os.path.basename(image_path)
        x = load_image(image_path).to(device)

        with torch.no_grad():
            out = model(x)
            x_hat = out['x_hat']
            likelihoods = out['likelihoods']
            bpp, psnr, msssim = calculate_metrics(x, x_hat, likelihoods)

        print(f"{img_name} | PSNR: {psnr:.2f} dB | MS-SSIM: {msssim:.4f} | BPP: {bpp:.4f}")

        total_psnr += psnr
        total_bpp += bpp
        total_msssim += msssim

        if args.save_recon:
            save_dir = os.path.join("reconstructed_images", os.path.basename(args.model).split('.')[0])
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, img_name)
            save_image(x_hat, save_path)

    n = len(image_paths)
    print(f"\n📊 Averages on Kodak — PSNR: {total_psnr/n:.2f} dB | MS-SSIM: {total_msssim/n:.4f} | BPP: {total_bpp/n:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", required=True, help="Path to Kodak PNG images")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--save_recon", action="store_true", help="Save reconstructed images")
    args = parser.parse_args()
    main(args)
