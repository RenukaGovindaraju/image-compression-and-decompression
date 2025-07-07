import argparse
import re
import matplotlib.pyplot as plt

# Parse log file argument
parser = argparse.ArgumentParser()
parser.add_argument('--log-file', type=str, required=True)
args = parser.parse_args()

# Lists to store metrics
bps, psnrs, losses = [], [], []

# Read the log file and extract validation metrics
with open(args.log_file, 'r') as f:
    for line in f:
        if 'Validation | Loss' in line:
            match = re.search(r'Loss: ([\d.]+) \| BPP: ([\d.]+) \| PSNR: ([\d.]+) dB', line)
            if match:
                loss, bpp, psnr = map(float, match.groups())
                losses.append(loss)
                bps.append(bpp)
                psnrs.append(psnr)

# Ensure data is available
if not bps:
    print("⚠️ No matching 'Validation' entries with Loss, BPP, and PSNR found in the log.")
    exit()

# Sort data by BPP
sorted_data = sorted(zip(bps, psnrs, losses), key=lambda x: x[0])
bps, psnrs, losses = zip(*sorted_data)

# Plot BPP vs PSNR and Loss
plt.figure(figsize=(15, 5))

# Plot BPP vs PSNR
plt.subplot(1, 2, 1)
plt.plot(bps, psnrs, marker='o', color='blue')
plt.xlabel('BPP')
plt.ylabel('PSNR (dB)')
plt.title('BPP vs PSNR')
plt.grid(True)

# Plot BPP vs Loss
plt.subplot(1, 2, 2)
plt.plot(bps, losses, marker='s', color='green')
plt.xlabel('BPP')
plt.ylabel('Loss')
plt.title('BPP vs Loss')
plt.grid(True)

# Save and display the plot
plt.tight_layout()
plt.savefig("validation_bpp_curves.png", dpi=300)
plt.show()
