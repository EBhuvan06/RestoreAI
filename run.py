import sys
import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add current directory so we can import the model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import KLARestorer


def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = KLARestorer(width=32).to(device)
    ckpt_path = Path(__file__).parent / "models" / "best_kla_restorer.pth"
    
    if not ckpt_path.exists():
        print(f"Error: Model checkpoint not found at {ckpt_path}")
        sys.exit(1)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # Get all .npy files
    files = sorted(list(input_dir.glob("*.npy")))
    if len(files) == 0:
        print("No .npy files found in input directory.")
        sys.exit(1)

    print(f"Found {len(files)} images to process.")

    with torch.no_grad():
        for fpath in tqdm(files, desc="Restoring"):
            # Load
            img = np.load(fpath).astype(np.float32)

            # Handle possible value ranges
            if img.max() > 1.5:
                img = img / 255.0

            if img.ndim == 3:
                img = img.squeeze()

            # To tensor
            x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(device)

            # Inference
            pred = model(x).clamp(0.0, 1.0)

            # Convert back to numpy (H, W)
            out = pred.squeeze().cpu().numpy().astype(np.float32)

            # Safety checks
            out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
            out = np.clip(out, 0.0, 1.0)

            # Save with same filename
            save_path = output_dir / fpath.name
            np.save(save_path, out)

    print(f"\nDone! Restored images saved to: {output_dir}")


if __name__ == "__main__":
    main()
