#!/usr/bin/env python
# scripts/evaluate.py

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.data.collision_generator import CollisionGenerator
from src.models.floss_ais import FLOSSAIS
from src.utils.metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='results')

    # Model config (used as fallback if not in checkpoint)
    parser.add_argument('--encoder_type', type=str, default='standard')
    parser.add_argument('--encoder_dim', type=int, default=512)
    parser.add_argument('--num_sources', type=int, default=2)
    parser.add_argument('--separator_layers', type=int, default=6)
    parser.add_argument('--separator_heads', type=int, default=8)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load checkpoint
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # Get model config from checkpoint or use args as fallback
    if 'model_config' in checkpoint:
        cfg = checkpoint['model_config']
        print(f"Using model config from checkpoint: {cfg}")
    else:
        print("No model_config in checkpoint, using command line args")
        cfg = {
            'encoder_type': args.encoder_type,
            'encoder_dim': args.encoder_dim,
            'num_sources': args.num_sources,
            'separator_layers': args.separator_layers,
            'separator_heads': args.separator_heads,
            'num_sampling_steps': args.num_steps,
            'decode_bits': False,
        }

    # Create model with correct config
    model = FLOSSAIS(
        encoder_type=cfg.get('encoder_type', 'standard'),
        encoder_dim=cfg.get('encoder_dim', 512),
        num_sources=cfg.get('num_sources', 2),
        separator_layers=cfg.get('separator_layers', 6),
        separator_heads=cfg.get('separator_heads', 8),
        num_sampling_steps=cfg.get('num_sampling_steps', 10),
        decode_bits=cfg.get('decode_bits', False),
    ).to(args.device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded: encoder_dim={cfg.get('encoder_dim')}, "
          f"separator_layers={cfg.get('separator_layers')}, "
          f"separator_heads={cfg.get('separator_heads')}")

    # Generate test data
    generator = CollisionGenerator()

    snrs = []
    si_snrs = []

    print(f"Evaluating on {args.num_samples} samples...")
    for i in range(args.num_samples):
        # Generate collision
        sample = generator.generate_collision()
        mixture = torch.tensor(sample['mixture']).unsqueeze(0).to(args.device)
        sources = torch.tensor(sample['sources']).unsqueeze(0).to(args.device)

        # Separate
        with torch.no_grad():
            separated = model.separate(mixture, num_steps=args.num_steps)

        # Compute metrics
        metrics = compute_metrics(separated, sources)
        snrs.append(metrics['snr'])
        si_snrs.append(metrics['si_snr'])

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{args.num_samples} samples...")

        # Save example plots
        if i < 5:
            fig, axes = plt.subplots(4, 1, figsize=(12, 10))

            axes[0].plot(sample['mixture'][0])
            axes[0].set_title('Mixture (I)')

            axes[1].plot(sample['sources'][0, 0], label='Source 1')
            axes[1].plot(sample['sources'][1, 0], label='Source 2', alpha=0.7)
            axes[1].set_title('Ground Truth')
            axes[1].legend()

            sep_np = separated[0].cpu().numpy()
            axes[2].plot(sep_np[0, 0], label='Separated 1')
            axes[2].plot(sep_np[1, 0], label='Separated 2', alpha=0.7)
            axes[2].set_title('Separated')
            axes[2].legend()

            axes[3].text(0.5, 0.5,
                         f"SNR: {metrics['snr']:.2f} dB\nSI-SNR: {metrics['si_snr']:.2f} dB",
                         ha='center', va='center', fontsize=14)
            axes[3].axis('off')

            plt.tight_layout()
            plt.savefig(output_dir / f'sample_{i}.png', dpi=150)
            plt.close()

    # Summary
    print(f"\n=== Results ===")
    print(f"SNR:    {np.mean(snrs):.2f} +/- {np.std(snrs):.2f} dB")
    print(f"SI-SNR: {np.mean(si_snrs):.2f} +/- {np.std(si_snrs):.2f} dB")

    # Save results
    results = {
        'snr_mean': float(np.mean(snrs)),
        'snr_std': float(np.std(snrs)),
        'si_snr_mean': float(np.mean(si_snrs)),
        'si_snr_std': float(np.std(si_snrs)),
        'num_samples': args.num_samples,
        'num_steps': args.num_steps,
        'model_config': cfg,
    }

    import json
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'results.json'}")


if __name__ == '__main__':
    main()
