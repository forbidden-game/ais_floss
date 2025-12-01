#!/usr/bin/env python
"""
Comprehensive evaluation script for FLOSS-AIS model.
Includes:
- SI-SNR and SNR at different input SNR levels
- BER (Bit Error Rate) analysis
- Visualization plots
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json

from src.data.ais_generator import AISConfig, AISSignalGenerator
from src.data.channel_model import ChannelConfig
from src.data.collision_generator import CollisionConfig, CollisionGenerator
from src.models.floss_ais import FLOSSAIS
from src.utils.metrics import compute_metrics, compute_si_snr, compute_snr


def demodulate_gmsk(signal_iq, samples_per_bit=10):
    """
    Simple GMSK demodulation using differential phase detection.

    Args:
        signal_iq: [2, T] IQ signal (I, Q)
        samples_per_bit: samples per bit

    Returns:
        bits: demodulated bits
    """
    # Convert to complex
    signal = signal_iq[0] + 1j * signal_iq[1]

    # Differential phase
    phase = np.angle(signal)
    phase_diff = np.diff(phase)

    # Unwrap phase
    phase_diff = np.unwrap(phase_diff)

    # Sample at bit centers
    num_bits = len(signal) // samples_per_bit
    bits = []

    for i in range(num_bits):
        start = i * samples_per_bit
        end = start + samples_per_bit
        if end > len(phase_diff):
            break

        # Average phase difference in this bit period
        avg_phase = np.mean(phase_diff[start:end])

        # Decision: positive phase = 1, negative = 0
        bits.append(1 if avg_phase > 0 else 0)

    return np.array(bits)


def compute_ber(pred_bits, true_bits):
    """Compute Bit Error Rate."""
    min_len = min(len(pred_bits), len(true_bits))
    if min_len == 0:
        return 1.0

    errors = np.sum(pred_bits[:min_len] != true_bits[:min_len])
    return errors / min_len


def find_best_permutation(separated, sources, bits_list):
    """
    Find the best permutation of separated signals to match sources.
    Returns reordered separated signals and corresponding source indices.
    """
    K = separated.shape[0]

    if K == 2:
        # Try both permutations
        si_snr1 = compute_si_snr(
            torch.tensor(separated[0:1]),
            torch.tensor(sources[0:1])
        ).item() + compute_si_snr(
            torch.tensor(separated[1:2]),
            torch.tensor(sources[1:2])
        ).item()

        si_snr2 = compute_si_snr(
            torch.tensor(separated[0:1]),
            torch.tensor(sources[1:2])
        ).item() + compute_si_snr(
            torch.tensor(separated[1:2]),
            torch.tensor(sources[0:1])
        ).item()

        if si_snr1 >= si_snr2:
            return separated, [0, 1]
        else:
            return separated[[1, 0]], [1, 0]

    return separated, list(range(K))


def evaluate_at_snr(model, generator, snr_db, num_samples, device, samples_per_bit=10):
    """Evaluate model at a specific SNR level."""
    si_snrs = []
    snrs = []
    bers = []

    for _ in range(num_samples):
        # Generate collision at specific SNR
        sample = generator.generate_collision(snr_db=snr_db)
        mixture = torch.tensor(sample['mixture']).unsqueeze(0).to(device)
        sources = sample['sources']
        true_bits_list = sample['bits']

        # Separate
        with torch.no_grad():
            separated = model.separate(mixture, num_steps=10)

        separated_np = separated[0].cpu().numpy()

        # Find best permutation
        separated_np, perm = find_best_permutation(separated_np, sources, true_bits_list)

        # Compute SI-SNR and SNR for each source
        for k in range(len(perm)):
            src_idx = perm[k]

            # SI-SNR
            si_snr = compute_si_snr(
                torch.tensor(separated_np[k:k+1]),
                torch.tensor(sources[src_idx:src_idx+1])
            ).item()
            si_snrs.append(si_snr)

            # SNR
            snr = compute_snr(
                torch.tensor(separated_np[k]),
                torch.tensor(sources[src_idx])
            ).item()
            snrs.append(snr)

            # BER
            pred_bits = demodulate_gmsk(separated_np[k], samples_per_bit)
            true_bits = true_bits_list[src_idx]
            ber = compute_ber(pred_bits, true_bits)
            bers.append(ber)

    return {
        'si_snr_mean': np.mean(si_snrs),
        'si_snr_std': np.std(si_snrs),
        'snr_mean': np.mean(snrs),
        'snr_std': np.std(snrs),
        'ber_mean': np.mean(bers),
        'ber_std': np.std(bers),
    }


def plot_snr_vs_performance(results, output_dir):
    """Plot performance metrics vs input SNR."""
    snr_levels = sorted(results.keys())

    si_snr_means = [results[snr]['si_snr_mean'] for snr in snr_levels]
    si_snr_stds = [results[snr]['si_snr_std'] for snr in snr_levels]

    snr_means = [results[snr]['snr_mean'] for snr in snr_levels]
    snr_stds = [results[snr]['snr_std'] for snr in snr_levels]

    ber_means = [results[snr]['ber_mean'] for snr in snr_levels]
    ber_stds = [results[snr]['ber_std'] for snr in snr_levels]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # SI-SNR plot
    axes[0].errorbar(snr_levels, si_snr_means, yerr=si_snr_stds,
                     marker='o', capsize=5, linewidth=2, markersize=8)
    axes[0].set_xlabel('Input SNR (dB)', fontsize=12)
    axes[0].set_ylabel('Output SI-SNR (dB)', fontsize=12)
    axes[0].set_title('SI-SNR vs Input SNR', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)

    # SNR plot
    axes[1].errorbar(snr_levels, snr_means, yerr=snr_stds,
                     marker='s', capsize=5, linewidth=2, markersize=8, color='green')
    axes[1].set_xlabel('Input SNR (dB)', fontsize=12)
    axes[1].set_ylabel('Output SNR (dB)', fontsize=12)
    axes[1].set_title('SNR vs Input SNR', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    # Add reference line (input = output)
    axes[1].plot(snr_levels, snr_levels, 'r--', alpha=0.5, label='Input=Output')
    axes[1].legend()

    # BER plot (log scale)
    axes[2].errorbar(snr_levels, ber_means, yerr=ber_stds,
                     marker='^', capsize=5, linewidth=2, markersize=8, color='red')
    axes[2].set_xlabel('Input SNR (dB)', fontsize=12)
    axes[2].set_ylabel('BER', fontsize=12)
    axes[2].set_title('BER vs Input SNR', fontsize=14)
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3, which='both')
    axes[2].set_ylim([1e-4, 1])

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_vs_snr.png', dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir / 'performance_vs_snr.png'}")


def plot_separation_examples(model, generator, output_dir, device, num_examples=5):
    """Plot separation examples at different SNR levels."""
    snr_examples = [-5, 0, 5, 10, 15]

    fig, axes = plt.subplots(num_examples, 4, figsize=(16, 3*num_examples))

    for i, snr_db in enumerate(snr_examples):
        sample = generator.generate_collision(snr_db=snr_db)
        mixture = torch.tensor(sample['mixture']).unsqueeze(0).to(device)
        sources = sample['sources']

        with torch.no_grad():
            separated = model.separate(mixture, num_steps=10)

        separated_np = separated[0].cpu().numpy()
        separated_np, _ = find_best_permutation(separated_np, sources, sample['bits'])

        # Time axis (in samples)
        t = np.arange(sources.shape[-1])

        # Mixture
        axes[i, 0].plot(t, sample['mixture'][0], 'b-', alpha=0.7, linewidth=0.5)
        axes[i, 0].set_title(f'Mixture (SNR={snr_db}dB)')
        axes[i, 0].set_ylabel('I component')

        # Ground truth sources
        axes[i, 1].plot(t, sources[0, 0], 'g-', alpha=0.7, linewidth=0.5, label='Source 1')
        axes[i, 1].plot(t, sources[1, 0], 'r-', alpha=0.7, linewidth=0.5, label='Source 2')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].legend(fontsize=8)

        # Separated sources
        axes[i, 2].plot(t, separated_np[0, 0], 'g-', alpha=0.7, linewidth=0.5, label='Sep 1')
        axes[i, 2].plot(t, separated_np[1, 0], 'r-', alpha=0.7, linewidth=0.5, label='Sep 2')
        axes[i, 2].set_title('Separated')
        axes[i, 2].legend(fontsize=8)

        # Error
        error1 = sources[0, 0] - separated_np[0, 0]
        error2 = sources[1, 0] - separated_np[1, 0]
        axes[i, 3].plot(t, error1, 'g-', alpha=0.7, linewidth=0.5, label='Error 1')
        axes[i, 3].plot(t, error2, 'r-', alpha=0.7, linewidth=0.5, label='Error 2')
        axes[i, 3].set_title('Separation Error')
        axes[i, 3].legend(fontsize=8)

    for ax in axes[-1]:
        ax.set_xlabel('Samples')

    plt.tight_layout()
    plt.savefig(output_dir / 'separation_examples.png', dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir / 'separation_examples.png'}")


def plot_spectrogram_comparison(model, generator, output_dir, device):
    """Plot spectrogram comparison for a sample."""
    sample = generator.generate_collision(snr_db=10)
    mixture = torch.tensor(sample['mixture']).unsqueeze(0).to(device)
    sources = sample['sources']

    with torch.no_grad():
        separated = model.separate(mixture, num_steps=10)

    separated_np = separated[0].cpu().numpy()
    separated_np, _ = find_best_permutation(separated_np, sources, sample['bits'])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Convert to complex for spectrogram
    mixture_complex = sample['mixture'][0] + 1j * sample['mixture'][1]
    source1_complex = sources[0, 0] + 1j * sources[0, 1]
    source2_complex = sources[1, 0] + 1j * sources[1, 1]
    sep1_complex = separated_np[0, 0] + 1j * separated_np[0, 1]
    sep2_complex = separated_np[1, 0] + 1j * separated_np[1, 1]

    # Spectrogram parameters
    nperseg = 128

    from scipy import signal as scipy_signal

    # Row 1: Mixture and ground truth
    f, t, Sxx = scipy_signal.spectrogram(mixture_complex, fs=96000, nperseg=nperseg)
    axes[0, 0].pcolormesh(t, f/1000, 10*np.log10(np.abs(Sxx)+1e-10), shading='gouraud', cmap='viridis')
    axes[0, 0].set_title('Mixture')
    axes[0, 0].set_ylabel('Frequency (kHz)')

    f, t, Sxx = scipy_signal.spectrogram(source1_complex, fs=96000, nperseg=nperseg)
    axes[0, 1].pcolormesh(t, f/1000, 10*np.log10(np.abs(Sxx)+1e-10), shading='gouraud', cmap='viridis')
    axes[0, 1].set_title('Ground Truth Source 1')

    f, t, Sxx = scipy_signal.spectrogram(source2_complex, fs=96000, nperseg=nperseg)
    axes[0, 2].pcolormesh(t, f/1000, 10*np.log10(np.abs(Sxx)+1e-10), shading='gouraud', cmap='viridis')
    axes[0, 2].set_title('Ground Truth Source 2')

    # Row 2: Separated
    axes[1, 0].axis('off')
    axes[1, 0].text(0.5, 0.5, 'FLOSS-AIS\nSeparation', ha='center', va='center',
                    fontsize=16, fontweight='bold')

    f, t, Sxx = scipy_signal.spectrogram(sep1_complex, fs=96000, nperseg=nperseg)
    axes[1, 1].pcolormesh(t, f/1000, 10*np.log10(np.abs(Sxx)+1e-10), shading='gouraud', cmap='viridis')
    axes[1, 1].set_title('Separated Source 1')
    axes[1, 1].set_ylabel('Frequency (kHz)')
    axes[1, 1].set_xlabel('Time (s)')

    f, t, Sxx = scipy_signal.spectrogram(sep2_complex, fs=96000, nperseg=nperseg)
    axes[1, 2].pcolormesh(t, f/1000, 10*np.log10(np.abs(Sxx)+1e-10), shading='gouraud', cmap='viridis')
    axes[1, 2].set_title('Separated Source 2')
    axes[1, 2].set_xlabel('Time (s)')

    plt.tight_layout()
    plt.savefig(output_dir / 'spectrogram_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir / 'spectrogram_comparison.png'}")


def plot_ber_heatmap(model, generator, output_dir, device, samples_per_snr=20):
    """Plot BER heatmap vs SNR and SIR."""
    snr_levels = [-5, 0, 5, 10, 15, 20]
    sir_levels = [-10, -5, 0, 5, 10]

    ber_matrix = np.zeros((len(snr_levels), len(sir_levels)))

    for i, snr_db in enumerate(tqdm(snr_levels, desc='SNR levels')):
        for j, sir_db in enumerate(sir_levels):
            bers = []

            # Modify collision config for specific SIR
            generator.collision_config.sir_range_db = (sir_db, sir_db)

            for _ in range(samples_per_snr):
                sample = generator.generate_collision(snr_db=snr_db)
                mixture = torch.tensor(sample['mixture']).unsqueeze(0).to(device)

                with torch.no_grad():
                    separated = model.separate(mixture, num_steps=10)

                separated_np = separated[0].cpu().numpy()
                separated_np, perm = find_best_permutation(
                    separated_np, sample['sources'], sample['bits']
                )

                # Compute BER for both sources
                for k in range(2):
                    pred_bits = demodulate_gmsk(separated_np[k], 10)
                    true_bits = sample['bits'][perm[k]]
                    bers.append(compute_ber(pred_bits, true_bits))

            ber_matrix[i, j] = np.mean(bers)

    # Reset collision config
    generator.collision_config.sir_range_db = (-10, 10)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(ber_matrix, cmap='RdYlGn_r', aspect='auto',
                   vmin=0, vmax=0.5)

    ax.set_xticks(range(len(sir_levels)))
    ax.set_xticklabels(sir_levels)
    ax.set_yticks(range(len(snr_levels)))
    ax.set_yticklabels(snr_levels)

    ax.set_xlabel('SIR (dB)', fontsize=12)
    ax.set_ylabel('SNR (dB)', fontsize=12)
    ax.set_title('BER Heatmap vs SNR and SIR', fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('BER', fontsize=12)

    # Add text annotations
    for i in range(len(snr_levels)):
        for j in range(len(sir_levels)):
            text = f'{ber_matrix[i, j]:.3f}'
            color = 'white' if ber_matrix[i, j] > 0.25 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'ber_heatmap.png', dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir / 'ber_heatmap.png'}")

    return ber_matrix


def main():
    parser = argparse.ArgumentParser(description='Comprehensive FLOSS-AIS evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--num_samples', type=int, default=50, help='Samples per SNR level')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')

    # Model config (fallback)
    parser.add_argument('--encoder_dim', type=int, default=128)
    parser.add_argument('--separator_layers', type=int, default=4)
    parser.add_argument('--separator_heads', type=int, default=4)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load model
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    if 'model_config' in checkpoint:
        cfg = checkpoint['model_config']
        print(f"Using model config from checkpoint: {cfg}")
    else:
        print("No model_config in checkpoint, using command line args")
        cfg = {
            'encoder_dim': args.encoder_dim,
            'separator_layers': args.separator_layers,
            'separator_heads': args.separator_heads,
            'num_sources': 2,
            'num_sampling_steps': 10,
        }

    model = FLOSSAIS(
        encoder_type=cfg.get('encoder_type', 'standard'),
        encoder_dim=cfg.get('encoder_dim', 128),
        num_sources=cfg.get('num_sources', 2),
        separator_layers=cfg.get('separator_layers', 4),
        separator_heads=cfg.get('separator_heads', 4),
        num_sampling_steps=cfg.get('num_sampling_steps', 10),
        decode_bits=cfg.get('decode_bits', False),
    ).to(args.device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded successfully")

    # Create generator
    ais_config = AISConfig()
    channel_config = ChannelConfig()
    collision_config = CollisionConfig()
    generator = CollisionGenerator(ais_config, channel_config, collision_config)

    # Evaluate at different SNR levels
    print("\n" + "="*60)
    print("Evaluating at different SNR levels...")
    print("="*60)

    snr_levels = [-5, 0, 5, 10, 15, 20]
    results = {}

    for snr_db in tqdm(snr_levels, desc='SNR levels'):
        results[snr_db] = evaluate_at_snr(
            model, generator, snr_db, args.num_samples, args.device,
            samples_per_bit=ais_config.samples_per_bit
        )
        print(f"SNR={snr_db:3d}dB: SI-SNR={results[snr_db]['si_snr_mean']:.2f}dB, "
              f"SNR={results[snr_db]['snr_mean']:.2f}dB, "
              f"BER={results[snr_db]['ber_mean']:.4f}")

    # Generate plots
    print("\n" + "="*60)
    print("Generating plots...")
    print("="*60)

    plot_snr_vs_performance(results, output_dir)
    plot_separation_examples(model, generator, output_dir, args.device)
    plot_spectrogram_comparison(model, generator, output_dir, args.device)

    print("\nGenerating BER heatmap (this may take a while)...")
    ber_matrix = plot_ber_heatmap(model, generator, output_dir, args.device,
                                   samples_per_snr=args.num_samples // 2)

    # Save results to JSON
    results_json = {
        'snr_results': {str(k): v for k, v in results.items()},
        'ber_heatmap': ber_matrix.tolist(),
        'model_config': cfg,
        'num_samples': args.num_samples,
    }

    with open(output_dir / 'comprehensive_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\nResults saved to {output_dir / 'comprehensive_results.json'}")

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"\n{'SNR (dB)':<10} {'SI-SNR (dB)':<15} {'Output SNR (dB)':<18} {'BER':<10}")
    print("-" * 55)
    for snr_db in snr_levels:
        r = results[snr_db]
        print(f"{snr_db:<10} {r['si_snr_mean']:>6.2f} +/- {r['si_snr_std']:<6.2f} "
              f"{r['snr_mean']:>6.2f} +/- {r['snr_std']:<6.2f} "
              f"{r['ber_mean']:.4f}")

    print("\n" + "="*60)
    print(f"All results saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
