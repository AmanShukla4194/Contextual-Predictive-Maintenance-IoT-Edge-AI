import numpy as np
import pandas as pd


def add_fft_features(df, sensor_cols, window=24):
    df = df.copy().sort_values(["machineID", "datetime"])

    for col in sensor_cols:
        dominant_freq = []
        spectral_energy = []

        for _, grp in df.groupby("machineID"):
            vals = grp[col].values
            freqs = []
            energies = []
            for i in range(len(vals)):
                segment = vals[max(0, i - window + 1): i + 1]
                if len(segment) < 4:
                    freqs.append(0.0)
                    energies.append(0.0)
                    continue
                fft_vals = np.abs(np.fft.rfft(segment - segment.mean()))
                freqs.append(float(np.argmax(fft_vals[1:]) + 1) if len(fft_vals) > 1 else 0.0)
                energies.append(float(np.sum(fft_vals ** 2)))
            dominant_freq.extend(freqs)
            spectral_energy.extend(energies)

        df[f"{col}_dom_freq"] = dominant_freq
        df[f"{col}_spectral_energy"] = spectral_energy

    return df


def spectral_entropy(signal):
    fft_vals = np.abs(np.fft.rfft(signal - signal.mean()))
    power = fft_vals ** 2
    total = power.sum() + 1e-9
    prob = power / total
    return float(-np.sum(prob * np.log(prob + 1e-9)))


def add_spectral_entropy(df, sensor_cols, window=24):
    df = df.copy().sort_values(["machineID", "datetime"])
    for col in sensor_cols:
        entropy_vals = []
        for _, grp in df.groupby("machineID"):
            vals = grp[col].values
            ents = []
            for i in range(len(vals)):
                segment = vals[max(0, i - window + 1): i + 1]
                ents.append(spectral_entropy(segment) if len(segment) >= 4 else 0.0)
            entropy_vals.extend(ents)
        df[f"{col}_spectral_entropy"] = entropy_vals
    return df
