import uproot
import awkward as ak
import numpy as np
import torch
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader, random_split
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

from carl import CARL 

# -------------------------------
# Dataset
# -------------------------------

def load_root_features(file_path, tree_name="LHEF", epsilon=1e-12, max_weight=1e6):
    tree = uproot.open(file_path)[tree_name]
    pt = tree["Particle.PT"].array()
    eta = tree["Particle.Eta"].array()
    phi = tree["Particle.Phi"].array()
    mass = tree["Particle.M"].array()
    SM_amp = tree["SM_Amplitude"].array()
    NC_amp = tree["NC_Amplitude"].array()

    mask = ak.num(pt) > 7

    # Extract 3rd and 4th particles
    pt3, pt4 = ak.to_numpy(pt[mask][:, 2]), ak.to_numpy(pt[mask][:, 3])
    eta3, eta4 = ak.to_numpy(eta[mask][:, 2]), ak.to_numpy(eta[mask][:, 3])
    phi3, phi4 = ak.to_numpy(phi[mask][:, 2]), ak.to_numpy(phi[mask][:, 3])
    mass3, mass4 = ak.to_numpy(mass[mask][:, 2]), ak.to_numpy(mass[mask][:, 3])

    safe_NC_amp = ak.to_numpy(NC_amp[mask])
    safe_SM_amp = ak.to_numpy(SM_amp[mask])
    
    # Safe ratio with epsilon
    ratio = np.zeros(len(safe_NC_amp))
    nonzero_mask = np.abs(safe_SM_amp) > epsilon
    ratio[nonzero_mask] = safe_NC_amp[nonzero_mask] / safe_SM_amp[nonzero_mask]

    sample_weight = 1.0 + ratio
    sample_weight = np.clip(sample_weight, 0, max_weight)

    # Stack features
    X = np.stack([pt3, eta3, phi3, mass3,
                  pt4, eta4, phi4, mass4,
                  ], axis=1)
    W = np.array(sample_weight)

    # Remove NaN/Inf
    mask_valid = (
      ~np.isnan(X).any(axis=1)
    & ~np.isinf(X).any(axis=1)
    & ~np.isnan(sample_weight)
    & ~np.isinf(sample_weight)
    )

    X = X[mask_valid]
    W = W[mask_valid]
    return X,W

def build_dataset(X, W, theta_value, is_signal, X_mean, X_std):

    X_norm = (X - X_mean) / X_std

    N = X.shape[0]
    y = np.ones(N) if is_signal else np.zeros(N)
    theta = np.full(N, theta_value)

    return TensorDataset(
        torch.tensor(X_norm, dtype=torch.float32),
        torch.tensor(theta, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(W, dtype=torch.float32),
    )


# -------------------------------
# Likelihood
# -------------------------------

def compute_log_likelihood(model, X, W, theta, eps=1e-6):
    model.eval()

    X_t = torch.tensor(X, dtype=torch.float32)
    theta_t = torch.ones(X.shape[0]) * theta
    W_t = torch.tensor(W, dtype=torch.float32)

    x_input = torch.cat([X_t, theta_t.view(-1,1)], dim=1)

    with torch.no_grad():
        s = model(x_input).flatten()

    s = torch.clamp(s, eps, 1 - eps)

    log_r = torch.log(s / (1 - s))

    return (log_r * W_t).sum().item()

# def compute_log_likelihood(model, X, theta, X_mean, X_std, eps=1e-6):
#     model.eval()

#     X_norm = (X - X_mean) / X_std

#     X_t = torch.tensor(X_norm, dtype=torch.float32)
#     theta_t = torch.ones(X.shape[0]) * theta

#     x_input = torch.cat([X_t, theta_t.view(-1,1)], dim=1)

#     with torch.no_grad():
#         s = model(x_input).flatten()

#     s = torch.clamp(s, eps, 1 - eps)

#     log_r = torch.log(s / (1 - s))

#     return (log_r).sum().item()


# -------------------------------
# Scan
# -------------------------------

def scan_theta(model, X, W, theta_grid):
    logL = []
    for t in theta_grid:
        logL.append(compute_log_likelihood(model, X, W, t))

    logL = np.array(logL)
    best_theta = theta_grid[np.argmax(logL)]

    return best_theta, logL

# def scan_theta(model, X, theta_grid, X_mean, X_std):
#     logL = []
#     for t in theta_grid:
#         logL.append(compute_log_likelihood(model, X, t, X_mean, X_std))

#     logL = np.array(logL)
#     best_theta = theta_grid[np.argmax(logL)]

#     return best_theta, logL

# -------------------------------
# Main
# -------------------------------

if __name__ == "__main__":
    files = ["/eos/user/y/yzhang4/LIV/LO/MG_LHE_ppZZto4L_LO_theta_1e-3/result/total.root",
             "/eos/user/y/yzhang4/LIV/LO/MG_LHE_ppZZto4L_LO_theta_1e-4/result/total.root",
             "/eos/user/y/yzhang4/LIV/LO/MG_LHE_ppZZto4L_LO_theta_1e-5/result/total.root",
             "/eos/user/y/yzhang4/LIV/LO/MG_LHE_ppZZto4L_LO_theta_1e-6/result/total.root"]
    theta_values = [1e-3, 1e-4, 1e-5,1e-6]
    # theta_ref = [0.0]

    data_list = [load_root_features(f) for f in files]
    X_signal = [d[0] for d in data_list]
    W_signal = [d[1] for d in data_list]
    X_ref, W_ref = load_root_features("/eos/user/y/yzhang4/LIV/LO/MG_LHE_ppZZto4L_LO_theta_1e-0/result/total.root")
    W_ref = np.ones_like(W_ref)

    # normalization
    all_features = np.concatenate(X_signal + [X_ref])
    X_mean = np.mean(all_features, axis=0)
    X_std = np.std(all_features, axis=0) + 1e-12

    datasets = []
    for X, W, theta in zip(X_signal, W_signal, theta_values):
        datasets.append(build_dataset(X, W, theta, True, X_mean, X_std))
    datasets.append(build_dataset(X_ref, W_ref, 0.0, False, X_mean, X_std))
    dataset = ConcatDataset(datasets)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=256, num_workers=8)

    model = CARL(
        n_features=X_ref.shape[1] + 1, # +1 for theta
        n_layers=3,
        n_nodes=128,
        learning_rate=1e-3
    )

    trainer = L.Trainer(max_epochs=20)
    trainer.fit(model, train_loader, val_loader)

    # scan
    theta_grid = np.geomspace(1e-6, 1e-1, 800)
    X_obs, W_obs = load_root_features("/eos/user/y/yzhang4/LIV/LO/MG_LHE_ppZZto4L_LO_theta_5e-5/result/total.root")
    # W_obs = np.ones_like(W_obs)
    # best_theta, logL = scan_theta(model, X_obs, theta_grid, X_mean, X_std)
    best_theta, logL = scan_theta(model, X_obs, W_obs, theta_grid)

    print("Best theta:", best_theta)


    delta_logL = -2 * (logL - np.max(logL))
    plt.figure()
    plt.plot(theta_grid, delta_logL)
    plt.xscale("log")

    plt.xlabel("theta")
    plt.ylabel(r"$-2\Delta \log L$")
    plt.title("Likelihood scan")

    plt.axhline(1.0, linestyle='--', color='red', label='1σ')
    plt.axhline(4.0, linestyle='--', color='blue', label='2σ')
    plt.axvline(best_theta, linestyle='--', color='black', label='Best fit')

    plt.legend()
    plt.savefig("Likelihood scan.pdf")

