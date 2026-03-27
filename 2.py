# ===============================================
# SBI-based Likelihood Inference using PyTorch + ROOT
# Safe version: no divide-by-zero or NaN/Inf
# ===============================================

import uproot
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import awkward as ak
import matplotlib.pyplot as plt

# -------------------------------
# 1. Safe ROOT feature loader
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
    nonzero_mask = np.abs(safe_NC_amp) > epsilon
    ratio[nonzero_mask] = safe_SM_amp[nonzero_mask] / safe_NC_amp[nonzero_mask]

    sample_weight = 1.0 + ratio
    sample_weight = np.clip(sample_weight, 0, max_weight)

    # Stack features
    X = np.stack([pt3, eta3, phi3, mass3,
                  pt4, eta4, phi4, mass4,
                  sample_weight], axis=1)

    # Remove NaN/Inf
    mask_valid = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
    X = X[mask_valid]

    return X

# -------------------------------
# 2. Parameterized Neural Network
# -------------------------------

class LikelihoodRatioNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            # nn.ReLU(),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 128),
            # nn.ReLU(),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1)
        )

    def forward(self, x, theta):
        theta = theta.view(-1,1)
        theta_log = torch.log10(torch.clamp(theta, min=1e-12))
        theta = (theta_log + 5.0) / 2.0 
        # (torch.log10(theta + 1e-12) - (-5.0)) / 1.0
        x_input = torch.cat([x, theta], dim=1)
        return self.net(x_input)

# -------------------------------
# 3. Safe Dataset builder
# -------------------------------

def prepare_dataset(X_theta, X_ref, theta_value, weight_theta=None, weight_ref=None):
    N_theta, N_ref = X_theta.shape[0], X_ref.shape[0]
    
    y_theta = np.ones(N_theta)
    y_ref = np.zeros(N_ref)
    
    theta_theta = np.ones(N_theta) * theta_value
    theta_ref = np.zeros(N_ref)
    
    if weight_theta is None: weight_theta = np.ones(N_theta)
    if weight_ref is None: weight_ref = np.ones(N_ref)
    weights = np.concatenate([weight_theta, weight_ref], axis=0)

    X_all = np.concatenate([X_theta, X_ref], axis=0)
    y_all = np.concatenate([y_theta, y_ref], axis=0)
    theta_all = np.concatenate([theta_theta, theta_ref], axis=0)

    # Remove NaN/Inf rows
    mask_valid = ~np.isnan(X_all).any(axis=1) & ~np.isinf(X_all).any(axis=1)
    X_all, y_all, theta_all, weights = X_all[mask_valid], y_all[mask_valid], theta_all[mask_valid], weights[mask_valid]

    return TensorDataset(
        torch.tensor(X_all, dtype=torch.float32),
        torch.tensor(theta_all, dtype=torch.float32),
        torch.tensor(y_all, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32)
    )

# -------------------------------
# 4. Training function (safe)
# -------------------------------

def train_network(model, dataset, epochs=20, batch_size=128, lr=1e-3, clip_value=20):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, theta_batch, y_batch, w_batch in loader:
            optimizer.zero_grad()
            logits = model(X_batch, theta_batch).squeeze()
            # Clip logits to avoid log(0)
            logits = torch.clamp(logits, -clip_value, clip_value)
            loss = criterion(logits, y_batch)
            loss = (loss * w_batch).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.shape[0]
        print(f"Epoch {epoch+1}/{epochs}, Loss = {total_loss/len(dataset):.4f}")
    return model

# -------------------------------
# 5. Compute log likelihood safely
# -------------------------------

def compute_log_likelihood(model, X_obs, theta_value, clip_value=20):
    model.eval()
    X_tensor = torch.tensor(X_obs, dtype=torch.float32)
    theta_tensor = torch.ones(X_obs.shape[0], dtype=torch.float32) * theta_value
    with torch.no_grad():
        s = model(X_tensor, theta_tensor).squeeze()
        s = torch.clamp(s, -clip_value, clip_value)
    return s.sum().item()

# -------------------------------
# 6. Theta scan
# -------------------------------

def scan_theta(model, X_obs, theta_grid):
    logL_list = [compute_log_likelihood(model, X_obs, theta) for theta in theta_grid]
    logL_array = np.array(logL_list)
    theta_best = theta_grid[np.argmax(logL_array)]
    return theta_best, logL_array

# -------------------------------
# 7. Plot classifier output
# -------------------------------

# def plot_score_distribution(model, X_theta, X_ref, theta):
#     model.eval()
#     with torch.no_grad():
        
#         X_theta_t = torch.as_tensor(X_theta, dtype=torch.float32)
#         theta_t = torch.full((len(X_theta),), theta, dtype=torch.float32)
#         s_theta = model(X_theta_t, theta_t).squeeze().cpu().numpy()
#         X_ref_t = torch.as_tensor(X_ref, dtype=torch.float32)
#         theta_ref = torch.zeros(len(X_ref), dtype=torch.float32)
#         s_ref = model(X_ref_t, theta_ref).squeeze().cpu().numpy()

#     plt.figure()
#     plt.hist(s_theta, bins=50, alpha=0.5, label='theta sample', density=True)
#     plt.hist(s_ref, bins=50, alpha=0.5, label='SM reference', density=True)
#     plt.xlabel('s(x, theta)')
#     plt.ylabel('Probability Density')
#     plt.legend()
#     plt.title('Classifier output distribution')
#     plt.savefig("classifier_distribution.pdf")

def plot_score_distribution(model, X_theta, X_ref, theta):
    model.eval()
    with torch.no_grad():
        X_theta_t = torch.as_tensor(X_theta, dtype=torch.float32)
        theta_t = torch.full((len(X_theta),), theta, dtype=torch.float32)
        s_theta = model(X_theta_t, theta_t).squeeze().cpu().numpy()
        
        X_ref_t = torch.as_tensor(X_ref, dtype=torch.float32)
        theta_ref = torch.zeros(len(X_ref), dtype=torch.float32)
        s_ref = model(X_ref_t, theta_ref).squeeze().cpu().numpy()

    s_theta = s_theta[np.isfinite(s_theta)]
    s_ref = s_ref[np.isfinite(s_ref)]
    
    if len(s_theta) == 0 or len(s_ref) == 0:
        print("Warning: No valid data to plot! (All NaN or Inf)")
        return

    plt.figure(figsize=(8, 6))
    
    all_data = np.concatenate([s_theta, s_ref])
    plot_range = (np.percentile(all_data, 1), np.percentile(all_data, 99))

    plt.hist(s_theta, bins=50, range=plot_range, alpha=0.5, label=f'theta={theta}', density=True)
    plt.hist(s_ref, bins=50, range=plot_range, alpha=0.5, label='SM reference', density=True)
    
    plt.xlabel('s(x, theta)')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.title('Classifier output distribution')
    
    save_path = "classifier_distribution.pdf"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}. Data range: {plot_range}")





    

# ===============================================
# Example usage
# ===============================================

if __name__ == "__main__":
    files = ["/eos/user/y/yzhang4/LIV/LO/MG_LHE_ppZZto4L_LO_theta_1e-3/result/total.root",
             "/eos/user/y/yzhang4/LIV/LO/MG_LHE_ppZZto4L_LO_theta_1e-4/result/total.root",
             "/eos/user/y/yzhang4/LIV/LO/MG_LHE_ppZZto4L_LO_theta_1e-5/result/total.root",
             "/eos/user/y/yzhang4/LIV/LO/MG_LHE_ppZZto4L_LO_theta_1e-6/result/total.root"]
    theta_values = [1e-3, 1e-4, 1e-5,1e-6]

    X_ref = load_root_features("/eos/user/y/yzhang4/LIV/LO/MG_LHE_ppZZto4L_LO_theta_1e-0/result/total.root")

    datasets = []
    for f, theta in zip(files, theta_values):
        X_theta = load_root_features(f)
        ds = prepare_dataset(X_theta, X_ref, theta_value=theta)
        datasets.append(ds)
    dataset = ConcatDataset(datasets)

    input_dim = X_ref.shape[1] + 1
    model = LikelihoodRatioNet(input_dim)

    model = train_network(model, dataset, epochs=20, batch_size=256, lr=1e-3)
    plot_score_distribution(model, X_theta, X_ref, theta=1e-4)

    # here need to put the observed data file path
    X_obs = load_root_features("/eos/user/y/yzhang4/LIV/LO/MG_LHE_ppZZto4L_LO_theta_5e-5/result/total.root")
    theta_grid = np.geomspace(5e-6, 1e-3, 400)
    theta_best, logL_array = scan_theta(model, X_obs, theta_grid)

    print(f"Best-fit theta = {theta_best}")

    delta_logL = logL_array - np.max(logL_array)
    plt.figure()
    plt.plot(theta_grid, -2 * delta_logL, marker='o')
    plt.xscale('symlog', linthresh=1e-7)
    plt.xlim(1e-7, 1e-3)
    plt.ylim(-1, 20)
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$-2\Delta \log \mathcal{L}$')
    plt.title('Likelihood scan')
    plt.axhline(1.0, linestyle='--', color='red', label='1σ')
    plt.axhline(4.0, linestyle='--', color='blue', label='2σ')
    plt.grid()
    plt.savefig("Likelihood scan.pdf")
