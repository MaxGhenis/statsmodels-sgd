"""Logit model with differential privacy support."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import statsmodels.api as sm
import pandas as pd
from ..privacy.mechanisms import clip_gradient, add_noise_to_gradient
from ..privacy.accounting import PrivacyAccountant
from ..tools import calculate_standard_errors, calculate_t_p_values


class LogitDP(nn.Module):
    """Logistic regression with differential privacy via SGD and gradient clipping."""
    
    def __init__(
        self,
        n_features,
        learning_rate=0.01,
        epochs=500,
        batch_size=32,
        clip_value=1.0,
        noise_multiplier=0.0,
        track_privacy=False,
        target_delta=1e-5,
    ):
        super().__init__()
        self.n_features = n_features
        self.linear = nn.Linear(n_features, 1, bias=False)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.clip_value = clip_value
        self.noise_multiplier = noise_multiplier
        self.track_privacy = track_privacy
        self.target_delta = target_delta
        self.results_ = None
        self.add_constant = True
        self.privacy_accountant = None
        self.n_samples_ = None

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def fit(self, X, y, sample_weight=None, add_constant=True):
        self.add_constant = add_constant
        if add_constant:
            X = sm.add_constant(X)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        
        self.n_samples_ = X.shape[0]

        if sample_weight is not None:
            sample_weight = torch.tensor(
                sample_weight, dtype=torch.float32
            ).reshape(-1, 1)

        # Initialize weights
        self.linear = nn.Linear(X.shape[1], 1, bias=False)
        init_weights = torch.zeros(1, X.shape[1])
        self.linear.weight = nn.Parameter(init_weights)

        # Setup privacy accounting
        if self.track_privacy:
            self.privacy_accountant = PrivacyAccountant(
                mechanism="gaussian",
                delta_target=self.target_delta,
            )
            coef_history = []

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=50
        )

        prev_loss = float("inf")
        patience_counter = 0
        best_weights = None
        min_loss = float("inf")

        # Training loop with batching
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        if sample_weight is not None:
            dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor, sample_weight)
        
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        for epoch in range(self.epochs):
            epoch_loss = 0
            n_batches = 0
            
            for batch in dataloader:
                if sample_weight is not None:
                    X_batch, y_batch, weight_batch = batch
                else:
                    X_batch, y_batch = batch
                    weight_batch = None
                
                optimizer.zero_grad()
                y_pred = self(X_batch)
                loss = self.weighted_binary_cross_entropy(y_pred, y_batch, weight_batch)
                
                loss.backward()
                
                # Apply DP mechanisms
                if self.noise_multiplier > 0:
                    # Clip and add noise to gradients
                    for param in self.parameters():
                        if param.grad is not None:
                            clip_gradient(param, self.clip_value)
                            param.grad = add_noise_to_gradient(
                                param.grad,
                                self.noise_multiplier,
                                self.clip_value,
                                mechanism="gaussian"
                            )
                else:
                    # Just clip without noise
                    torch.nn.utils.clip_grad_value_(self.parameters(), self.clip_value)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
                
                if self.track_privacy and self.privacy_accountant:
                    sample_rate = self.batch_size / self.n_samples_
                    self.privacy_accountant.step(
                        noise_multiplier=self.noise_multiplier,
                        sample_rate=sample_rate,
                        sensitivity=self.clip_value
                    )
            
            avg_loss = epoch_loss / n_batches
            scheduler.step(avg_loss)
            
            # Track coefficient history for SE adjustment
            if self.track_privacy:
                current_coef = self.linear.weight.detach().numpy().flatten()
                coef_history.append(current_coef.copy())
            
            if avg_loss < min_loss:
                min_loss = avg_loss
                best_weights = self.linear.weight.clone().detach()
            
            if abs(avg_loss - prev_loss) < 1e-6:
                patience_counter += 1
                if patience_counter >= 50:
                    break
            else:
                patience_counter = 0
            prev_loss = avg_loss

        if best_weights is not None:
            self.linear.weight.data = best_weights

        # Calculate standard errors with DP adjustment
        params = self.linear.weight.detach().numpy().flatten()
        
        # Use full dataset for final SE calculation
        with torch.no_grad():
            y_pred_final = self(X_tensor).numpy()
        
        # Calculate base standard errors using Fisher information
        y_pred_np = y_pred_final.squeeze()
        V = np.diag(y_pred_np * (1 - y_pred_np))
        if sample_weight is not None:
            V = np.diag(sample_weight.numpy().squeeze()) @ V
        
        try:
            fisher_info = X.T @ V @ X
            var_covar_matrix = np.linalg.inv(fisher_info)
            base_std_errors = np.sqrt(np.diag(var_covar_matrix))
        except np.linalg.LinAlgError:
            # Fallback if matrix is singular
            base_std_errors = np.ones(len(params)) * 0.1
        
        # Adjust for DP noise
        if self.track_privacy and len(coef_history) > 1:
            coef_array = np.array(coef_history)
            empirical_var = np.var(coef_array, axis=0)
            
            # Similar calibration as OLS
            empirical_inflation = np.sqrt(1 + 0.6 * empirical_var / (base_std_errors ** 2 + 1e-10))
            theoretical_inflation = 1 + 0.2 * self.noise_multiplier * np.sqrt(self.epochs / self.n_samples_)
            noise_inflation_factor = 0.7 * empirical_inflation + 0.3 * theoretical_inflation
            
            std_errors = base_std_errors * noise_inflation_factor
        else:
            # Without DP tracking, use base SEs
            std_errors = base_std_errors
        
        # Calculate test statistics
        z_values = params / std_errors
        p_values = 2 * (1 - np.abs(np.vectorize(lambda z: 
            0.5 * (1 + np.sign(z) * (1 - np.exp(-2 * z**2 / np.pi)**0.5)))(z_values)))
        
        self.results_ = {
            "coef": params,
            "std_errors": std_errors,
            "z_values": z_values,
            "p_values": p_values,
            "coef_history": coef_history if self.track_privacy else None
        }

    def weighted_binary_cross_entropy(self, y_pred, y_true, weights=None):
        eps = 1e-7
        y_pred = torch.clamp(y_pred, eps, 1 - eps)
        if weights is None:
            weights = torch.ones_like(y_true)
        loss = -(
            weights
            * (
                y_true * torch.log(y_pred)
                + (1 - y_true) * torch.log(1 - y_pred)
            )
        ).mean()
        return loss

    def predict(self, X):
        if self.add_constant:
            X = sm.add_constant(X)
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self(X).numpy()
    
    def predict_proba(self, X):
        """Return probability estimates."""
        probs = self.predict(X)
        return np.column_stack([1 - probs, probs])

    def summary(self):
        if self.results_ is None:
            raise ValueError("Model has not been fit yet.")
        
        # Create summary dict compatible with OLS
        summary_dict = {
            'coefficients': self.results_['coef'],
            'std_errors': self.results_['std_errors'],
            'z_values': self.results_['z_values'],
            'p_values': self.results_['p_values'],
        }
        
        # Add privacy info if tracked
        if self.track_privacy and self.privacy_accountant:
            privacy = self.get_privacy_guarantee()
            summary_dict['privacy_epsilon'] = privacy['epsilon']
            summary_dict['privacy_delta'] = privacy['delta']
        
        # Also return as DataFrame for compatibility
        df = pd.DataFrame(
            {
                "coef": self.results_["coef"],
                "std err": self.results_["std_errors"],
                "z": self.results_["z_values"],
                "P>|z|": self.results_["p_values"],
                "[0.025": self.results_["coef"] - 1.96 * self.results_["std_errors"],
                "0.975]": self.results_["coef"] + 1.96 * self.results_["std_errors"],
            },
            index=["const"]
            + [f"x{i}" for i in range(1, len(self.results_["coef"]))],
        )
        
        # Store both formats
        summary_dict['dataframe'] = df
        
        return summary_dict
    
    def get_privacy_guarantee(self):
        """Get the current privacy guarantee (epsilon, delta)."""
        if self.privacy_accountant is None:
            return {"epsilon": float("inf"), "delta": 0.0}
        
        epsilon = self.privacy_accountant.get_privacy_spent(delta=self.target_delta)
        return {"epsilon": epsilon, "delta": self.target_delta}