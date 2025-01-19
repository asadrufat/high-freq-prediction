from typing import Any, Dict, Tuple

import numpy as np
import pymc3 as pm

from ..config import Config


class MCMCOptimizer:
    """MCMC-based optimizer for prediction refinement."""

    def __init__(self, config: Config):
        """
        Initialize MCMC optimizer.

        Args:
            config: Configuration object containing MCMC parameters
        """
        self.config = config
        self.model = None
        self.trace = None
        self.scaling_params = None

    def _prepare_data(
        self, multipliers: np.ndarray, index: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Prepare and scale data for MCMC.

        Args:
            multipliers: Historical multiplier values
            index: Time index for the multipliers

        Returns:
            Tuple of (scaled_multipliers, scaled_index, scaling_params)
        """
        # Remove any missing values
        mask = ~np.isnan(multipliers)
        clean_multipliers = multipliers[mask]
        clean_index = index[mask]

        # Scale data for numerical stability
        scaling_params = {
            "index_mean": clean_index.mean(),
            "index_std": clean_index.std(),
            "mult_mean": clean_multipliers.mean(),
            "mult_std": clean_multipliers.std(),
        }

        scaled_index = (clean_index - scaling_params["index_mean"]) / scaling_params[
            "index_std"
        ]
        scaled_multipliers = (
            clean_multipliers - scaling_params["mult_mean"]
        ) / scaling_params["mult_std"]

        return scaled_multipliers, scaled_index, scaling_params

    def fit_predict(
        self, multipliers: np.ndarray, index: np.ndarray, future_steps: int = 48
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Fit MCMC model and generate future multipliers with uncertainty estimates.

        Args:
            multipliers: Historical multiplier values
            index: Time index for the multipliers
            future_steps: Number of future steps to predict

        Returns:
            Tuple of (predicted_multipliers, model_diagnostics)
        """
        # Prepare data
        scaled_multipliers, scaled_index, scaling_params = self._prepare_data(
            multipliers, index
        )
        self.scaling_params = scaling_params

        # Define and fit MCMC model
        with pm.Model() as model:
            # Priors - using weakly informative priors
            alpha = pm.Normal("alpha", mu=0, sd=10)
            beta = pm.Normal("beta", mu=0, sd=10)
            sigma = pm.HalfCauchy("sigma", beta=5)

            # Add autoregressive component
            phi = pm.Uniform("phi", lower=0, upper=1)  # AR(1) coefficient

            # Model with AR(1) component
            mu = alpha + beta * scaled_index
            ar1_like = pm.AR1(
                "ar1_like", phi=phi, sigma=sigma, mu=mu, observed=scaled_multipliers
            )

            # Sample using NUTS sampler
            trace = pm.sample(
                draws=self.config.model.mcmc_samples,
                tune=self.config.model.mcmc_tune,
                chains=4,
                cores=4,
                target_accept=0.95,
                return_inferencedata=True,
                nuts_sampler="auto",
            )

        self.model = model
        self.trace = trace

        # Generate future predictions
        future_x = np.arange(index[-1] + 1, index[-1] + future_steps + 1)
        scaled_future_x = (future_x - scaling_params["index_mean"]) / scaling_params[
            "index_std"
        ]

        with model:
            pm.set_data({"ar1_like": scaled_multipliers})

            # Generate posterior predictive samples
            posterior_pred = pm.sample_posterior_predictive(
                trace, var_names=["ar1_like"], keep_size=True
            )

        # Calculate predictions and intervals
        predictions = posterior_pred["ar1_like"].mean(axis=0)
        predictions = (
            predictions * scaling_params["mult_std"] + scaling_params["mult_mean"]
        )

        credible_intervals = {
            "lower": np.percentile(posterior_pred["ar1_like"], 2.5, axis=0)
            * scaling_params["mult_std"]
            + scaling_params["mult_mean"],
            "upper": np.percentile(posterior_pred["ar1_like"], 97.5, axis=0)
            * scaling_params["mult_std"]
            + scaling_params["mult_mean"],
        }

        # Get model diagnostics
        summary = pm.summary(trace, var_names=["alpha", "beta", "sigma", "phi"])

        return predictions, {
            "trace": trace,
            "summary": summary,
            "credible_intervals": credible_intervals,
            "diagnostics": {
                "r_hat": summary["r_hat"].values,
                "ess": summary["ess_bulk"].values,
                "mcse": summary["mcse_mean"].values,
            },
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get detailed MCMC diagnostics.

        Returns:
            Dictionary containing diagnostic plots and statistics
        """
        if self.trace is None:
            raise ValueError("Model hasn't been fit yet. Call fit_predict first.")

        return {
            "energy_plot": pm.plots.energy(self.trace),
            "trace_plot": pm.plots.trace(self.trace),
            "forest_plot": pm.plots.forest(self.trace),
            "autocorr_plot": pm.plots.autocorrplot(self.trace),
            "summary": pm.summary(self.trace),
            "divergences": self.trace.sampler_stats.diverging.sum(),
        }
