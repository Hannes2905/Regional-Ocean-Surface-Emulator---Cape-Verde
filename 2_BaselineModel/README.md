# Baseline Model

**[Notebook](baseline_model.ipynb)**

## Baseline Model Results

### Model Selection
- **Baseline Model Type:** Persistence (Naive) Forecast.
- **Rationale:** In geophysical fluid dynamics, persistence is the standard "no-skill" benchmark. It assumes that the ocean state at the next time step is identical to the current state ($\hat{\theta}_{o}(t + \Delta t) = \theta_{o}(t)$). Because Sea Surface Temperature (SST) exhibits high thermal inertia and strong seasonal autocorrelation, this provides a rigorous baseline that any deep learning architecture must outperform to prove it has learned advective physics.

### Model Performance
- **Evaluation Metric:** Mean Squared Error (MSE), Root Mean Square Error (RMSE).
- **Performance Score:** - **Mean MSE:** [0.14590] $^\circ\text{C}^2$
  - **Mean RMSE:** [0.3698] $^\circ\text{C}$

### Evaluation Methodology
- **Data Split:** The baseline is evaluated on the final 20% of the temporal dataset (the "Test Set"). Covering the most recent years (approximately 2021–2025).
- **Evaluation Metrics:** - **MSE:** Chosen as the primary optimization metric because its quadratic penalty ensures the model is sensitive to large thermal anomalies and sharp gradients.
  - **RMSE:** Used for physical interpretability, as it maintains the same units ($^\circ\text{C}$) as the potential temperature ($\theta_o$) target variable.

### Metric Practical Relevance
These metrics measure how accurately a model predicts temporal changes in the ocean's thermal state.

- **MSE / RMSE Impact:** These calculate the exact pixel-by-pixel temperature differences. A high error in the persistence baseline highlights regions where the ocean changed the most over a week, typically where currents physically moved water masses (e.g., along the coastal upwelling front).
- **Proving Model Skill:** Outperforming the persistence RMSE is the basic requirement to prove a model is actually learning physical dynamics (like advection) rather than simply memorizing the previous week's state.


## Next Steps
This baseline model serves as a reference point for evaluating more sophisticated models in the [Model Definition and Evaluation](../3_Model/README.md) phase. Specifically, future iterations aim to:
1. **Reduce Prediction Lag:** Outperform the persistence RMSE during high-dynamic transition seasons (Spring/Autumn).
2. **Integrate Advection:** Leverage $u_o$ and $v_o$ velocity vectors to predict the spatial displacement of thermal features that persistence inherently misses.
3. **Boundary Refinement:** Improve predictive performance near the African coastline where the land-mask creates complex boundary conditions.