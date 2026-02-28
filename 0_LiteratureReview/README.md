# Literature Review
## 1. Introduction
Recent leaps in AI-driven weather prediction have spurred similar machine learning applications across broader Earth system modeling. Because conventional numerical ocean simulations require massive computational resources, data-driven emulators for specific regions offer a cost-effective alternative. These models can supplement physical observation networks and enable the affordable generation of forecast ensembles at a fraction of the cost. However, predicting variables like Sea Surface Temperature (SST) remains an intricate spatiotemporal problem. While early emulators often rely on standard 2D Convolutional Neural Networks (CNNs), purely spatial architectures can struggle to track sequential changes over time. This literature review assesses the specific requirements of regional ocean modeling and highlights the structural network improvements necessary for accurate spatiotemporal predictions.

## 2. Domain Context and Data Foundations
**Global versus Regional Emulation**
Existing global AI models (e.g., Dheeshjith et al. 2025, El Aouni et al. 2025) have demonstrated viability for long-term climate analysis. Nevertheless, their restricted spatial fidelity often prevents the examination of finer, localized ocean behaviors. Shifting to a regional focus solves this computational bottleneck, permitting higher grid resolutions within a constrained area without demanding exponentially more processing power.

**Data Requirements for Training**
Training robust regional emulators requires decades of high-quality data. For this project, we rely on the GLORYS12V1 dataset, an 8 km-resolution global ocean reanalysis. Powered by the NEMO model and ECMWF atmospheric forcing, this dataset goes beyond pure simulation by assimilating real-world satellite and in-situ observations. This gives us the reliable, daily ocean states necessary to train neural networks on complex localized dynamics.

**Physical Dynamics and Phased Approach**
The oceanography around the Cape Verde Archipelago is highly dynamic, driven by strong eddy activity and complex upwelling systems (Schütte et al. 2016, Schütte et al. 2025). This variability makes SST forecasting a difficult but rewarding spatiotemporal challenge.  To manage this complexity, we are building the project in phases. This first phase utilizes a simple 2D U-Net trained solely on surface fields to establish a baseline. This step isolates the sequential predictive capabilities of the architecture, providing a benchmark before we incrementally add physics-informed loss terms, additional variables like ocean velocities, and more complex network layers in future iterations.

## 3. Advanced Spatiotemporal Architectures
The following models represent the current state-of-the-art in ocean emulation, each offering specific architectural insights for our development roadmap.

**Samudra: An AI Global Ocean Emulator for Climate**
*Dheeshjith et al. (2025)* built a stable autoregressive 3D ocean emulator designed to predict full-depth variables over centuries. They utilized a modified ConvNeXt U-Net architecture, altering standard blocks by reducing kernel sizes to 3x3 and using batch normalization to boost predictive skill. Notably, instead of using volumetric 3D convolutions, they processed 3D spatial data across 19 depth levels by mapping them directly to distinct input/output channels—essentially collapsing the vertical dimension into the channel dimension. The model operates recursively, taking two consecutive past states plus atmospheric boundary conditions to predict future states. While it runs 150 times faster than numerical models and maintains long-term stability, it still struggles to capture the full magnitude of extreme climate-warming trends.

**GLONET: Mercator's End-to-End Neural Global Ocean Forecasting System**
*El Aouni et al. (2025)* introduced GLONET, a system that predicts 3D ocean states up to ten days in advance in mere seconds. To handle the different scales of ocean dynamics, it employs a dual-branch spatial architecture. One branch uses Fourier Neural Operators (FNO) to capture broad, basin-wide circulations, while a parallel CNN branch resolves localized features like submesoscale eddies. An encoder-decoder network then fuses these representations into a unified latent space, ensuring large- and small-scale physics interact coherently. To ensure stability, GLONET uses an autoregressive pipeline where the model is forced to predict multiple days ahead during training. Although it preserves physical consistencies like energy cascades, it slightly underperforms traditional models in predicting SST due to a lack of real-time observational data assimilation.

**ST-UNet: Spatiotemporal Sea Surface Temperature Prediction**
*Ren et al. (2024)* developed ST-UNet to predict daily SST at lead times of up to 7 days. Structurally, the network is built on a 3D U-Net framework but incorporates significant temporal upgrades.  In the encoding phase, a "fusion block" integrates ConvLSTM layers to capture long-term dependencies within the 35-day input time series. To extract spatial features of varying sizes, the network employs parallel convolutional branches with different kernel sizes. At the bottleneck, an Atrous Spatial Pyramid Pooling (ASPP) module uses dilated convolutions to capture broad spatial context without increasing the parameter count. While highly effective, the model still faces challenges in accurately predicting sharp SST shifts during extreme weather events like tropical cyclones.

## 4. Strategic Synthesis and Roadmap
Reviewing the current literature reveals a pragmatic developmental roadmap for ocean emulators. Starting with a standard 2D U-Net as a transparent baseline provides a clear understanding of the model's sequential mapping capabilities—the focus for this year's project. This baseline allows us to isolate the model's performance before moving toward the systematic upgrades suggested by the literature.

Future iterations will involve transitioning to advanced convolutional architectures designed to expand the model's receptive field and capture broader spatial contexts. Subsequently, introducing attention mechanisms at the network's bottleneck will allow the model to recognize wider spatial dependencies that purely local convolutions might miss. Concurrently, our flexible data pipeline is designed to incrementally ingest additional kinematic and physical drivers, shifting the network from projecting smoothed historical trends toward predicting dynamically forced anomalies. This phased approach ensures that every technical upgrade is measurable and actively drives the model toward higher physical accuracy.

## 5. Sources
*1. El Aouni, A. et al. GLONET: Mercator’s End-to-End Neural Global Ocean Forecasting System. Journal of Geophysical Research: Machine Learning and Computation 2, e2025JH000686 (2025).*

*2. Dheeshjith, S. et al. Samudra: An AI Global Ocean Emulator for Climate. Geophysical Research Letters 52, e2024GL114318 (2025).*

*3. Schütte, F., Brandt, P. & Karstensen, J. Occurrence and characteristics of mesoscale eddies in the tropical northeastern Atlantic Ocean. Ocean Science 12, 663–685 (2016).*

*4. Schütte, F. et al. Linking physical processes to biological responses: Interdisciplinary observational insights into the enhanced biological productivity of the Cape Verde Archipelago. Progress in Oceanography 235, 103479 (2025).*

*5. Ren, J. et al. Prediction of Sea Surface Temperature Using U-Net Based Model. Remote Sensing 16, (2024).*
