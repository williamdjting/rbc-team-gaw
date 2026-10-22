# Week 1

**Date:** October 15, 2026

## Topics

- Take one path of the pipeline to begin:
	- Machine model > ML ops > web UI
	- Contained project
	- ML stack

## Actions

- Focus on one path, specifically the ML model.
- Assign responsibilities across the team and define initial tasks.
- Share files with mentors.
- Find wildfire and weather datasets and check their quality.
- Research how wildfires occur.

## Deliverable

### Wildfire science and machine-learning background

#### 1. Why wildfires occur

Combustion requires fuel, heat, and oxygen. Lightning or human activity can provide the heat that ignites vegetation; whether a small ignition becomes a wildfire depends on the fuel and surrounding conditions. The amount, arrangement, and moisture of vegetation determine how readily fire starts and spreads; hot, dry, and windy weather dries fuels and can make fire growth faster and more intense. Slope and terrain also affect how quickly heat preheats fuels ahead of a fire. These factors interact: dry weather raises the chance that an ignition becomes a fire, but does not itself cause an ignition.

Climate change can make fire-conducive conditions more frequent or extend fire seasons in some regions. It is one influence on fire-weather and fuel conditions, not a complete explanation for any individual fire. The Canadian Forest Fire Weather Index (FWI) System summarizes weather-related fuel dryness and potential fire behaviour. Its components are useful indicators of burning conditions, but an FWI value is not a probability that a fire will ignite.

#### 2. Scientific basis of existing prevention programs

Prevention programs reduce either the chance of an ignition or the consequences if a fire starts. In B.C., the BC Wildfire Service describes prevention as including [open-fire restrictions](https://www2.gov.bc.ca/gov/content/safety/wildfire-status/prevention/fire-bans-and-restrictions), [fuel management](https://www2.gov.bc.ca/gov/content/safety/wildfire-status/prevention/fire-fuel-management), [cultural and prescribed fire](https://www2.gov.bc.ca/gov/content/safety/wildfire-status/prevention/prescribed-burning), and FireSmart practices.

- **Reduce human-caused ignitions:** bans, seasonal restrictions, safe-burning rules, and public education limit risky activities when fuels and weather are conducive to fire. Targeting these measures to local ignition causes and current conditions makes them more useful.
- **Change fuels and fire behaviour:** thinning, removing hazardous surface fuels, and creating strategically placed fuel breaks aim to reduce fuel amount or continuity. Carefully planned prescribed and cultural burning can consume some fuels under controlled conditions and restore fire as an ecological process. The expected benefit is lower intensity or easier-to-manage fire in treated areas—not a guarantee that a wildfire will stop there.
- **Protect homes and communities:** FireSmart actions such as managing vegetation and combustible materials near structures address how embers, radiant heat, and flames can ignite buildings. This reduces potential damage even when it cannot prevent a landscape fire.
- **Use fire-weather information for decisions:** the [Canadian Forest Fire Weather Index System](https://cwfis.cfs.nrcan.gc.ca/background/summary/fwi) provides indicators of fuel moisture and potential fire behaviour. Agencies can use fire weather, forecasts, and local conditions to time restrictions, prescribed burns, and readiness. These indicators inform prevention and preparedness; they do not identify the exact place or time of a future ignition.

These approaches are grounded in combustion and fire-behaviour science, but their effectiveness depends on ecosystem, weather, treatment design, and where ignitions occur. They reduce risk rather than eliminate wildfire. B.C. also notes that fire is a natural process in many ecosystems and that long-term suppression can allow fuels to build up; prevention therefore includes managing fire and fuels, not simply suppressing every fire.

#### 3. Machine-learning approaches in the four sources

The papers use satellite data in related but distinct ways. **Active-fire detection** asks where a fire is visible now; **early detection** tries to identify it sooner; **prediction or forecasting** estimates future hotspot imagery or the likelihood that an area will burn. Their performance figures are not directly comparable because their labels, sensors, regions, time horizons, and evaluation methods differ.

| Paper | Approach | What it shows |
| --- | --- | --- |
| [Zhao and Ban, “GOES-R Time Series for Early Detection of Wildfires with Deep GRU-Network” (*Remote Sensing*, 2022)](https://www.mdpi.com/2072-4292/14/17/4347) | A gated recurrent unit (GRU) learns from dense time series of GOES-R geostationary imagery to detect fires at an early stage. The study evaluates 36 fires and compares detection timing with NASA FIRMS VIIRS active-fire products. | The authors report earlier GOES-R/GRU detections than VIIRS in most study areas, with more precise early-stage locations than the GOES-R active-fire product in mid- and low-latitude regions. |
| [Saqer, “Deep Learning for Short-Term Prediction of Wildfire Using Geostationary Satellite Observations” (master’s thesis, 2024)](https://hdl.handle.net/10217/239802) | Uses GOES band 7 shortwave-infrared radiance, which responds to hotspots, as a time series. A Self-Attention Gated Recurrent Unit (SaGRU) predicts future image sequences from 25 minutes of input, using three fires for training and evaluating forecasts at 75, 100, and 125 minutes. | This is hotspot-image forecasting rather than simply classifying whether a fire starts. The thesis reports error metrics at different hotspot thresholds and finds that prediction error increases for later forecast frames. |
| [“Remote Sensing meets Deep Learning: Exploiting Spatio-Temporal-Spectral Satellite Images for Early Wildfire Detection” (EPFL publication record)](https://infoscience.epfl.ch/entities/publication/dcada51b-0c39-457c-9886-f938f2992e17) | A deep-learning system processes GOES-16 image streams and uses spatial, temporal, and spectral information to locate fires at pixel level, with outputs shown in an interactive dashboard. The record highlights challenges such as clouds and illumination. | The publication record reports a 94% F1 score and detections 1.5 times faster than its baselines. These are results for that study and test setup, not a general guarantee of operational accuracy. |
| [Porta et al., “CanadaFireSat: Towards High-Resolution Wildfire Forecasting with Multiple Modalities” (2026)](https://www.sciencedirect.com/science/article/pii/S0924271626002972) | Frames forecasting as binary classification of 100 m image patches using temporal Sentinel-2 imagery, MODIS products, and ERA5 environmental variables. It compares convolutional ResNet and Vision Transformer (ViT) models. | Multimodal temporal inputs outperform single-modality inputs in the reported experiments; the best reported F1 score is 60.3% for the unseen 2023 season. This predicts burned patches/occurrence at a forecast scale; it is not the same task as detecting an active fire pixel in a live image stream. |

Across the papers, common design choices are satellite sensors with different trade-offs, temporal sequences rather than isolated images, deep networks that learn spatial or temporal patterns, and comparison against active-fire products or other model baselines. Geostationary sensors provide frequent observations useful for timeliness but at coarser spatial resolution; higher-resolution or multisource data can add spatial detail and environmental context. Clouds, illumination, false alarms, label quality, and geographic transfer remain important limitations.

For this project, these studies provide context rather than a direct recipe: our initial model uses historical fire records and weather for one B.C. region, not satellite image sequences. We can use the papers to understand the value of temporal context, careful validation on unseen periods, and the difference between predicting fire occurrence and detecting an existing fire.