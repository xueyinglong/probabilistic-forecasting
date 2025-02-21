# Top-down forecasting framework
This repository contains the implementations of the top-down forecasting framework proposed in https://doi.org/10.1016/j.ijpe.2024.109449.

## Global forecasting models with the aggregated series

Run the Rscript [experiments/top_down_experiments.R]().

Available model options: LightGBM, pooled regression, LASSO, generalised additive models (GAM)... Please refere to [utils/global_models.R]()

An implementation of the negative binomial loss function under the LightGBM package (Python) is in [experiments/negbin_loss.py](), with details in the Appendix of the [paper](https://doi.org/10.1016/j.ijpe.2024.109449).

## Forecasts at the lower (decision) level

Run [experiments/produce_probf_top_down.R]().

The point forecasts at the aggregated level are disaggregate down by historical proportions.

Probabilistic forecasts are produced based on assumptions, i.e., negative binomial distribution and Poisson distribution.

# Software/Package Versions

The versions of the software and packages that are used to conduct the experiments are mentioned in the following table.

| Software/Package        | Version        | 
|-------------------------|:--------------:|
| R                       |  4.1.2         |
| Python                  |  3.7.0         |
| LightGBM                |  3.3.2         |
| glmnet                  |  4.1.8         |
| tidyverse               |  1.3.2         |
| dplyr                   |  1.1.4         |
| GluonTS                 |  0.8.0         |

# Citing Our Work
When using this repository, please cite:

```{r} 
@article{long2025scalable,
  title={Scalable probabilistic forecasting in retail with gradient boosted trees: A practitionerï¿½s approach},
  author={Long, Xueying and Bui, Quang and Oktavian, Grady and Schmidt, Daniel F and Bergmeir, Christoph and Godahewa, Rakshitha and Lee, Seong Per and Zhao, Kaifeng and Condylis, Paul},
  journal={International Journal of Production Economics},
  volume={279},
  pages={109449},
  year={2025},
  publisher={Elsevier}
}
```
