# Week 1 - PneumoniaMNIST

model: resnet18

epochs: 5

batch size: 128

## Issues
Code execution was successful with no issues during setup, training and testing.

---

**Notes:** Model can distinguish positive cases from negative one relatively well (auroc ~ 0.83) and has a somewhat good but not perfect accuracy score (~ 0.75). Model confidence and accuracy do not match too well (ece = 0.239). The imbalance of the dataset might explain the high ece and accuracy. Temperature scaling or class weighting could help improve ece score without impacting accuracy significantly.

