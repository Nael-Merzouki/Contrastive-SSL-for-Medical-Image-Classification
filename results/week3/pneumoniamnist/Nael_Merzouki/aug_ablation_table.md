| Model / Variant            | Finetune | Train Aug   | Test Acc | Test AUROC | Notes               |
|--------------------------- |----------|------------ |---------:|----------: |---------------------|
| ResNet-18 (head-only)      | head     | basic       |   0.76   |    0.84    | Week 2 baseline     |
| ResNet-18 (all, basic)     | all      | basic       |   0.87   |    0.95    | mild overfitting    |
| ResNet-18 (all, AugA)      | all      | flip+rot    |   0.87   |    0.97    | better test auroc     |
| ResNet-18 (all, AugD)      | all      | flip+rot+jit+norm|   0.9   |    0.98    | best test metrics, more noise |