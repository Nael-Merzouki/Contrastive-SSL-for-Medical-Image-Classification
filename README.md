# Personal Contributions — MedMNIST SSL

This repository contains my personal contributions (scripts, results, and reports) to the MedMNIST-SSL project proposed during a McGill AI Lab meeting. It documents the work I conducted up to Week 4 of this 8-week project, focusing on benchmarking architectures, improving evaluation metrics, and analyzing model performance for the medical imaging classification task.

For full project details, including overall structure and team contributions, refer to the main repository: [dk1028/medmnist-ssl](https://github.com/dk1028/medmnist-ssl).

## Project Questions/Goals
The project aimed to:
- Evaluate the performance of different convolutional neural network (CNN) architectures on MedMNIST benchmarks.
- Address calibration and overconfidence issues in medical image classification.
- Explore the effects of data augmentation strategies on model robustness and generalization.

## Phase (Week 4 Status)
This repository reflects intermediate results from Week 4, incorporating both initial comparisons of models and early-stage findings on augmentation and calibration methods.

## Key Contributions and Results
### Contributions:
- Implemented and benchmarked multiple CNN architectures (e.g., ResNet-18, custom small CNNs) to identify top-performing models with respect to classification accuracy, AUROC, and calibration metrics.
- Developed data loading, preprocessing, and evaluation scripts that standardized experiment protocols and ensured reproducibility.
- Explored finetuning effectiveness by comparing "head-only" and "end-to-end" training strategies, revealing substantial performance gains in full finetuning.

### Key Findings:
#### Week 1:
- Successfully trained ResNet-18 on PneumoniaMNIST, achieving **AUROC: 0.83** and **Test Accuracy: 0.75**. Calibration, however, was suboptimal (**ECE: 0.239**) due to class imbalance and overconfidence predictions.
- Suggested using temperature scaling to address calibration issues.

#### Week 2:
- Benchmarked ResNet-18 and small CNN models, with ResNet-18 consistently outperforming small CNNs on all metrics (**Test Accuracy: 0.76**, **AUROC: 0.84**, **ECE: 0.25**). 
- Identified severe overconfidence in predictions and associated calibration challenges.
- Highlighted dataset imbalance and model limitations as constraints on performance.

#### Week 3:
- Demonstrated significant improvements with "end-to-end" finetuning: ResNet-18 achieved **Test Accuracy: 0.87**, **AUROC: 0.95**, maintaining its leading performance among tested architectures.
- Conducted augmentation ablation studies, finding that augmentation strategies (e.g., random horizontal flips, rotation, normalization) boosted robustness, with the best test metrics observed using combined augmentations (**Test Accuracy: 0.90**, **AUROC: 0.98**).

Full weekly reports and detailed metrics can be found under `results/weekx`.

## Repository Contents
- **Scripts:** Data preprocessing, model definitions, training loops, and metrics evaluations.
- **Reports:** Weekly summaries of experiments and findings with tables, visualizations, and ablation studies.
- **Notebooks:** Reproducible workflows for training and evaluation.
- **Config files:** Dependency specifications for easy experiment replication.

---

For additional information and full project context, visit the [main repository](https://github.com/dk1028/medmnist-ssl).
```
