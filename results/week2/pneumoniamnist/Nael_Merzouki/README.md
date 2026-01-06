# Week 2 Report — PneumoniaMNIST (resnet18--head, Adam, 8 epochs)

## 1. Dataset Recap

- **Class balance:** PneumoniaMNIST is designed to be balanced between the “pneumonia” and “normal” classes, with a very similar number of examples for each in train, validation, and test splits.
- **Artifacts/quirks:** Images are greyscale, consistently sized at 32x32, and present subtle differences between disease and healthy scans. A close inspection reveals low image resolution and possible preprocessing artifacts (strong blurring/smoothing, visible pixelation).
- **Class separability:** Some images of normal and pneumonia classes appear visually similar, with subtle differences that may be hard to discern either for humans or models, especially in low-contrast cases.
- No obvious text overlays or padding, but some images present strong intensity gradients or are overall much darker/lighter than others, hinting at possible heterogeneity due to acquisition or preprocessing.

## 2. Baseline Comparison

I compared two baseline models (from Week 1 and Week 2 runs):

| Model           | Val Acc | Test Acc | Test AUROC |
| --------------- | ------: | -------: | ---------: |
| smallcnn        |  0.62   |   0.62   |    0.72    |
| resnet18 (head) |  0.76   |   0.76   |    0.84    |

- The **resnet18 (head)** model substantially outperformed smallcnn on both accuracy and AUROC, indicating a much stronger capacity to discriminate between pneumonia and normal cases.
- In the learning curves, both models plateaued quickly, but smallcnn showed clear underfitting (low capacity, lower scores), while resnet18 converged to a higher baseline but did not overfit with just 8 epochs and a frozen backbone.

## 3. Calibration Snapshot

- With resnet18 (head) + Adam, the reliability diagram (see below) indicated some **overconfidence**: several bins near high confidence had lower accuracy than predicted probability, contributing to a relatively high ECE (e.g., ECE ≈ 0.25).
- The model confidently makes mistakes; in some bins, predicted confidence is nearly 1, but actual accuracy is much lower.
- The equal-frequency reliability diagram (see figure) clearly shows the model's tendency to assign very high probabilities even when it does not always get the label correct.
- I did **not** attempt temperature scaling or other explicit post-hoc calibration this week, but plan to do so in Week 5/6.

![image1](image1)

## 4. Error Analysis

- Many of the model's confident mistakes come from **predicting "pneumonia" on images that look visually close to normal**, possibly because of subtle features or artifacts that mislead the model.
- **Most errors are associated with very high confidence:** e.g., probability 0.99 or 1.0, even when the prediction is incorrect. This is visible in both the reliability curve and the misclassification gallery below.
- In most misclassified images, **contrast is low and structures are not clearly defined**, hinting that edge/boundary information is lost in downsampling or preprocessing.
- Some normal x-rays marked as pneumonia could be due to **borderline presentation or atypical cases**, or potential label noise in the dataset.
- The model appears to latch onto **subtle, possibly spurious intensity patterns** as pneumonia indicators in normal cases.

![image2](image2)

---

*Summary:* This week’s baseline and calibration analysis revealed that while resnet18 is a strong baseline for classifying PneumoniaMNIST, it is notably overconfident, and some mistakes are due to visually ambiguous or low-contrast images. Further work might include exploring regularization, augmentation, and temperature scaling to reduce overconfidence.