# STCRNet：A Semi-Supervised Network Based on Self-Training and Consistency Regularization for Change Detection in Remote Sensing Images

Welcome to the initial version of our project!

**Note:** This release represents an initial version of the project. The method's steps are not fully integrated yet, and automation is currently lacking. We plan to release a more automated version in the near future.

## Project Overview

This project aims to provide a solution for semi-supervised change detection tasks. The current code version can perform semi-supervised change detection tasks, but it might involve a bit more manual effort.

## Usage Steps

1. Run `train_URNet.py` to train the initial model F1 and generate model checkpoints. Select the parameters of the 30th, 40th, and 50th epochs as the checkpoints.
2. Execute `test.py` to generate pseudo-labels for unlabeled data. Use `Grad_CAM.py` to create class activation maps.
3. Utilize `MerIoU.py` to identify reliable unlabeled data.
4. Use `train_consitency.py` to train the re-trained model.
5. Evaluate the model's performance using `metric_total.py`.

## More results

| CDD Dataset | Precision (%) | Recall (%) | F1 Score (%) | IoU (%) |
|---------|--------------:|-----------:|-------------:|--------:|
| 5%      |          91.93|       82.73|         87.09|    77.13|
| 10%     |          92.87|       85.33|         88.94|    80.08|
| 20%     |          93.77|       87.00|         90.26|    82.25|
| 40%     |          94.15|       90.69|         92.39|    85.86|

## Acknowledgments

We express gratitude to the [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) project for enabling us to create CAM images. Special thanks to their team for their valuable contribution.

## Citation

This will be updated after the paper is published.
