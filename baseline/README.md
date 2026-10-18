# B0: DenseNet121 baseline for mpox skin lesion classification

A PyTorch reconstruction of the DenseNet121 baseline from Ali et al. (2024), trained on MSLD v2.0 with the dataset's five patient-independent folds. It is the first step (B0) of our project *Does It Still See Mpox? Stress Testing Skin Lesion Classifiers Under Attack, Corruption and Skin Tone Shift* (CMU-Africa, 11-785, Fall 2026).

> **Not the authors' code.** The paper does not release its implementation, so this is rebuilt from the paper's text. Every choice the paper leaves open is listed under [Deviations and assumptions](#deviations-and-assumptions).
> **Research use only.** This is not a medical device and must not be used for diagnosis.

## Results

Test results over the 5 official folds, mean ± sample standard deviation (n−1). The paper's row is the ImageNet-initialised DenseNet121 in its Table 2.

|  | Accuracy (%) | Precision | Sensitivity | F1 score |
|---|---|---|---|---|
| Ali et al. (2024), Table 2 | 81.70 ± 5.39 | 0.83 ± 0.04 | 0.79 ± 0.06 | 0.80 ± 0.06 |
| Ours, final model | 79.14 ± 3.99 | 0.82 ± 0.06 | 0.79 ± 0.03 | 0.79 ± 0.04 |
| Ours, best-val model | 81.91 ± 5.26 | 0.84 ± 0.07 | 0.79 ± 0.06 | 0.80 ± 0.06 |

- **Final model** is the weights after epoch 100. **Best-val model** is the epoch with the highest validation accuracy, chosen on the validation split only (the test split is never used for selection). The best-val epochs were 20, 96, 87, 28 and 54 for folds 1 to 5.
- Both rows are within one paper standard deviation of the paper's accuracy (81.70 ± 5.39). The final model is 2.56 points below the paper's mean. The paper does not say which checkpoint it reports, and each test fold has only 63 to 82 images, so one image moves a fold's accuracy by 1.2 to 1.6 points. Differences of this size are within noise.

Final model, per fold (`results/b0_results_final.csv`). Precision, sensitivity and F1 are macro-averaged over the 6 classes; mpox sensitivity is the recall of the mpox class alone.

|  | Accuracy (%) | Precision | Sensitivity | F1 | Mpox sensitivity |
|---|---|---|---|---|---|
| fold 1 | 79.730 | 0.841 | 0.771 | 0.784 | 0.815 |
| fold 2 | 75.610 | 0.755 | 0.783 | 0.755 | 0.677 |
| fold 3 | 78.125 | 0.819 | 0.763 | 0.780 | 0.682 |
| fold 4 | 76.543 | 0.783 | 0.804 | 0.781 | 0.593 |
| fold 5 | 85.714 | 0.900 | 0.834 | 0.856 | 0.833 |
| mean | 79.144 | 0.820 | 0.791 | 0.791 | 0.720 |
| std | 3.994 | 0.056 | 0.029 | 0.038 | 0.102 |

**Observations**
- Mpox sensitivity is the weakest number: 0.720 ± 0.102, and 0.593 on fold 4.
- The model overfits heavily. Training loss falls to about 0.005 while validation loss stays high (roughly 3.0 to 3.9 late in fold 5), yet validation accuracy stays near 79%.
- A 10-epoch test run scored 78.59 ± 4.54, so the extra 90 epochs added a net 2 correct test images out of 364.

## Data (not included)

MSLD v2.0 ([Kaggle](https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20) · [GitHub](https://github.com/mHealthBuet/Mpox-Skin-Lesion-Dataset-v2)). The images were scraped from the web and may be copyrighted (paper, Section 4.1), so **do not commit images or weights to this repository**; follow the dataset's own terms.

- The notebook uses the original images only, from `Original Images/Original Images/FOLDS/fold{1..5}/{Train,Valid,Test}/<class>`. It finds the `FOLDS` folder by itself under `/kaggle/input`.
- Every image has a byte-identical copy with an extra `_1` in its name (checked for all 3,775 copies). The notebook keeps only the originals, giving 755 images per fold, as in the paper.
- The folder splits are identical to the official `datalog.xlsx` for all 15 fold/split combinations (checked). For each fold the notebook also asserts that no image or patient appears in two splits.
- The shipped `Augmented Images` folder is **not used** (see Deviations).

## Method

| Stage | Setting |
|---|---|
| Backbone | torchvision DenseNet121, ImageNet weights (`IMAGENET1K_V1`), all layers trainable |
| Head | Flatten (50,176) → FC 4096 → FC 1072 → FC 256, each with ReLU and dropout 0.30 / 0.20 / 0.15 → Linear 6 |
| Parameters | 217,147,062 (checked by an assertion in the notebook) |
| Training | Adam (lr 1e-5, eps 1e-7), batch 16, 100 epochs, cross-entropy, fp16 mixed precision |
| Initialisation | Keras-style: Glorot-uniform weights and zero biases in the head, BatchNorm momentum 0.01 |
| Seeds | `42 + fold`; not bit-for-bit reproducible (GPU nondeterminism, data-loader workers) |
| Checkpoints | Resume file every 10 epochs; finished folds are saved to `fold{k}.json` and skipped on rerun |

**Augmentation (training images only, applied on the fly):** horizontal and vertical flips, affine (±20° rotation, 10% translation, 0.9 to 1.1 scale, 10° shear), colour jitter (brightness, contrast and saturation 0.2, hue 0.02), and Gaussian noise (std 0.02) on half the images. Each epoch repeats every training image `REPEAT = 14` times with fresh random augmentation, for example 537 × 14 = 7,518 samples and 470 steps for fold 1. Validation and test images are not augmented.

## Deviations and assumptions

1. **Epoch size.** The paper says augmentation made the training set about 14× larger (Section 4.3) but not how many images an epoch used. We assume a full pass over a 14× set and imitate it with `REPEAT = 14`. If the authors used fewer images per epoch, their 100 epochs were fewer steps than ours.
2. **On-the-fly augmentation** instead of the authors' pre-generated files, which were made once in MATLAB (per the dataset README). The paper gives the transform types but not their ranges, so ours are our own choices. We avoid the shipped folder to rule out leakage between augmented copies and test images.
3. **Head details the paper does not give:** ReLU as the activation and dropout after each FC block.
4. **Framework.** The paper used Keras. We use PyTorch with torchvision's ImageNet weights, which are not the same numbers as Keras's, plus the Keras-style defaults above.
5. **Duplicate `_1` files removed** (see Data).
6. **Checkpoint choice.** The paper does not say whether it reports the last or the best-validation epoch, so we report both.

## How to run (Kaggle)

1. Create a Kaggle notebook from `notebooks/b0_densenet121_baseline.ipynb`.
2. Add the dataset `joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20` (Add Input), enable a GPU, and enable Internet (needed once to download the ImageNet weights).
3. Optional logging: add a Kaggle secret named `WANDB_API_KEY`. If it is missing, logging switches itself off.
4. Set `QUICK = True` in the settings cell and run all cells once (1 fold, 2 short epochs) to check the pipeline. Then set it back to `False`.
5. Run the full job with **Save Version → Save & Run All**.

**Time and disk.** Our run took about 11.4 hours on a Kaggle T4 (about 83 s per epoch per fold, roughly 2.3 hours per fold), close to Kaggle's session limit. The notebook therefore supports splitting the run: run `FOLDS = [1, 2, 3]` first, then in a new version add the first run's output as an input, set `RESTORE_FROM` to its folder, set `FOLDS = [1, 2, 3, 4, 5]`, and the finished folds are skipped. With `SAVE_WEIGHTS = True`, each fold keeps two ~870 MB model files.

The notebook is written for Kaggle (`/kaggle/input`, `/kaggle/working`). Running elsewhere means changing the data-finding cell.

## Repository layout

```
notebooks/b0_densenet121_baseline.ipynb   training and evaluation (no saved outputs)
results/b0_results_final.csv              per-fold, mean and std, final model
results/b0_vs_paper.csv                   comparison with the paper
requirements.txt
```

## Reference

S. N. Ali, M. T. Ahmed, T. Jahan, J. Paul, S. M. S. Sani, N. Noor, A. N. Asma and T. Hasan. A web-based mpox skin lesion detection system using state-of-the-art deep learning models considering racial diversity. *Biomedical Signal Processing and Control*, 98:106742, 2024. doi: 10.1016/j.bspc.2024.106742

## License

No license has been chosen yet. The dataset and the paper have their own terms.
