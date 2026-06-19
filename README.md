# Vision and Recognition 2025–26

**Burak Can Arikan (ID: 247248)**

This repository contains my end-of-course project for Vision and Recognition (2025–26). The brief asks for two classifiers.

1. **Cats vs. Dogs (image classification).** I start with a baseline multilayer perceptron (MLP), then improve it with **data augmentation**, and add a small **convolutional neural network (CNN)** for comparison.
2. **Predicting experience level (SVM).** I use a **Support Vector Machine (SVM)** to predict an employee's experience level (entry, mid, senior, or executive) from the "Salary Insights by Job Role" 2024 dataset.

## What's in here

| File | What it is |
|------|------------|
| `Classification_MLP.ipynb` | Part 1 (cats vs. dogs): baseline MLP, augmented MLP, CNN, and the epoch/learning-rate studies. |
| `Classification_SVM.ipynb` | Part 2 (SVM on the salary dataset): a `C` sweep and a class-weight experiment. |
| `report.tex` | The technical report in LaTeX (LNCS format), built on Overleaf. |
| `arikan_burak_can_247248_vision_and_recognition.pdf` | Compiled report. |
| `Dataset salary 2024.csv` | The salary dataset for Task 2. |
| `dogs-vs-cat-small/` | The cats-vs-dogs images, split into `train` / `validation` / `test`. |
| `mlpaugmentedmlpcnn.png`, `svm_accuracy_vs_C.png`, `confusionmatrix.png` | Figures used in the report. |

## How to run

Both notebooks are written for **Google Colab** and run from top to bottom. I first prototyped the code in Colab, then ran the final neural-network experiments on the university GPU cluster as SLURM batch jobs. The SVM is light enough to train on a single CPU.

**Cats vs. Dogs (`Classification_MLP.ipynb`)**
1. Add `dogs-vs-cat-small.zip` to the Colab session or mount Google Drive.
2. Choose a GPU runtime (*Runtime → Change runtime type → GPU*), which is significantly faster.
3. Run each cell in order. The first cell installs PyTorch Lightning, then the data is unzipped, the models are built, and they are trained.

**Salary and SVM (`Classification_SVM.ipynb`)**
1. Download `Dataset salary 2024.csv` from Kaggle, or upload it manually into the session.
2. Run each cell in order. The SVM needs no GPU.

## How the data is prepared

- **Images** are resized to 224×224, turned into tensors, and normalized with the standard ImageNet mean/std. Training images can also be augmented: randomly flipped, rotated, and altered for brightness and contrast, while the validation and test images remain untouched, so the comparison stays fair.
- **Salary table:** the category columns are one-hot encoded, the salary is standardized to the same scale as the 0/1 columns, and the free-text and redundant salary columns are dropped. `experience_level` is removed from the inputs and kept only as the label, so the model never sees the answer.

## Results

**Cats vs. dogs (1000 test images):**

| Model | Test accuracy |
|-------|---------------|
| Baseline MLP | 0.587 |
| MLP + data augmentation | 0.525 |
| CNN | 0.699 |

The CNN won by keeping the two-dimensional structure of the images, while augmentation only weakened the already-weak MLP. Training the CNN longer (5, 10, and 20 epochs) lowered test accuracy and raised test loss, a sign of overfitting, so 5 epochs gave the best result.

**SVM (best at C = 100):** Overall accuracy is around 68%, but the per-class numbers tell the real story. The "senior" class accounts for about 64.5% of the data, so always guessing "senior" would already score around 0.645. The SVM recalls 95% of seniors but only 16% of entry-level and 2% of executives, so the headline accuracy mostly reflects the class imbalance. Re-running with `class_weight='balanced'` drops overall accuracy to 0.49 but raises entry recall to 0.61 and executive recall to 0.48, a fairer model that confirms the original 68% came from the majority class rather than real skill.

The MLP learning-rate study (0.0001 / 0.001 / 0.01 → 0.580 / 0.590 / 0.561) shows 0.001 is the best of the three: a smaller rate learns too slowly in ten epochs, while a larger one overshoots.
