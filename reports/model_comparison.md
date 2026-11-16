| # | Algorithm | Features | Settings | Accuracy | Precision | Recall | F1 | ROC-AUC | 5-fold CV ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Gradient Boosting | date | 800 rounds, lr 0.05, 63 leaves, L2 1.0 | 0.674 | 0.364 | 0.624 | 0.460 | 0.715 | 0.716 ± 0.003 |
| 2 | Gradient Boosting | date | 800 rounds, lr 0.05, 63 leaves, L2 1.0 | 0.674 | 0.364 | 0.624 | 0.460 | 0.715 |  |
| 3 | Gradient Boosting | date | 500 rounds, lr 0.1, 63 leaves | 0.681 | 0.370 | 0.613 | 0.461 | 0.715 |  |
| 4 | Gradient Boosting | date | 300 rounds, lr 0.1, 31 leaves | 0.670 | 0.360 | 0.623 | 0.457 | 0.712 |  |
| 5 | Random Forest | date | 200 trees, depth 16 | 0.664 | 0.351 | 0.602 | 0.443 | 0.699 |  |
| 6 | Gradient Boosting | base | 300 rounds, lr 0.1 | 0.644 | 0.337 | 0.619 | 0.436 | 0.688 |  |
| 7 | Extra Trees | date | 300 trees, depth 20 | 0.667 | 0.347 | 0.563 | 0.430 | 0.685 |  |
| 8 | Random Forest | base | 200 trees, depth 16 | 0.646 | 0.335 | 0.598 | 0.429 | 0.682 |  |
| 9 | Neural Network (MLP) | date | layers 128,64, alpha 0.001 | 0.631 | 0.329 | 0.629 | 0.432 | 0.679 |  |
| 10 | Decision Tree | date | depth 10 | 0.644 | 0.333 | 0.597 | 0.427 | 0.668 |  |
| 11 | Random Forest | base | 100 trees, depth 8 | 0.619 | 0.317 | 0.616 | 0.418 | 0.667 |  |
| 12 | Logistic Regression | date | C=1.0 | 0.606 | 0.305 | 0.602 | 0.405 | 0.639 |  |
| 13 | Logistic Regression | base | C=1.0 | 0.607 | 0.305 | 0.601 | 0.405 | 0.638 |  |
