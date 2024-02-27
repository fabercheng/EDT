from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
from model_training import test_dataset, model

y_true = []
y_scores = []
for x, y in test_dataset:
    predictions = model.predict(x)
    y_true.append(y.numpy())
    y_scores.append(predictions.numpy())


accuracy = accuracy_score(y_true, np.round(y_scores))
f1 = f1_score(y_true, np.round(y_scores))
mcc = matthews_corrcoef(y_true, np.round(y_scores))
auc = roc_auc_score(y_true, y_scores)

fpr, tpr, _ = roc_curve(y_true, y_scores)
smooth_fpr = np.linspace(0, 1, 100)
smooth_tpr = np.interp(smooth_fpr, fpr, tpr)

plt.plot(smooth_fpr, smooth_tpr, label='ROC curve (area = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()