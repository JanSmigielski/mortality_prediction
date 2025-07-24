import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report, precision_recall_curve, auc, f1_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV 

def xgb_random_search(X, y, param_grid):
    xgb = XGBClassifier(random_state=2020, eval_metric='logloss', verbosity=0, use_label_encoder=False)
    
    random_search_xgb = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_grid,
        n_iter=200, 
        scoring='roc_auc', 
        cv=5, 
        verbose=2,
        random_state=2020,
        n_jobs=-1 
    )
    
    random_search_xgb.fit(X, y)
    
    best_random_xgb = random_search_xgb.best_estimator_
    y_pred_proba = random_search_xgb.predict_proba(X)[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)

    if thresholds[0] > 1:
        thresholds = thresholds[1:]
        fpr = fpr[1:]
        tpr = tpr[1:]

    youdenJ = tpr - fpr
    best_index = youdenJ.argmax()
    best_thresh = thresholds[best_index]
    
    print("Best parameters for XGBoost (Random Search):", random_search_xgb.best_params_)
    print("Best ROC-AUC for XGBoost (Random Search):", random_search_xgb.best_score_)
    print("Optimal threshold (in terms of Youden statistic): {:.3f}".format(best_thresh))
    print("True Positive Rate: {:.3f}".format(tpr[best_index]))
    print("False Positive Rate: {:.3f}".format(fpr[best_index]))
    
    return  best_random_xgb, best_thresh

def xgb_grid_search(X, y, param_grid):
    xgb = XGBClassifier(random_state=2020, eval_metric='logloss', verbosity=0, use_label_encoder=False)

    grid_search_xgb = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='roc_auc',  
        cv=5,
        verbose=1,
        n_jobs=-1,
        refit=True  
    )

    grid_search_xgb.fit(X, y)

    best_grid_xgb = grid_search_xgb.best_estimator_
    y_pred_proba = grid_search_xgb.predict_proba(X)[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)

    if thresholds[0] > 1:
        thresholds = thresholds[1:]
        fpr = fpr[1:]
        tpr = tpr[1:]

    youdenJ = tpr - fpr
    best_index = youdenJ.argmax()
    best_thresh = thresholds[best_index]
    
    print("Best parameters for XGBoost (Grid Search):", grid_search_xgb.best_params_)
    print("Best ROC-AUC for XGBoost (Grid Search):", grid_search_xgb.best_score_)
    print("Optimal threshold (in terms of Youden statistic): {:.3f}".format(best_thresh))
    print("True Positive Rate: {:.3f}".format(tpr[best_index]))
    print("False Positive Rate: {:.3f}".format(fpr[best_index]))
    
    return best_grid_xgb, best_thresh

def rf_random_search(X, y, param_grid):
    rf = RandomForestClassifier(random_state=2020)

    random_search_rf = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=200,
        scoring='roc_auc',
        cv=5,
        verbose=2,
        random_state=2020,
        n_jobs=-1
    )

    random_search_rf.fit(X, y)

    best_random_rf = random_search_rf.best_estimator_
    y_pred_proba = random_search_rf.predict_proba(X)[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)

    if thresholds[0] > 1:
        thresholds = thresholds[1:]
        fpr = fpr[1:]
        tpr = tpr[1:]

    youdenJ = tpr - fpr
    best_index = youdenJ.argmax()
    best_thresh = thresholds[best_index]
    
    print("Best parameters for Random Forest (Random Search):", random_search_rf.best_params_)
    print("Best ROC-AUC for Random Forest (Random Search):", random_search_rf.best_score_)
    print("Optimal threshold (in terms of Youden statistic): {:.3f}".format(best_thresh))
    print("True Positive Rate: {:.3f}".format(tpr[best_index]))
    print("False Positive Rate: {:.3f}".format(fpr[best_index]))

    return best_random_rf, best_thresh

def rf_grid_search(X, y, param_grid):
    rf = RandomForestClassifier(random_state=2020)

    grid_search_rf = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        verbose=1,
        n_jobs=-1,
        refit=True
    )

    grid_search_rf.fit(X, y)

    best_grid_rf = grid_search_rf.best_estimator_
    y_pred_proba = grid_search_rf.predict_proba(X)[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)

    if thresholds[0] > 1:
        thresholds = thresholds[1:]
        fpr = fpr[1:]
        tpr = tpr[1:]

    youdenJ = tpr - fpr
    best_index = youdenJ.argmax()
    best_thresh = thresholds[best_index]
    
    print("Best parameters for Random Forest (Grid Search):", grid_search_rf.best_params_)
    print("Best ROC-AUC for Random Forest (Grid Search):", grid_search_rf.best_score_)
    print("Optimal threshold (in terms of Youden statistic): {:.3f}".format(best_thresh))
    print("True Positive Rate: {:.3f}".format(tpr[best_index]))
    print("False Positive Rate: {:.3f}".format(fpr[best_index]))

    return best_grid_rf, best_thresh

def model_evaluation(X, y, model, best_thresh):
    y_pred_proba = model.predict_proba(X)[:, 1]
    auc_roc = roc_auc_score(y, y_pred_proba)
    print(f'AUC-ROC: {auc_roc:.4f}')
    
    y_preds_opt = (y_pred_proba >= best_thresh).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, y_preds_opt).ravel()
    print("Confusion matrix (threshold = {:.3f}):".format(best_thresh))
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
 
    print("\nClassification report:")
    print(classification_report(y, y_preds_opt, digits=3))
    
    def plot_confusion_matrix(y_true, y_pred, model):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
        plt.title(f"Confusion Matrix for {model}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

    plot_confusion_matrix(y, y_preds_opt, model)
