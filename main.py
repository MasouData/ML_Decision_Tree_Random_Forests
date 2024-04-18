import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

class TreeNodes:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = None
        self.threshold = 0
        self.left = None
        self.right = None
        self.length_node_left = None
        self.length_node_right = None

def gini(y):
    classes = np.unique(y)
    gini = 1.0
    for c in classes:
        gini -= (len(y[y == c]) / len(y)) ** 2
    return gini

def split(X, y, feature_index, threshold):
    left_mask = X[:, feature_index] <= threshold
    right_mask = X[:, feature_index] > threshold
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

def best_split(X, y, minleaf, nfeat):
    best_gini = 2
    best_feature_index = None
    best_threshold = None
    for feature_index in range(0, nfeat):  # nfeat here is the total number of features
        feature = X[:, feature_index]
        sorted_feat = np.unique(np.sort(feature))
        thresholds = (sorted_feat[0:len(sorted_feat) - 1] + sorted_feat[1:len(sorted_feat)]) / 2
        for threshold in thresholds:
            X_left, y_left, X_right, y_right = split(X, y, feature_index, threshold)
            if len(y_left) < minleaf or len(y_right) < minleaf:  # checking the minleaf constraint
                continue

            gini_left = gini(y_left)
            gini_right = gini(y_right)
            weighted_gini = len(y_left) / len(y) * gini_left + len(y_right) / len(y) * gini_right
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature_index = feature_index
                best_threshold = threshold

    return best_feature_index, best_threshold

def tree_grow(X, y, nmin, minleaf, nfeat):
    if len(y) < nmin or len(np.unique(y)) == 1:
        return TreeNodes(predicted_class=np.argmax(np.bincount(y)))

    feature_index, threshold = best_split(X, y, minleaf, nfeat)

    if feature_index is None:
        return TreeNodes(predicted_class=np.argmax(np.bincount(y)))

    X_left, y_left, X_right, y_right = split(X, y, feature_index, threshold)

    node = TreeNodes(predicted_class=np.argmax(np.bincount(y)))
    node.feature_index = feature_index
    node.threshold = threshold
    node.length_node_left = len(y_left)
    node.length_node_right = len(y_right)
    node.left = tree_grow(X_left, y_left, nmin, minleaf, nfeat)
    node.right = tree_grow(X_right, y_right, nmin, minleaf, nfeat)
    return node

def tree_pred(x, tr):
    if tr.left is None and tr.right is None:  # root without split
        return tr.predicted_class
    if x[tr.feature_index] <= tr.threshold:
        return tree_pred(x, tr.left)
    else:
        return tree_pred(x, tr.right)


def tree_grow_b(X, y, nmin, minleaf, nfeat, m=100):
    trees = []
    for _ in range(m):
        sample_indexs = np.random.choice(len(y), size=len(y), replace=True)
        X_sample = X[sample_indexs]
        y_sample = y[sample_indexs]
        tree = tree_grow(X_sample, y_sample, nmin, minleaf, nfeat)
        trees.append(tree)
    return trees


def tree_pred_b(X, trees):
    all_predictions = []
    for tree in trees:
        predictions = [tree_pred(x, tree) for x in X]
        all_predictions.append(predictions)
    all_predictions = np.array(all_predictions)
    all_predictions = np.transpose(all_predictions)
    y_pred = [np.argmax(np.bincount(predictions)) for predictions in all_predictions]
    return np.array(y_pred)

# ###### Data ###############
df = pd.read_csv('./dataset/promise-2_0a-packages-csv/eclipse-metrics-packages-2.0.csv', sep=';')
selected_features_train = df[["pre", "ACD_avg", "ACD_max", "ACD_sum", "FOUT_avg", "FOUT_max", "FOUT_sum", "MLOC_avg", "MLOC_max", "MLOC_sum", "NBD_avg", "NBD_max", "NBD_sum", "NOCU", "NOF_avg", "NOF_max", "NOF_sum", "NOI_avg", "NOI_max", "NOI_sum", "NOM_avg", "NOM_max", "NOM_sum", "NOT_avg", "NOT_max", "NOT_sum", "NSF_avg", "NSF_max", "NSF_sum", "NSM_avg", "NSM_max", "NSM_sum", "PAR_avg", "PAR_max", "PAR_sum", "TLOC_avg", "TLOC_max", "TLOC_sum", "VG_avg", "VG_max", "VG_sum"]]
label_train = df[["post"]]

df1 = pd.read_csv('./dataset/promise-2_0a-packages-csv/eclipse-metrics-packages-3.0.csv', sep=';')
selected_features_test = df1[["pre", "ACD_avg", "ACD_max", "ACD_sum", "FOUT_avg", "FOUT_max", "FOUT_sum", "MLOC_avg", "MLOC_max", "MLOC_sum", "NBD_avg", "NBD_max", "NBD_sum", "NOCU", "NOF_avg", "NOF_max", "NOF_sum", "NOI_avg", "NOI_max", "NOI_sum", "NOM_avg", "NOM_max", "NOM_sum", "NOT_avg", "NOT_max", "NOT_sum", "NSF_avg", "NSF_max", "NSF_sum", "NSM_avg", "NSM_max", "NSM_sum", "PAR_avg", "PAR_max", "PAR_sum", "TLOC_avg", "TLOC_max", "TLOC_sum", "VG_avg", "VG_max", "VG_sum"]]
label_test = df1[["post"]]

X_train = selected_features_train.to_numpy()
y_train = np.where(label_train["post"].to_numpy() == 0, 0, 1)

X_test = selected_features_test.to_numpy()
y_test = np.where(label_test["post"].to_numpy() == 0, 0, 1)


################# Single Classification Tree ############

# These numbers should be changed
nmin = 15
minleaf = 5


decision_trees = tree_grow(X_train, y_train, nmin, minleaf, nfeat=41)
# y_pred = tree_pred(X_test, decision_trees)
y_pred = np.array([tree_pred(x, decision_trees) for x in X_test])

cm = confusion_matrix(y_test, y_pred)
accuracy = np.sum(y_pred == y_test) / len(y_test)
precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
print('The results for the single tree')
print(f"Confusion Matrix is:\n {cm}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

################## Bagging tree ################
decision_trees = tree_grow_b(X_train, y_train, nmin, minleaf, nfeat=41, m=100)
y_pred = tree_pred_b(X_test, decision_trees)


cm = confusion_matrix(y_test, y_pred)
accuracy = np.sum(y_pred == y_test) / len(y_test)
precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])

print('The results for Bagging tree')
print(f"Confusion Matrix is:\n {cm}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

############ Random Forest ##############################
decision_trees = tree_grow_b(X_train, y_train, nmin, minleaf, nfeat=6, m=100)
y_pred = tree_pred_b(X_test, decision_trees)

cm = confusion_matrix(y_test, y_pred)
accuracy = np.sum(y_pred == y_test) / len(y_test)
precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])

print('The results for Random Forest')
print(f"Confusion Matrix is:\n {cm}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

##################### Statistical test#333333333
from sklearn.model_selection import KFold
from scipy import stats


# Define a function for each model
def dt_model(X_train, y_train, X_test):
    decision_trees = tree_grow(X_train, y_train, nmin, minleaf, nfeat=41)
    y_pred = np.array([tree_pred(x, decision_trees) for x in X_test])
    return y_pred


def bt_model(X_train, y_train, X_test):
    decision_trees = tree_grow_b(X_train, y_train, nmin, minleaf, nfeat=41, m=100)
    y_pred = tree_pred_b(X_test, decision_trees)
    return y_pred


def rf_model(X_train, y_train, X_test):
    decision_trees = tree_grow_b(X_train, y_train, nmin, minleaf, nfeat=6, m=100)
    y_pred = tree_pred_b(X_test, decision_trees)
    return y_pred


# Define a function for 5x2-fold cross-validation
def cross_val_5x2(model_func, X, y):
    kf = KFold(n_splits=5)
    scores = []

    for train_index, test_index in kf.split(X):
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        # First round
        y_pred = model_func(X_train_fold, y_train_fold, X_test_fold)
        accuracy = np.sum(y_pred == y_test_fold) / len(y_test_fold)
        scores.append(accuracy)

        # Second round (swap train and test sets)
        y_pred = model_func(X_test_fold, y_test_fold, X_train_fold)
        accuracy = np.sum(y_pred == y_train_fold) / len(y_train_fold)
        scores.append(accuracy)

    return scores


# Perform 5x2-fold cross-validation
scores_dt = cross_val_5x2(dt_model, X_train, y_train)
scores_bt = cross_val_5x2(bt_model, X_train, y_train)
scores_rf = cross_val_5x2(rf_model, X_train, y_train)

# Perform paired t-tests
t_statistic_bt_vs_dt, p_value_bt_vs_dt = stats.ttest_rel(scores_bt, scores_dt)
t_statistic_rf_vs_dt, p_value_rf_vs_dt = stats.ttest_rel(scores_rf, scores_dt)
t_statistic_rf_vs_bt, p_value_rf_vs_bt = stats.ttest_rel(scores_rf, scores_bt)

print(f"BT vs DT: t-statistic = {t_statistic_bt_vs_dt}, p-value = {p_value_bt_vs_dt}")
print(f"RF vs DT: t-statistic = {t_statistic_rf_vs_dt}, p-value = {p_value_rf_vs_dt}")
print(f"RF vs BT: t-statistic = {t_statistic_rf_vs_bt}, p-value = {p_value_rf_vs_bt}")

