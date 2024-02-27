import final_model.h5
from sklearn.model_selection import train_test_split

edfp_qsar = datasets.load_generate_edfp()
data = molecular_descriptors.getAllDescriptors(edfp_qsar)
y = data['Target']
X = data.drop(['Target'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
feature_names = model.feature_importances(X_train, y_train)

def qsar_model( feature_names):
    X = X_train[feature_names]
    X_test = X_test[feature_names]
    classification.fit_RandomForestClassifier(X, X_test, y_train, y_test, 'n')