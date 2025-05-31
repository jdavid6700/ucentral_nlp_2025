from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.sparse import issparse

class LabelEncoderWrapper:
    def __init__(self, multilabel=False):
        self.multilabel = multilabel
        self.encoder = MultiLabelBinarizer() if multilabel else LabelEncoder()

    def fit_transform(self, y):
        return self.encoder.fit_transform(y)

    def transform(self, y):
        return self.encoder.transform(y)

    def inverse_transform(self, y):
        return self.encoder.inverse_transform(y)
    
    def balance_labels(self, X, y, test_size=0.2, random_state=42, min_samples=2):

        y = pd.Series(y).reset_index(drop=True)

        # Índices de clase 'OTHER'
        idx_other = y[y == 'OTHER'].index
        idx_not_other = y[y != 'OTHER'].index

        # Subconjuntos X e y sin OTHER
        if issparse(X):
            X_not_other = X[idx_not_other.to_list()]
            X_other = X[idx_other.to_list()]
        else:
            X_not_other = X.loc[idx_not_other]
            X_other = X.loc[idx_other]

        y_not_other = y.loc[idx_not_other]
        y_other = y.loc[idx_other]

        # Eliminar clases con menos de min_samples
        valid_classes = y_not_other.value_counts()
        valid_classes = valid_classes[valid_classes >= min_samples].index
        idx_valid = y_not_other[y_not_other.isin(valid_classes)].index

        if issparse(X):
            X_filtered = X_not_other[idx_valid.to_list()]
        else:
            X_filtered = X_not_other.loc[idx_valid]

        y_filtered = y_not_other.loc[idx_valid]

        # División estratificada
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y_filtered,
            test_size=test_size,
            stratify=y_filtered,
            random_state=random_state
        )

        # Sobremuestreo en entrenamiento
        ros = RandomOverSampler(random_state=random_state)
        X_train_bal, y_train_bal = ros.fit_resample(X_train, y_train)

        # Reinsertar OTHER en test
        if issparse(X):
            from scipy.sparse import vstack
            X_test = vstack([X_test, X_other])
        else:
            X_test = pd.concat([X_test, X_other])

        y_test = pd.concat([y_test, y_other]).reset_index(drop=True)

        #print("[INFO] Distribución original de entrenamiento (sin OTHER):")
        #print(y_train.value_counts())
        #print("[INFO] Distribución balanceada de entrenamiento:")
        #print(pd.Series(y_train_bal).value_counts())
        #print("[INFO] Distribución total en test (incluye OTHER):")
        #print(pd.Series(y_test).value_counts())

        return X_train_bal, X_test, y_train_bal, y_test