from river import preprocessing
from river import tree
from river import forest
import typing
import numpy as np
from river import drift

class DynamicCostSensitiveWrapper:
    """
    Wrapper kustom untuk menduplikasi efek pembobotan cost-sensitive 
    secara dinamis berdasarkan Imbalance Ratio (IR_t) dan Drift Confidence (D_t).
    """
    def __init__(self, model, gamma: float = 2.0, alpha: float = 0.999, theta: float = 0.99, minority_class: int = 1):
        self.model = model
        self.gamma = gamma
        self.alpha = alpha
        self.theta = theta
        self.minority_class = minority_class
        
        self.count_maj = 0.0
        self.count_min = 0.0
        
        self.adwin = drift.ADWIN()
        self.d_t = 0.0
        
    def learn_one(self, x: dict, y: typing.Any):
        # Update Imbalance Ratio (IR_t)
        self.count_maj = self.alpha * self.count_maj + (1 if y != self.minority_class else 0)
        self.count_min = self.alpha * self.count_min + (1 if y == self.minority_class else 0)
        
        min_count = max(1e-5, self.count_min)
        ir_t = self.count_maj / min_count
        
        # Update Drift Confidence (D_t)
        y_pred = self.model.predict_one(x)
        error = 0.0 if y_pred == y else 1.0
        self.adwin.update(error)
        
        if self.adwin.change_detected:
            self.d_t = 1.0
        else:
            self.d_t = self.theta * self.d_t
            
        # Hitung weight dinamis (lambda_t)
        if y == self.minority_class:
            lambda_t = min(1000.0, ir_t * (1.0 + self.gamma * self.d_t))
            weight = np.random.poisson(max(1.0, lambda_t))
        else:
            weight = np.random.poisson(1.0)
        
        for _ in range(weight):
            self.model.learn_one(x, y)
            
        return self

    def predict_one(self, x: dict):
        return self.model.predict_one(x)
        
    def predict_proba_one(self, x: dict):
        return self.model.predict_proba_one(x)

def get_baseline_model():
    """
    Mengembalikan model baseline: StandardScaler digabungkan dengan Hoeffding Tree.
    Ini adalah model tree tunggal yang dilatih secara online tanpa mekanisme
    adaptasi drift khusus (ADWIN) maupun penanganan imbalance eksplisit.
    """
    model = (
        preprocessing.StandardScaler() |
        tree.HoeffdingTreeClassifier()
    )
    return model

def get_arf_model():
    """
    Mengembalikan model proposed tahap awal: StandardScaler digabungkan dengan
    Adaptive Random Forest (ARF).
    ARF memiliki ADWIN terintegrasi pada background trees-nya untuk mendeteksi drift.
    """
    model = (
        preprocessing.StandardScaler() |
        forest.ARFClassifier(
            n_models=10, # Dibatasi 10 pohon dulu agar eksekusi di notebook lebih cepat
            seed=42
        )
    )
    return model

def get_cost_sensitive_arf(gamma: float = 2.0):
    """
    Mengembalikan model usulan final: Dynamic Cost-Sensitive Adaptive Random Forest.
    Menggunakan DynamicCostSensitiveWrapper untuk mempenalti secara algoritmik jika 
    model salah memprediksi Fraud.
    """
    base_arf = (
        preprocessing.StandardScaler() |
        forest.ARFClassifier(
            n_models=10, 
            seed=42
        )
    )
    return DynamicCostSensitiveWrapper(model=base_arf, gamma=gamma, minority_class=1)
