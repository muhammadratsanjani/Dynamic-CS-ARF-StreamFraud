from river import preprocessing
from river import tree
from river import forest
import typing

class CostSensitiveWrapper:
    """
    Wrapper kustom untuk menduplikasi efek parameter `class_weight` 
    atau pembobotan cost-sensitive pada stream learning.
    """
    def __init__(self, model, beta: float = 100.0, minority_class: int = 1):
        self.model = model
        self.beta = beta
        self.minority_class = minority_class
        
    def learn_one(self, x: dict, y: typing.Any):
        # Karena modul ARF dari River mengabaikan sample_weight/w dari luar,
        # kita menyimulasikan "Cost" dengan melatih sampel Fraud berulang kali (Online Oversampling).
        weight = int(self.beta) if y == self.minority_class else 1
        
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

def get_cost_sensitive_arf(beta: float = 100.0):
    """
    Mengembalikan model usulan final: Cost-Sensitive Adaptive Random Forest.
    Menggunakan CostSensitiveWrapper untuk mempenalti secara algoritmik jika 
    model salah memprediksi Fraud.
    """
    base_arf = (
        preprocessing.StandardScaler() |
        forest.ARFClassifier(
            n_models=10, 
            seed=42
        )
    )
    return CostSensitiveWrapper(model=base_arf, beta=beta, minority_class=1)
