# P3 — Adaptive Fraud Detection on Financial Transaction Streams

**Kode:** P3
**Tipe:** Eksperimen / Applied Research
**Status:** 🔴 Belum mulai (target mulai setelah P2 submit)

---

## 📌 Judul Tentatif

```
"Adaptive Fraud Detection on Financial Transaction Streams using
 Online Learning with Concept Drift Awareness"
```

**Alternatif judul:**
- *"Real-Time Credit Card Fraud Detection via Incremental Learning with ADWIN-based Drift Adaptation"*
- *"Online Ensemble Learning for Evolving Fraud Patterns in Financial Data Streams"*

---

## 🎯 Target Venue

| Pilihan | Jurnal | Ranking | Alasan |
|---|---|---|---|
| **Utama** | Applied Soft Computing | Q1 (Elsevier) | Inklusif, applied ML, review ~3 bln |
| **Alternatif 1** | Expert Systems with Applications | Q1 (Elsevier) | Sangat cocok untuk aplikasi fraud |
| **Alternatif 2** | IEEE Access | Q2 (IEEE) | Fallback, open access, cepat |
| **Alternatif 3** | Engineering Applications of AI | Q1 (Elsevier) | Cocok untuk applied stream ML |

---

## 💡 Motivasi & Gap Riset

### Latar Belakang
- Fraud keuangan berkembang secara adaptif — pola fraud berubah terus (*concept drift*)
- Sistem deteksi fraud konvensional dilatih secara batch → lambat beradaptasi
- Volume transaksi keuangan sangat besar → butuh *online/incremental learning*
- *Class imbalance* ekstrem: fraud <<1% dari total transaksi

### Gap yang Diisi Paper Ini
> Sebagian besar studi deteksi fraud hanya menggunakan **batch ML** (Random Forest, XGBoost).
> Studi yang menggabungkan **online learning + drift adaptation khusus untuk pola fraud yang berevolusi** masih terbatas.

### Koneksi ke Prof. Bifet
- Deteksi anomali pada stream = salah satu fokus lab Bifet (lihat paper ARES)
- Concept drift pada fraud = problem nyata yang ia angkat di AAAI-26

---

## 📊 Dataset

| Dataset | Sumber | Jumlah Instance | Keterangan |
|---|---|---|---|
| **Credit Card Fraud** | Kaggle (ULB) | 284.807 | Benchmark klasik, imbalance 0.17% |
| **PaySim Synthetic** | Kaggle | 6.3 juta | Simulasi mobile money fraud |
| **IEEE-CIS Fraud** | Kaggle | ~590.000 | Kompetisi IEEE, lebih realistis |

---

## 🔬 Metodologi

```
[Transaction Stream]
        │
        ▼
[Preprocessing + Feature Engineering]
        │
        ▼
[Online Classifier]           ← Adaptive Random Forest (ARF) / Hoeffding Tree
        │
        ▼
[ADWIN Drift Detector]        ← Deteksi perubahan pola fraud
        │
        ▼
[Imbalance Handler]           ← Online SMOTE / cost-sensitive learning
        │
        ▼
[Prequential Evaluation]
        │
        ▼
[Metrics: AUC, G-Mean, F1, Detection Delay]
```

### Algoritma yang Dibandingkan
| Algoritma | Tipe | Tools |
|---|---|---|
| Hoeffding Tree (HT) | Baseline online | River |
| Adaptive Random Forest (ARF) | Online ensemble | CapyMOA |
| Streaming Random Patches (SRP) | Advanced ensemble | CapyMOA |
| Online Bagging + ADWIN | Adaptive | River |
| **Proposed: ARF + ADWIN + Cost-Sensitive** | Proposed method | River + custom |

---

## 🧪 Rencana Eksperimen

| ID | Eksperimen | Metrik |
|---|---|---|
| E1 | Perbandingan algoritma online vs batch (RF, XGBoost) | AUC, F1, waktu |
| E2 | Efek drift adaptation (dengan/tanpa ADWIN) | G-Mean under drift |
| E3 | Penanganan class imbalance | Recall fraud class |
| E4 | Simulasi drift buatan (fraud pattern shift) | Detection delay |
| E5 | Cross-dataset validation | Generalizability |

---

## 📐 Struktur Paper

1. **Abstract** (~250 kata)
2. **Introduction** (~1 halaman)
3. **Related Work** (~2 halaman)
   - Fraud detection: batch vs online
   - Concept drift dalam konteks fraud
   - Class imbalance pada data stream
4. **Methodology** (~2 halaman)
5. **Experimental Setup** (~1 halaman)
6. **Results & Discussion** (~3 halaman)
7. **Conclusion & Future Work** (~0.5 halaman)
8. **References** (~35–45 referensi)

---

## 📅 Timeline

| Bulan | Aktivitas |
|---|---|
| **Oktober 2026** | Setup dataset, eksplorasi EDA |
| **November 2026** | Implementasi baseline & algoritma proposed |
| **Desember 2026** | Eksperimen lengkap + analisis |
| **Januari 2027** | Penulisan draft |
| **Februari 2027** | Revisi + submit |

---

## 📝 Catatan Tambahan

- **Bonus angle**: Hubungkan ke konteks fintech Indonesia (OJK, GoPay, Dana) → menarik untuk jurnal applied
- Kode + dataset preprocessing upload ke GitHub
- Paper ini bisa dikembangkan menjadi topik PhD jika Bifet tertarik pada angle finansial
