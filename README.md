# R Scripts and RDS Files for Detecting AI-Generated Texts Using Machine Learning Models

This repository contains the R scripts and RDS files used to generate the results of our study:

> **"Detecting AI-Generated Texts Using Machine Learning Models"**

It is organized following FAIR principles (Findable, Accessible, Interoperable, Reusable) to facilitate reproducibility.

---

## 📂 Files in this repository

### 🔧 R scripts
- `1-tratamento_base.R` – Data cleaning and preprocessing
- `2-definicao_num_termos_zipf_law.R` – Defining the number of terms based on Zipf's law
- `3-criacao_dos_folds.R` – Creating cross-validation folds
- `4.1-validacao_cruzada_MTD_xgb.R` – Cross-validation with BoW (MTD) + XGBoost
- `4.2-validacao_cruzada_TFIDF_xgb.R` – Cross-validation with TF-IDF + XGBoost
- `4.3-validacao_cruzada_doc2vec_xgb.R` – Cross-validation with Doc2Vec + XGBoost

### 💾 RDS files
- `base_completa.RDS` – Complete preprocessed base
- `base_completa_final.RDS` – Final processed dataset
- `folds.RDS` – List of cross-validation folds
- `resultados_treino_BoW.RDS`, `resultados_teste_BoW.RDS` – Training and tes
