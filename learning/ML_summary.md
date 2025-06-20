# Résumé Machine Learning – Workflow Complet

Ce document synthétise les étapes clés du machine learning, des données brutes à l’amélioration du modèle, en s’appuyant sur les fonctions et concepts vus dans les notebooks du dossier `learning`.  
**Pour chaque point, le(s) fichier(s) source sont indiqués entre parenthèses.**

---

## 1. Préparation des données

### Gestion des valeurs manquantes  
- **`SimpleImputer`** : Remplace les valeurs manquantes (`np.nan`) par la moyenne, la médiane, la valeur la plus fréquente ou une constante.  
  *(Fichier : `learn_sklearn_imputer.ipynb`)*
- **`KNNImputer`** : Remplace les valeurs manquantes par la valeur des plus proches voisins.  
  *(Fichier : `learn_sklearn_imputer.ipynb`)*
- **`MissingIndicator`** : Ajoute des colonnes binaires indiquant la présence de valeurs manquantes.  
  *(Fichier : `learn_sklearn_imputer.ipynb`)*
- **Combinaison avec `make_union`** : Permet d’appliquer plusieurs transformations en parallèle (ex : imputation + indicateur de valeurs manquantes).  
  *(Fichier : `learn_sklearn_imputer.ipynb`)*

### Encodage des variables catégorielles  
- **`LabelEncoder`** : Encode une colonne de labels (y) en valeurs numériques.  
  *(Fichier : `learn_preprocessing.ipynb`)*
- **`OrdinalEncoder`** : Encode plusieurs colonnes catégorielles en valeurs numériques.  
  *(Fichier : `learn_preprocessing.ipynb`)*
- **`OneHotEncoder`** : Transforme chaque catégorie en une colonne binaire (one-hot).  
  *(Fichier : `learn_preprocessing.ipynb`)*

### Normalisation et standardisation  
- **`MinMaxScaler`** : Met les données à l’échelle [0, 1].  
  *(Fichier : `learn_preprocessing.ipynb`)*
- **`StandardScaler`** : Centre les données (moyenne=0, écart-type=1).  
  *(Fichier : `learn_preprocessing.ipynb`)*
- **`RobustScaler`** : Standardise en utilisant la médiane et l’IQR, moins sensible aux outliers.  
  *(Fichier : `learn_preprocessing.ipynb`)*

### Transformation des features  
- **`PolynomialFeatures`** : Génère des combinaisons polynomiales des features (utile pour modèles non-linéaires).  
  *(Fichier : `learn_preprocessing.ipynb`)*
- **`Binarizer`** : Transforme les valeurs selon un seuil en 0 ou 1.  
  *(Fichier : `learn_preprocessing.ipynb`)*
- **`KBinsDiscretizer`** : Discrétise les données en plusieurs catégories.  
  *(Fichier : `learn_preprocessing.ipynb`)*

### Pipelines de prétraitement  
- **`make_pipeline`** : Enchaîne plusieurs transformations et un modèle dans un pipeline unique.  
  *(Fichiers : `learn_preprocessing.ipynb`, `learn_advanced_pipelines.ipynb`)*
- **`make_column_transformer`** : Applique différents pipelines à différentes colonnes (numériques/catégorielles).  
  *(Fichier : `learn_advanced_pipelines.ipynb`)*
- **`make_union`** : Applique plusieurs transformations en parallèle et concatène les résultats.  
  *(Fichiers : `learn_sklearn_imputer.ipynb`, `learn_advanced_pipelines.ipynb`)*

---

## 2. Séparation en données d’entraînement et de test

- **`train_test_split`** : Sépare le dataset en un ensemble d’entraînement et un ensemble de test (ex : 80%/20%).  
  Permet d’évaluer la performance sur des données jamais vues par le modèle.  
  *(Fichiers : `learn_sklearn_modele_selection.ipynb`, `learn_preprocessing.ipynb`, `learn_sklearn_basics.ipynb`)*

---

## 3. Choix et entraînement du modèle

### Modèles supervisés

- **`LinearRegression`** : Modèle de régression linéaire (prédit une variable continue).  
  - Entraînement : `model.fit(X_train, y_train)`  
  *(Fichier : `learn_sklearn_basics.ipynb`)*
- **`KNeighborsClassifier`** : Classification par les k plus proches voisins.  
  - Hyperparamètre principal : `n_neighbors`  
  *(Fichiers : `learn_sklearn_basics.ipynb`, `learn_sklearn_modele_selection.ipynb`)*
- **`SGDClassifier`** : Classifieur linéaire utilisant la descente de gradient stochastique.  
  *(Fichiers : `learn_preprocessing.ipynb`, `learn_advanced_pipelines.ipynb`)*
- **`DecisionTreeClassifier`** : Arbre de décision pour la classification.  
  *(Fichier : `learn_ensemble_learning.ipynb`)*
- **`SVR`** : Support Vector Regression, pour des problèmes de régression non-linéaire.  
  *(Fichier : `learn_sklearn_basics.ipynb`)*

### Modèles non-supervisés

- **`KMeans`** : Algorithme de clustering (regroupe les données en k clusters).  
  *(Fichier : `learn_unsupervised_learning.ipynb`)*
- **`IsolationForest`** : Détection d’anomalies (outliers).  
  *(Fichier : `learn_unsupervised_learning.ipynb`)*
- **`PCA`** : Réduction de dimensionnalité (conserve la variance maximale).  
  *(Fichier : `learn_unsupervised_learning.ipynb`)*

---

## 4. Évaluation du modèle

### Métriques de régression

- **`mean_absolute_error` (MAE)** : Moyenne des erreurs absolues.  
  *(Fichier : `learn_metriques.ipynb`)*
- **`mean_squared_error` (MSE)** : Moyenne des erreurs quadratiques.  
  *(Fichier : `learn_metriques.ipynb`)*
- **`median_absolute_error`** : Médiane des erreurs absolues.  
  *(Fichier : `learn_metriques.ipynb`)*
- **`R²` (score)** : Coefficient de détermination, proportion de variance expliquée.  
  *(Fichier : `learn_metriques.ipynb`)*

### Métriques de classification

- **`accuracy`** : Proportion de bonnes prédictions.  
  *(Fichier : `learn_sklearn_modele_selection.ipynb`)*
- **`confusion_matrix`** : Matrice des vrais/faux positifs/négatifs.  
  *(Fichiers : `learn_metriques.ipynb`, `learn_sklearn_modele_selection.ipynb`)*

### Visualisation des erreurs

- Histogramme des erreurs pour analyser la distribution des résidus.  
  *(Fichier : `learn_metriques.ipynb`)*

---

## 5. Amélioration du modèle

### Validation croisée et tuning

- **`cross_val_score`** : Évalue le modèle sur plusieurs découpages du train set (cross-validation).  
  *(Fichiers : `learn_sklearn_modele_selection.ipynb`, `learn_cross_validation.ipynb`)*
- **`validation_curve`** : Analyse l’impact d’un hyperparamètre sur la performance (train/validation).  
  *(Fichier : `learn_sklearn_modele_selection.ipynb`)*
- **`GridSearchCV`** : Recherche automatique des meilleurs hyperparamètres via cross-validation.  
  *(Fichier : `learn_sklearn_modele_selection.ipynb`, `learn_preprocessing.ipynb`)*
- **`learning_curve`** : Analyse la performance en fonction de la taille du jeu d’entraînement.  
  *(Fichier : `learn_sklearn_modele_selection.ipynb`)*

#### Types de validation croisée  
- **`KFold`** : Découpe le dataset en x parties égales.  
  *(Fichier : `learn_cross_validation.ipynb`)*
- **`LeaveOneOut`** : Chaque split ne garde qu’un seul échantillon pour la validation.  
  *(Fichier : `learn_cross_validation.ipynb`)*
- **`ShuffleSplit`** : Mélange et découpe plusieurs fois le dataset.  
  *(Fichier : `learn_cross_validation.ipynb`)*
- **`StratifiedKFold`** : Respecte la proportion des classes dans chaque split.  
  *(Fichier : `learn_cross_validation.ipynb`)*
- **`GroupKFold`** : Utilisé quand des groupes de données sont corrélés.  
  *(Fichier : `learn_cross_validation.ipynb`)*

### Sélection de features

- **`VarianceThreshold`** : Supprime les variables à faible variance.  
  *(Fichier : `learn_feature_selection.ipynb`)*
- **`SelectKBest`** : Sélectionne les k variables les plus corrélées à la cible (ex : test chi²).  
  *(Fichier : `learn_feature_selection.ipynb`)*
- **`SelectFromModel`** : Sélectionne les variables importantes selon un modèle (ex : coefficients d’un classifieur).  
  *(Fichier : `learn_feature_selection.ipynb`)*
- **`RFE` / `RFECV`** : Sélection récursive des variables (élimine les moins importantes à chaque étape).  
  *(Fichier : `learn_feature_selection.ipynb`)*

### Techniques avancées

- **Ensemble learning** :
  - **`VotingClassifier`** : Combine plusieurs modèles pour améliorer la robustesse (bagging).  
    *(Fichier : `learn_ensemble_learning.ipynb`)*
- **Clustering avancé** :
  - Autres algos : `DBScan`, `AgglomerativeClustering` (non détaillés ici).  
    *(Fichier : `learn_unsupervised_learning.ipynb`)*

---

## 6. Bonnes pratiques et conseils

- **Ne jamais évaluer la performance sur les données d’entraînement** : toujours utiliser un jeu de test ou la validation croisée.  
  *(Fichier : `learn_sklearn_modele_selection.ipynb`)*
- **Toujours normaliser/standardiser AVANT le split** pour éviter le data leakage.  
  *(Fichier : `learn_preprocessing.ipynb`)*
- **Utiliser les pipelines** pour éviter les erreurs de séquence dans le prétraitement et l’entraînement.  
  *(Fichier : `learn_preprocessing.ipynb`, `learn_advanced_pipelines.ipynb`)*
- **Bien choisir les métriques selon le problème** (classification ou régression).  
  *(Fichier : `learn_metriques.ipynb`)*

---

Ce résumé couvre l’ensemble du workflow machine learning vu dans les notebooks, avec les fonctions et concepts essentiels pour chaque étape, et indique où approfondir chaque point dans les fichiers du dossier.