# Machine Learning Toolbox

## All

### Imports et Setup essentiels
* **Imports standards** : `import numpy as np`, `pandas as pd`, `matplotlib.pyplot as plt`, `seaborn as sns`
* **Imports scipy** : `from scipy.stats import ttest_ind` pour les tests statistiques
* **Imports sklearn complets** : model_selection, preprocessing, pipeline, metrics, feature_selection, ensemble
* **Reset options pandas** : `pd.reset_option('display.max_rows')` pour réinitialiser l'affichage
* **Encodage fichier** : `pd.read_csv("file.csv", encoding="latin1")` pour gérer les caractères spéciaux
* **Copie sécurisée** : `df = data.copy()` pour éviter de modifier les données originales

### Manipulation de base Python
* **List comprehension** : `[i**2 for i in range(10)]` pour créer des listes efficacement
* **Dict comprehension** : `{k: v for k, v in dict.items() if condition}` pour filtrer/transformer dictionnaires
* **String formatting** : `f"il fait {temp} degrés à {ville}"` pour formater les chaînes
* **Gestion fichiers** : `with open("file.txt", "r") as f:` pour ouvrir/fermer automatiquement
* **Fonctions utiles** : `abs()`, `round()`, `max()`, `min()`, `len()`, `sum()`, `all()`, `any()`

### Fonctions utilitaires personnalisées
* **Fonction d'encodage personnalisée** : mapper des valeurs textuelles vers des valeurs numériques
* **Fonction d'inspection des objets** : identifier les types de valeurs uniques dans les colonnes object
* **Fonction get_unique_object_values** : `isinstance(value, str)` pour tester les types de données

## EDA

### Analyse de forme
* **Dimensions du dataset** : `df.shape` pour connaître lignes et colonnes
* **Types de variables** : `df.dtypes.value_counts()` pour compter les types
* **Visualisation des types** : `plt.pie(df.dtypes.value_counts(), autopct='%1.1f%%')` pour voir la répartition
* **Informations générales** : `df.info()` pour aperçu complet du dataset
* **Noms colonnes** : `df.columns` pour lister les colonnes
* **Aperçu données** : `df.head()` et `df.tail()` pour voir début/fin

### Analyse des valeurs manquantes
* **Heatmap des NaN** : `sns.heatmap(df.isna(), cbar=False)` avec `figsize=(20, 10)` pour visualiser les patterns
* **Pourcentage de NaN** : `(df.isna().sum() / df.shape[0]).sort_values(ascending=False)` pour trier par taux
* **Statistiques NaN** : `((df.isna().sum() / df.shape[0]) > 0.90).sum()` pour compter colonnes avec >90% NaN
* **Suppression colonnes NaN** : `df.loc[:, df.isna().sum()/df.shape[0] < 0.90]` pour garder colonnes <90% NaN
* **Groupement par taux NaN** : `missing_rate = df.isna().sum() / df.shape[0]` pour identifier groupes de colonnes

### Exploration des variables
* **Sélection par type** : `df.select_dtypes("float")` pour sélectionner colonnes numériques
* **Distribution des variables numériques** : `sns.displot(df[col])` pour chaque colonne float
* **Valeurs uniques des catégorielles** : `df[col].unique()` pour explorer les modalités
* **Distribution des catégorielles** : `df.value_counts(col).plot.pie()` pour visualiser les proportions
* **Comptage valeurs** : `df[col].value_counts(normalize=True)` pour fréquences relatives
* **Statistiques descriptives** : `df.describe()` pour résumé statistique

### Analyse des relations
* **Comparaison par groupes** : `sns.histplot()` avec `kde=True`, `alpha=0.5`, `stat="density"` pour comparer distributions
* **Matrice de corrélation** : `sns.heatmap(df.corr())` et `sns.clustermap(df.corr())` pour voir les corrélations
* **Relations catégorielles** : `pd.crosstab()` avec `sns.heatmap(annot=True, fmt="d")` pour tableaux croisés
* **Graphiques par groupes** : `sns.countplot(x="var", hue="target", data=df)` pour comparer distributions
* **Pairplot** : `sns.pairplot(df)` pour relations entre toutes variables numériques
* **Corrélation linéaire** : `sns.lmplot(x="var1", y="var2", hue="target", data=df)` pour relations linéaires

### Création de sous-ensembles
* **Filtrage par condition** : `positive_df = df[df["target"] == "positive"]` pour créer sous-groupes
* **Groupement par missing rate** : `blood_columns = df.columns[(missing_rate < 0.90) & (missing_rate > 0.89)]`
* **Création variable composite** : `df["malade"] = np.sum(df[viral_columns] == "detected", axis=1)` pour compter occurrences

### Tests statistiques
* **Test t de Student** : `ttest_ind(group1.dropna(), group2.dropna())` pour comparer moyennes
* **Équilibrage des échantillons** : `df.sample(n, random_state=42)` pour créer échantillons équilibrés
* **Fonction t-test personnalisée** : avec seuil alpha et interprétation automatique

### Visualisations avancées
* **Boucles de visualisation** : `for col in df.select_dtypes("float"): sns.displot(df[col])` pour automatiser
* **Comparaison multi-groupes** : histogrammes superposés avec transparence et couleurs
* **Hospitalisation custom** : fonction pour créer statuts complexes basés sur plusieurs colonnes

## Preprocessing

### Préparation initiale
* **Copie de travail** : `df1 = data.copy()` pour préserver données originales
* **Sélection colonnes stratégique** : `df1[key_columns + list(blood_columns) + list(viral_columns)]`
* **Suppression colonnes inutiles** : `df.drop("Patient ID", axis=1)` pour enlever identifiants
* **Création listes colonnes** : `key_columns = ["col1", "col2"]` pour organiser les features

### Split des données
* **Division train/test** : `train_test_split(df, test_size=0.2, random_state=0)` pour séparer les données
* **Stratification** : `train_test_split(X, y, stratify=y)` pour préserver proportions classes
* **Vérification des proportions** : `df["target"].value_counts()` pour s'assurer de la répartition

### Encodage et transformation
* **Encodage par mapping** : `df[col].map({"positive": 1, "negative": 0})` pour convertir valeurs textuelles
* **Fonction encodage globale** : boucle sur `df.select_dtypes("object")` pour encoder toutes colonnes
* **Label Encoder** : `LabelEncoder()` pour variable cible
* **One-Hot Encoder** : `OneHotEncoder()` pour variables catégorielles nominales
* **Ordinal Encoder** : `OrdinalEncoder()` pour variables catégorielles ordinales

### Feature Engineering
* **Variables composites** : `df["new_var"] = df[viral_columns].sum(axis=1) >= 1` pour créer indicateurs
* **Suppression après création** : `df.drop(viral_columns, axis=1)` après création variable composite
* **Variables binaires** : `df["is_positive"] = df["value"] > 0` pour créer indicateurs
* **Fonction apply personnalisée** : `df.apply(custom_function, axis=1)` pour transformations complexes

### Gestion des valeurs manquantes
* **Suppression NaN** : `df.dropna(axis=0)` pour supprimer lignes avec valeurs manquantes
* **Imputation simple** : `SimpleImputer(strategy="mean")` pour remplacer par moyenne
* **Indicateur NaN** : `df["is_na"] = df.isna().any(axis=1)` pour créer colonne indicatrice
* **Remplacement valeur** : `df.fillna(-999)` pour remplir par valeur spécifique

### Normalisation/Standardisation
* **StandardScaler** : `StandardScaler()` pour centrer/réduire (moyenne=0, écart-type=1)
* **MinMaxScaler** : `MinMaxScaler()` pour normaliser entre 0 et 1
* **RobustScaler** : `RobustScaler()` pour standardisation robuste aux outliers

### Fonction preprocessing complète
* **Pipeline preprocessing** : fonction combinant encodage, feature engineering, et imputation
* **Séparation X/y** : `X = df.drop("target", axis=1)` et `y = df["target"]` pour préparer la modélisation
* **Vérification target** : `print(y.value_counts())` pour contrôler distribution finale

## Modelling

### Sélection de features
* **SelectKBest** : `SelectKBest(f_classif, k=10)` pour sélectionner k meilleures features
* **SelectFromModel** : `SelectFromModel(model)` pour sélectionner selon importance du modèle
* **RFE** : `RFE(estimator, n_features_to_select=5)` pour élimination récursive
* **RFECV** : `RFECV(estimator, cv=5)` pour RFE avec validation croisée
* **VarianceThreshold** : `VarianceThreshold(threshold=0.01)` pour supprimer faible variance

### Création de pipelines
* **Pipeline de base** : `make_pipeline(PolynomialFeatures(2), SelectKBest(f_classif, k=10), Model())` pour enchaîner transformations
* **Preprocessing pipeline** : `make_pipeline(PolynomialFeatures(2, include_bias=False), SelectKBest(f_classif, k=10))` pour réutiliser
* **Pipeline avec scaling** : `make_pipeline(preprocessor, StandardScaler(), KNeighborsClassifier())` pour modèles sensibles à l'échelle
* **Column Transformer** : `make_column_transformer((pipeline, columns))` pour traiter colonnes différemment

### Modèles supervisés
* **Decision Tree** : `DecisionTreeClassifier(random_state=0)` pour arbre de décision (bon pour débuter)
* **Random Forest** : `RandomForestClassifier(random_state=0)` pour ensemble d'arbres
* **SVM** : `SVC(random_state=0)` pour machines à vecteurs de support
* **AdaBoost** : `AdaBoostClassifier(random_state=0)` pour adaptive boosting
* **Gradient Boosting** : `GradientBoostingClassifier()` pour boosting gradient
* **KNN** : `KNeighborsClassifier()` pour classification par voisins
* **Régression linéaire** : `LinearRegression()` pour problèmes de régression

### Modèles multiples et comparaison
* **Dictionnaire de modèles** : `{"RandomForest": pipeline1, "SVC": pipeline2}` pour comparer facilement
* **Comparaison systématique** : `for name, model in dict_of_models.items()` pour évaluer tous modèles
* **Ensemble methods** : `VotingClassifier()`, `StackingClassifier()` pour combiner modèles

### Évaluation des modèles
* **Fonction d'évaluation complète** : combiner `confusion_matrix`, `classification_report`, et `learning_curve`
* **Learning curves** : `learning_curve(model, X, y, cv=4, train_sizes=np.linspace(0.1, 1.0, 10))` pour détecter overfitting
* **Importance des features** : `model.feature_importances_` avec `plot.bar(figsize=(20, 10))` pour visualiser importance
* **Validation croisée** : `cross_val_score(model, X, y, cv=5)` pour évaluation robuste

### Métriques d'évaluation
* **Classification** : `classification_report(y_true, y_pred)` pour rapport complet
* **Matrice confusion** : `confusion_matrix(y_test, y_pred)` pour erreurs détaillées
* **Scores spécifiques** : `f1_score()`, `recall_score()`, `precision_score()`
* **Métriques custom** : définir seuils minimum (ex: F1 min 50%, recall min 70%)

### Optimisation des hyperparamètres
* **Grid Search** : `GridSearchCV(model, param_grid, cv=4, scoring="recall")` pour recherche exhaustive
* **Randomized Search** : `RandomizedSearchCV(model, param_grid, cv=4, scoring="recall", n_iter=50)` pour recherche aléatoire
* **Identification des paramètres** : `print(model)` pour voir paramètres disponibles à optimiser
* **Paramètres par pipeline** : `"pipeline__selectkbest__k": range(4, 15)` pour paramètres imbriqués
* **Meilleurs paramètres** : `grid.best_params_` et `grid.best_estimator_` pour récupérer optimaux

### Ajustement du seuil
* **Precision-Recall Curve** : `precision_recall_curve(y_true, y_score)` pour analyser trade-off precision/recall
* **Seuil personnalisé** : `model.decision_function(X) > threshold` pour ajuster seuil de classification
* **Optimisation métrique** : ajuster seuil selon objectif (privilégier recall ou precision)
* **Fonction finale** : `def final_model(model, X, threshold=0)` pour application seuil custom
* **Visualisation seuil** : `plt.plot(threshold, precision[:-1])` pour voir impact seuil

### Diagnostic et amélioration
* **Analyse overfitting** : comparer train_score et val_score dans learning_curve
* **Feature importance** : `pd.DataFrame(model.feature_importances_, index=X.columns)` pour analyser
* **Réduction fenêtre params** : rétrécir progressivement hyperparamètres dans RandomizedSearch
* **Itération preprocessing** : retour EDA/preprocessing selon résultats modèle

### Bonnes pratiques
* **Arbre de décision en premier** : `DecisionTreeClassifier` comme baseline pour première modélisation
* **Utilisation pipelines** : toujours encapsuler preprocessing et modèle ensemble
* **Validation croisée** : utiliser `cv=4` minimum pour évaluer robustesse
* **Random state** : toujours fixer `random_state=0` pour reproductibilité
* **Métrique adaptée** : choisir scoring selon problème (F1 pour classes déséquilibrées)
* **Éviter data leakage** : séparer données avant preprocessing
* **Polynomial features** : utiliser `PolynomialFeatures(2)` pour capturer interactions

## Modèles non-supervisés

### Clustering
* **K-Means** : `KMeans(n_clusters=3)` pour clustering
* **Isolation Forest** : `IsolationForest()` pour détection anomalies

### Réduction dimensionnalité
* **PCA** : `PCA(n_components=2)` pour réduction dimensionnalité
* **Analyse composantes** : utiliser pour visualisation et preprocessing

## Validation avancée

### Types de validation croisée
* **KFold** : `KFold(n_splits=5)` pour découpage en k parties égales
* **StratifiedKFold** : `StratifiedKFold(n_splits=5)` pour respecter proportions classes
* **LeaveOneOut** : `LeaveOneOut()` pour validation leave-one-out
* **ShuffleSplit** : `ShuffleSplit(n_splits=5)` pour mélange et découpage multiple

### Validation personnalisée
* **Validation métier** : adapter validation selon contexte spécifique
* **Données temporelles** : utiliser `TimeSeriesSplit` pour séries temporelles
* **Groupes corrélés** : `GroupKFold(n_splits=5)` pour groupes corrélés