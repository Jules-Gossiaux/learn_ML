
# <p style="text-align:center">ML Project Guide </p>

📋 **Exploratory Data Analysis** `video 27/30`

* Objectif :    
    - Prédire si une personne est infectée en fonction des données cliniques disponibles
* Métrique adaptée choisie ? (ex : F1, Recall...)
    - F1 min 50% et recall min 70%

**I - Analyse de la forme :** `video 28/30`

* [X]Ai-je identifié la variable cible (target) ?
    - target = data["SARS-Cov-2 exam result	"]

* [X]Ai-je noté le nombre de lignes et colonnes du dataset ?
    - (5644, 111)

* [X]Les types de variables sont-ils bien identifiés (numériques, catégorielles...) ? (sns.pie and value_counts)
    -   Numérique pour la plupart sauf pour le résultat test et l' ID

* [X]Ai-je repéré les valeurs manquantes ?
    - La plupart des colonnes à partir de la 7eme lorsque la personne est négative


**II - Analyse du fond :**

* Ai-je visualisé la distribution de la target (histogramme, boxplot) ?
    - 10% positive test => Classes pas équilibrées => utiliser score F1
* Est-ce que je comprends bien la signification de chaque variable ?
    - 
* Ai-je exploré les relations entre les features et la target ?
* Ai-je identifié les valeurs aberrantes (outliers) ?

---

📋 **Pre-processing** `video 29/30`

Objectif : Préparer les données pour les algorithmes de machine learning.

* Ai-je séparé les données en Train Set et Test Set ?
* Les valeurs manquantes (NaN) ont-elles été traitées (suppression ou imputation) ?
* Ai-je encodé les variables catégorielles ?
* Les outliers problématiques ont-ils été traités ou supprimés ?
* Ai-je sélectionné les variables les plus utiles (feature selection) ?
* Ai-je créé de nouvelles variables pertinentes (feature engineering) ?
* Les variables numériques sont-elles mises à l’échelle (normalisation / standardisation) ?

---

📋 **Modelling**

Objectif : Construire un modèle capable d’atteindre l’objectif défini.

* Ai-je choisi une fonction d’évaluation adaptée (F1, Recall, etc.) ?
* Ai-je testé plusieurs types de modèles ?
* Ai-je utilisé GridSearchCV ou une autre méthode pour optimiser les hyperparamètres ?
* Ai-je analysé les erreurs du modèle ?
* Suis-je revenu à l’étape de preprocessing ou EDA si nécessaire ?
* Ai-je étudié les courbes d’apprentissage (learning curves) pour prendre des décisions éclairées ?














































# <p style="text-align:center">ML Project Guide </p>

## 1. Définir un objectif mesurable:

**Objectif**: Prédire si une personne est infectée en fonction des données cliniques disponibles

**Métrique**: ~~accuracy min => 90%~~ non car si 1/10 infectée et que model prédit toujours saine il aura 90% donc **F1 min 50% et recall min 70%**

## 2. EDA(Exploraty Data Analysis) `video 27/30`


#### I. Analyse de la forme

Avant de plonger dans l'analyse des données, il est crucial de bien comprendre leur **structure** et leurs caractéristiques fondamentales. Cette étape, souvent sous-estimée, pose les bases d'une modélisation réussie.

* **Identification de la target** : C'est la variable que vous cherchez à prédire ou à expliquer. La connaître précisément est le point de départ de toute analyse.
* **Nombre de lignes et de colonnes** : Ces dimensions vous donnent une première idée de la taille de votre jeu de données. Un grand nombre de lignes peut nécessiter des optimisations de performance, tandis qu'un grand nombre de colonnes (features) peut indiquer le besoin de réduction de dimensionnalité.
* **Identification des valeurs manquantes** : Les données manquantes (NaN pour "Not a Number") sont monnaie courante et peuvent biaiser vos analyses. Il est essentiel de les repérer tôt pour choisir la stratégie de traitement adéquate.
* **Types de variables** : Distinguer les variables numériques (continues ou discrètes) des variables catégorielles (nominales ou ordinales) est fondamental. Chaque type de variable nécessite un traitement spécifique avant d'être utilisé dans un modèle.



#### II. Analyse du fond

Une fois la structure appréhendée, il est temps de comprendre le **contenu** de vos données et les relations qu'elles entretiennent. Cette phase d'exploration est essentielle pour dégager des insights et identifier les problèmes potentiels.

* **Visualisation de la target (histogramme/boxplot)** : Visualiser la distribution de votre variable cible vous aide à comprendre sa nature (symétrique, asymétrique, présence d'outliers) et à orienter le choix de vos modèles.
* **Compréhension des différentes variables (recherche)** : Ne vous contentez pas des noms de colonnes. Effectuez des recherches sur la signification de chaque variable dans votre domaine d'étude. Une bonne compréhension métier est inestimable.
* **Visualisation des relations : features/target** : Explorez comment chaque variable explicative (`feature`) est liée à votre variable cible. Des graphiques de dispersion, des boxplots groupés ou des corrélations peuvent révéler des tendances importantes.
* **Identification des outliers** : Les valeurs aberrantes peuvent significativement fausser les résultats de vos modèles. Il est crucial de les identifier pour décider si elles doivent être traitées, transformées ou supprimées.

---

## 3. Pre-processing `video 28/30`


### Objectif : Préparer les Données pour le Machine Learning

Le but est de transformer vos données brutes en un format optimal pour les algorithmes de machine learning. Voici les étapes clés :

* **Création du Train Set / Test Set** : Séparez vos données en ensembles d'entraînement et de test pour évaluer la performance de votre modèle.
* **Élimination des NaN** : Gérez les valeurs manquantes via des méthodes comme `dropna()`, l'imputation, ou le traitement des colonnes "vides".
* **Encodage** : Convertissez les variables catégorielles en un format numérique compréhensible par les modèles.
* **Suppression des outliers néfastes au modèle** : Identifiez et traitez les valeurs aberrantes qui pourraient biaiser l'apprentissage.
* **Feature selection** : Sélectionnez les variables les plus pertinentes pour améliorer la performance et réduire la complexité du modèle.
* **Feature engineering** : Créez de nouvelles variables à partir des existantes pour enrichir l'information de votre dataset.
* **Feature scaling** : Normalisez ou standardisez vos variables numériques pour assurer que toutes les caractéristiques contribuent équitablement au modèle.


## 4. Modelling `video 29/30`

 
### Objectif : Développer un Modèle de Machine Learning Robuste

Une fois vos données préparées, l'étape suivante consiste à construire et à affiner un modèle capable de répondre à votre problématique. C'est ici que l'apprentissage automatique prend tout son sens.

* **Définir une fonction d'évaluation** : Avant même de commencer à entraîner des modèles, il est crucial de choisir la bonne métrique pour mesurer leur performance. Que ce soit la précision, le rappel, le F1-score, le RMSE ou autre, votre fonction d'évaluation doit refléter l'objectif de votre projet et la nature de vos données.

* **Entraînement de différents modèles** : Ne mettez pas tous vos œufs dans le même panier ! Il est souvent bénéfique d'expérimenter avec plusieurs types d'algorithmes (régression linéaire, arbres de décision, forêts aléatoires, boosting, SVM, etc.). Chaque modèle a ses forces et ses faiblesses, et celui qui convient le mieux à vos données n'est pas toujours évident à deviner.

* **Optimisation avec GridSearchCV** : Une fois que vous avez identifié quelques modèles prometteurs, l'optimisation des hyperparamètres devient essentielle. `GridSearchCV` (ou d'autres méthodes comme `RandomizedSearchCV`) vous permet de tester systématiquement différentes combinaisons de paramètres pour trouver la configuration qui maximise la performance de votre modèle selon votre fonction d'évaluation.

* **Analyse des erreurs et retour au Preprocessing / EDA** : Un modèle n'est jamais parfait. L'analyse des erreurs (où et pourquoi le modèle se trompe) est une étape critique. Ces erreurs peuvent révéler des lacunes dans votre prétraitement (`Preprocessing`) ou des informations manquantes dans votre analyse exploratoire des données (`EDA`). N'hésitez pas à faire des allers-retours entre ces étapes pour améliorer votre modèle.

* **Learning Curve et prise de décision** : Les courbes d'apprentissage sont un outil puissant pour diagnostiquer si votre modèle souffre de sous-apprentissage (biais élevé) ou de surapprentissage (variance élevée). En analysant ces courbes, vous pouvez prendre des décisions éclairées : collecter plus de données, simplifier ou complexifier le modèle, ou ajuster les hyperparamètres.
---