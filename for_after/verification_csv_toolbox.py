import csv
import os

# Vérifier si le fichier existe
file_path = "for_after/toolbox.csv"
if not os.path.exists(file_path):
    print(f"Le fichier {file_path} n'existe pas dans le répertoire courant.")
    print(f"Répertoire courant : {os.getcwd()}")
    print(f"Fichiers disponibles : {os.listdir('.')}")
else:
    print(f"Fichier trouvé : {file_path}")
    
    try:
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            nb_cols = len(header)
            print(f"Header trouvé avec {nb_cols} colonnes : {header}")
            
            problemes_trouves = 0
            for i, row in enumerate(reader, start=2):  # ligne 2 car header = 1
                if len(row) != nb_cols:
                    print(f"Ligne {i} a {len(row)} colonnes (au lieu de {nb_cols}) : {row}")
                    problemes_trouves += 1
            
            if problemes_trouves == 0:
                print("✅ Aucun problème détecté dans le fichier CSV")
            else:
                print(f"❌ {problemes_trouves} problème(s) détecté(s)")
                
    except FileNotFoundError:
        print(f"Erreur : Le fichier {file_path} n'a pas pu être ouvert")
    except Exception as e:
        print(f"Erreur inattendue : {e}")
