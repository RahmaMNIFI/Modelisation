# Modélisation

## Résumé du Projet

Ce projet explore l'application de l'apprentissage profond à l'analyse de données oncologiques à travers l'utilisation d'images entières de coupes de tissus, appelées Whole Slide Images (WSI). Ces images, obtenues par des scanners haute résolution des sections de tissus, sont composées de milliards de pixels. En raison de leur taille massive, elles sont segmentées en patches de taille uniforme pour faciliter le traitement.

## Objectif

L'objectif principal est d'utiliser le jeu de données Camelyon16, comprenant 400 WSI de coupes de ganglions lymphatiques sentinelles, pour entraîner un modèle basé sur l'attention. Ce modèle vise à classifier les tumeurs et à détecter le cancer du sein.

## Modèle Choisi

Le modèle choisi est le CLAM (Data-efficient and weakly supervised computational pathology on whole-slide images), tel que décrit dans l'article de recherche "Lu et al., Nat Biomed Eng, 2021". Ce modèle utilise l'apprentissage supervisé faiblement pour simplifier le processus en utilisant uniquement les étiquettes au niveau de la diapositive plutôt qu'au niveau du patch.

## Méthodologie

### Étape 1 : Segmentation et Extraction des Patches

La première étape consiste à segmenter les parties de tissu des images WSI à l'aide de seuillage binaire et de fonctions OpenCV. Ensuite, des patches uniformes sont extraits à partir de ces images à l'aide de la bibliothèque OpenSlide.

**Commandes pour créer des patches :**

```bash
python create_patches_fp.py --source "/chemin/vers/votre_dossier_source/" --save_dir "/chemin/vers/votre_dossier_de_sauvegarde/" --patch_size 256 --seg --patch --stitch
```
```bash
python create_patches_fp.py --source "/chemin/vers/votre_dossier_source/" --save_dir "/chemin/vers/votre_dossier_de_sauvegarde/" --patch_size 256 --seg --patch --stitch
```

### Étape 2 : Extraction des Caractéristiques avec ResNet-50

Les caractéristiques des patches de chaque WSI sont extraites à l'aide d'une version simplifiée du modèle ResNet-50, pré-entraînée sur ImageNet. Cette étape est critique et prend du temps en raison de l'extraction de caractéristiques pour chaque patch.

Commandes pour extraire les caractéristiques :
        python extract.py --wsi_csv "/chemin/vers/votre_dossier_de_sauvegarde/process_list_autogen.csv" --wsi_path "/chemin/vers/votre_dossier_source/" --patches_path "/chemin/vers/votre_dossier_de_sauvegarde/patches/" --output_path "/chemin        /vers/votre_dossier_de_sauvegarde/features/"
### Étape 3 : Modèle CLAM

Dans la troisième et dernière étape, les caractéristiques extraites sont utilisées dans le modèle CLAM. Ce modèle calcule les scores d'attention à partir des caractéristiques en passant à travers une couche d'attention. Ensuite, il utilise des classificateurs simples pour calculer deux types de pertes :

Calcul des Scores d'Attention : Cette étape vise à effectuer un pooling intelligent des scores d'attention, optimisant la détection des tumeurs plutôt qu'un simple pooling maximal.
Calcul des Pertes Instance : Les pertes d'instance sont calculées pour entraîner le modèle à distinguer les instances pertinentes (tumeurs) des non-pertinentes.

À la fin, les deux pertes sont combinées dans une somme pondérée pour obtenir la perte finale. Les pertes d'instance ne sont utilisées que pendant l'entraînement, tandis que la perte globale du lot est utilisée pour la rétropropagation et l'évaluation.

Commandes pour entraîner et tester le modèle :
```bash
python train_.py --wsi_labels_path "/content/drive/MyDrive/data/label_test1.csv" --features_path "/content/drive/MyDrive/features/test/" --model_state_dict_path "/content/drive/MyDrive/CLAM/weights_25epochs.pt" --num_epochs 50 --batch_size 1 --bag_weight 0.7
```
```bash

python test.py --wsi_labels_path "/content/drive/MyDrive/data/label_test1.csv" --features_path "/content/drive/MyDrive/features/test/" --labels_path "/content/drive/MyDrive/data/reference.csv" --model_state_dict_path "/content/drive/MyDrive/CLAM/weights_25epochs.pt"
```
Ce projet vise à automatiser et améliorer la précision de la détection du cancer du sein à partir de WSI en utilisant des techniques avancées d'apprentissage profond et d'attention, spécifiquement adaptées aux besoins de la pathologie computationnelle.
Prérequis

Pour exécuter ce projet, assurez-vous d'avoir Python 3.10 installé sur votre système. Utilisez ensuite le fichier requirements.txt pour installer toutes les dépendances nécessaires. Voici la commande à exécuter :
```bash
pip install -r requirements.txt
```
### Références

    Camelyon16 Grand Challenge
    Lu et al., Data-efficient and weakly supervised computational pathology on whole-slide images. Nat Biomed Eng 5, 555–570 (2021)
    CLAM GitHub Repository
