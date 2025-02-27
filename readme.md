# DETECTCAM v0.7.0 🔍

> Un système avancé de détection d'objets et de mouvement basé sur YOLO et PyQt6

![Licence](https://img.shields.io/badge/licence-MIT-blue.svg)
![Version](https://img.shields.io/badge/version-0.7.0-green.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)

<p align="center">
  <img src="resources/detectcam_logo.png" alt="DETECTCAM Logo" width="300"/>
</p>

## 📋 Sommaire

- [Introduction](#-introduction)
- [Fonctionnalités](#-fonctionnalités)
- [Configuration requise](#-configuration-requise)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Interface](#-interface)
- [Configuration](#-configuration)
- [Structure du projet](#-structure-du-projet)
- [Dépannage](#-dépannage)
- [Roadmap](#-roadmap)
- [Licence et contact](#-licence-et-contact)

## 📖 Introduction

DETECTCAM est une application avancée de détection d'objets et de mouvement qui utilise les modèles d'intelligence artificielle YOLO (You Only Look Once) pour détecter, suivre et enregistrer des objets et mouvements dans des flux vidéo. Idéal pour la surveillance de sécurité, l'analyse de trafic, ou tout besoin de détection et d'alerte automatisées.

L'application est construite avec une architecture modulaire, multithreadée et optimisée pour offrir d'excellentes performances même sur des systèmes à ressources limitées.

## ✨ Fonctionnalités

### 🔍 Détection et analyse
- Détection d'objets en temps réel basée sur YOLOv8
- Support pour plus de 80 classes d'objets (personnes, véhicules, animaux, etc.)
- Configuration de zones de détection personnalisées
- Sensibilité ajustable par zone
- Filtrage d'objets par type et confiance
- Tracking d'objets avec ByteTrack ou DeepSORT (opt.)

### 📸 Capture et enregistrement
- Enregistrement vidéo automatique lors des détections
- Buffer de pré-enregistrement pour ne rien manquer
- Capture d'images des détections
- Stockage organisé des détections
- Optimisation de l'espace de stockage

### 🔔 Alertes et notifications
- Notifications système (Windows, macOS, Linux)
- Alertes par e-mail avec images jointes
- Intégration via webhooks (Discord, Slack, etc.)
- Alertes sonores personnalisables
- Seuil d'alerte configurable

### 📊 Statistiques et analyses
- Tableau de bord des détections
- Visualisation temporelle des détections
- Analyse par zone, heure, et type d'objet
- Exportation des données (CSV, JSON)
- Détection d'anomalies

### ⚙️ Performance et optimisation
- Architecture multithreadée
- Support de l'accélération matérielle (CUDA, MPS)
- Mode FastBoost pour les systèmes moins puissants
- Paramètres d'optimisation configurables
- Contrôle des ressources système utilisées

## 💻 Configuration requise

### Exigences minimales
- **Système d'exploitation** : Windows 10/11, macOS 11+, Linux (Ubuntu 20.04+)
- **Processeur** : Dual-core 2.0 GHz
- **Mémoire** : 4 Go RAM
- **Espace disque** : 500 Mo + espace pour les enregistrements
- **Python** : 3.8 ou plus récent

### Configuration recommandée
- **Processeur** : Quad-core 3.0 GHz ou plus
- **Mémoire** : 8 Go RAM ou plus
- **Carte graphique** : NVIDIA avec support CUDA (pour accélération GPU)
- **Webcam** : 720p ou meilleure résolution
- **Connexion Internet** : Pour les alertes e-mail et webhooks

## 🔧 Installation

### Installation avec pip

```bash
# Cloner le dépôt
git clone https://github.com/mickael-btn/detectcam.git
cd detectcam

# Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
python main.py
```

### Installation spécifique à la plateforme

#### Windows
```bash
# Pour les notifications Windows
pip install win10toast
```

#### macOS
```bash
# Pour les notifications macOS
pip install pync
```

#### Linux
```bash
# Pour les notifications Linux
pip install notify2
sudo apt-get install libnotify-bin  # Sur Ubuntu/Debian
```

### Installation de l'accélération GPU (optionnel)
```bash
# Pour support CUDA (NVIDIA)
pip install torch==1.12.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html
```

## 🚀 Utilisation

### Démarrage rapide

1. Lancez l'application avec `python main.py`
2. Sélectionnez une source vidéo (webcam ou fichier)
3. Définissez des zones de détection (optionnel)
4. Cliquez sur "Démarrer" pour lancer la détection
5. Les détections seront affichées en temps réel et enregistrées

### Définir des zones de détection

1. Cliquez sur "Zones" dans la barre d'outils
2. Utilisez l'éditeur de zones pour dessiner des polygones de détection
3. Ajustez la sensibilité pour chaque zone
4. Sauvegardez les modifications

### Configuration des alertes

1. Accédez aux paramètres via le bouton "Paramètres"
2. Sélectionnez l'onglet "Alertes"
3. Activez les méthodes d'alerte souhaitées et configurez-les
4. Définissez le seuil d'alerte (nombre de détections avant notification)

## 🖥️ Interface

### Fenêtre principale
- **Zone de visualisation** : Affichage en temps réel avec détections
- **Panneau de contrôle** : Boutons et paramètres principaux
- **Barre d'état** : Informations et statistiques actuelles

### Éditeur de zones
- Interface intuitive pour dessiner, modifier et paramétrer des zones
- Ajustement visuel de la sensibilité par zone
- Options pour nommer et configurer chaque zone

### Visualisation des statistiques
- Graphiques temporels des détections
- Distribution par heure et jour de la semaine
- Répartition par type d'objet et zone
- Identification des périodes anormales

## ⚙️ Configuration

DETECTCAM est hautement configurable via l'interface ou en modifiant directement les fichiers de configuration.

### Paramètres principaux

| Paramètre | Description | Valeur par défaut |
|-----------|-------------|-------------------|
| Modèle YOLO | Modèle de détection à utiliser | YOLOv8m |
| Seuil de confiance | Confiance minimale pour les détections | 0.5 |
| Intervalle min. | Temps minimum entre détections | 2s |
| FastBoost | Mode d'optimisation de performance | Désactivé |
| Accélération GPU | Utilisation du GPU pour la détection | Activé |

### Configuration avancée

La configuration avancée peut être modifiée dans le fichier `config/default_config.json` ou via l'interface des paramètres.

## 📁 Structure du projet

```
detectcam/
├── main.py                     # Point d'entrée principal
├── config/
│   ├── settings.py             # Gestion de la configuration
│   └── default_config.json     # Configuration par défaut
├── core/
│   ├── detection_engine.py     # Moteur de détection principal
│   ├── video_capture.py        # Capture vidéo multithreadée
│   ├── object_detector.py      # Wrapper YOLO avec optimisations
│   ├── recorder.py             # Enregistrement vidéo
│   └── analytics.py            # Analyses statistiques
├── ui/
│   ├── main_window.py          # Fenêtre principale
│   ├── detection_view.py       # Vue de détection
│   ├── settings_dialog.py      # Dialogues de paramètres
│   ├── zone_editor.py          # Éditeur de zones
│   └── stats_view.py           # Visualisation des statistiques
├── utils/
│   ├── logger.py               # Système de journalisation
│   ├── storage.py              # Gestion du stockage
│   └── alerts.py               # Système d'alertes
├── resources/                  # Ressources (icônes, sons, etc.)
└── detections/                 # Dossier des détections (créé automatiquement)
    ├── videos/                 # Vidéos enregistrées
    └── images/                 # Images capturées
```

## 🔧 Dépannage

### Problèmes courants

| Problème | Solution |
|----------|----------|
| L'application ne démarre pas | Vérifiez que Python 3.8+ est installé et que les dépendances sont correctement installées |
| La webcam n'est pas détectée | Vérifiez les autorisations et les pilotes de la webcam |
| Détection lente | Activez FastBoost ou l'accélération GPU, réduisez la résolution |
| Alertes email non envoyées | Vérifiez votre configuration SMTP et les autorisations d'application |

### Journaux et débogage

Les journaux sont sauvegardés dans le dossier `logs/` et peuvent être utiles pour diagnostiquer les problèmes.

## 🔮 Roadmap

Fonctionnalités et améliorations prévues pour les versions futures :

- [ ] Interface Qt Quick/QML pour des performances UI améliorées
- [ ] Reconnaissance faciale et identification de personnes
- [ ] Intégration avec les systèmes domotiques (Home Assistant, Google Home, etc.)
- [ ] API REST pour l'intégration avec d'autres applications
- [ ] Mode serveur avec interface web
- [ ] Détection d'anomalies comportementales
- [ ] Analyse audio combinée à la détection visuelle
- [ ] Application mobile compagnon (Android/iOS)

## 📄 Licence et contact

### Licence
Ce projet est distribué sous la licence MIT. Voir le fichier `LICENSE` pour plus d'informations.

### Auteur
Mickael BTN - [GitHub](https://github.com/mickael-btn)

### Contact
Pour les questions, suggestions ou retours :
- Créez une issue sur GitHub
- Email : contact@detectcam.com

---

<p align="center">
  <b>DETECTCAM</b> - Surveillance intelligente et sécurité améliorée par l'IA
</p>
