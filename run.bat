@echo off
cd /d "%~dp0"

echo Verification et installation des dependances dans l'environnement virtuel...

:: Vérifier si le dossier "venv" existe
if not exist venv (
    echo Creation de l'environnement virtuel...
    python -m venv venv
)

:: Activer l'environnement virtuel
call venv\Scripts\activate.bat

:: Vérifier si pip est installé
python -m pip --version >NUL 2>&1
if errorlevel 1 (
    echo Installation de Pip
    python -m ensurepip --default-pip
)

:: Mettre à jour pip
python -m pip install --upgrade pip

:: Vérifier et installer les packages requis
echo Verification des packages...

:: Installer les dépendances depuis requirements.txt si le fichier existe
if exist requirements.txt (
    echo Installation des dépendances depuis requirements.txt...
    pip install -r requirements.txt
) else (
    echo Aucun fichier requirements.txt trouvé.
)

:: Installation de PyTorch avec CUDA si disponible
echo Installation de PyTorch avec support CUDA...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

:: Vérifier si les dossiers nécessaires existent
if not exist "detections" mkdir detections
if not exist "detections\videos" mkdir detections\videos
if not exist "detections\images" mkdir detections\images
if not exist "exports" mkdir exports

:: Télécharger le modèle YOLO s'il n'existe pas
if not exist "yolo11m.pt" (
    echo Le modele YOLO n'est pas present. Telechargement en cours...
    python -c "from ultralytics import YOLO; YOLO('yolo11m.pt')"
)

echo.
echo Configuration terminee. Lancement de l'application...
echo.

:: Lancer l'application
python main.py

pause