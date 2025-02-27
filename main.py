#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🔍 DETECTCAM v0.7.0 - APPLICATION DE DÉTECTION DE MOUVEMENT
----------------------------------------------------
Un système avancé de détection d'objets utilisant YOLO et PyQt6
avec architecture modulaire et optimisations de performance.

Auteur: Mickael BTN
Version: 0.7.0
"""
import sys
import os
import logging
from PyQt6.QtWidgets import QApplication , QSplashScreen
from PyQt6.QtCore import Qt, QThread
from PyQt6.QtGui import QPixmap

# Import des modules internes
from config.settings import load_app_settings, APP_NAME, APP_VERSION, ORGANIZATION
from ui.main_window import MainWindow
from utils.logger import setup_logger

def show_splash_screen():
    """Affiche un écran de démarrage pendant le chargement de l'application"""
    app_path = os.path.dirname(os.path.abspath(__file__))
    splash_path = os.path.join(app_path, 'resources', 'splash.png')
    
    # Utiliser une image par défaut si l'image de splash n'existe pas
    if not os.path.exists(splash_path):
        splash = QSplashScreen()
        splash.showMessage(f"Chargement de {APP_NAME} v{APP_VERSION}...", 
                        Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter, 
                        Qt.GlobalColor.white)
    else:
        splash_pixmap = QPixmap(splash_path)
        splash = QSplashScreen(splash_pixmap)
    
    splash.show()
    return splash

def main():
    """Point d'entrée principal de l'application"""
    # Configurer la journalisation avant tout
    setup_logger()
    logger = logging.getLogger('DetectCam')
    logger.info(f"Démarrage de {APP_NAME} v{APP_VERSION}")
    
    # Créer l'application Qt
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORGANIZATION)
    app.setApplicationVersion(APP_VERSION)
    
    # Afficher l'écran de démarrage
    splash = show_splash_screen()
    app.processEvents()
    
    # Charger les paramètres globaux
    settings = load_app_settings()
    logger.info("Paramètres de l'application chargés")
    
    try:
        # Créer et afficher la fenêtre principale
        window = MainWindow(settings)
        
        # Fermer l'écran de démarrage une fois la fenêtre principale chargée
        splash.finish(window)
        window.show()
        
        # Exécuter l'application
        sys.exit(app.exec())
        
    except Exception as e:
        logger.critical(f"Erreur critique lors du démarrage: {str(e)}", exc_info=True)
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(None, "Erreur critique", 
                           f"Impossible de démarrer l'application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
