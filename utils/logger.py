#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module de journalisation pour DETECTCAM
Fournit des fonctions pour configurer et utiliser le système de journalisation
"""

import os
import sys
import logging
import logging.handlers
from datetime import datetime
from typing import Optional

# Constantes
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
DEFAULT_LOG_LEVEL = logging.INFO

# Définir les niveaux de log personnalisés
PERFORMANCE = 15  # Entre DEBUG et INFO
logging.addLevelName(PERFORMANCE, "PERFORMANCE")

# Ajouter une méthode pour logger les performances
def log_performance(self, message, *args, **kwargs):
    """Log un message de performance au niveau PERFORMANCE"""
    self.log(PERFORMANCE, message, *args, **kwargs)

# Ajouter la méthode à la classe Logger
logging.Logger.performance = log_performance

def get_log_path() -> str:
    """Retourne le chemin du dossier de logs"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_folder = os.path.join(base_path, 'logs')
    
    # Créer le dossier si nécessaire
    if not os.path.exists(log_folder):
        os.makedirs(log_folder, exist_ok=True)
    
    return log_folder

def setup_logger(
    level: int = DEFAULT_LOG_LEVEL, 
    log_to_console: bool = True,
    log_to_file: bool = True,
    log_name: Optional[str] = None
) -> logging.Logger:
    """
    Configure le système de journalisation
    
    Args:
        level: Niveau de log (ex: logging.INFO)
        log_to_console: Si True, envoie les logs vers la console
        log_to_file: Si True, envoie les logs vers un fichier
        log_name: Nom du logger (si None, utilise 'DetectCam')
    
    Returns:
        Logger configuré
    """
    logger_name = log_name or 'DetectCam'
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # Éviter les doublons de handlers
    if logger.handlers:
        return logger
    
    # Format des logs
    formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
    
    # Log vers la console
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    
    # Log vers un fichier
    if log_to_file:
        log_folder = get_log_path()
        current_date = datetime.now().strftime('%Y%m%d')
        log_file = os.path.join(log_folder, f'detectcam_{current_date}.log')
        
        # Rotation des logs (nouveau fichier chaque jour, garde 30 jours)
        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_file, when='midnight', interval=1, backupCount=30, encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
    
    logger.info(f"Logger {logger_name} configuré avec le niveau {logging.getLevelName(level)}")
    return logger

def get_module_logger(module_name: str) -> logging.Logger:
    """
    Crée un logger pour un module spécifique
    
    Args:
        module_name: Nom du module (ex: 'ui', 'detection')
        
    Returns:
        Logger configuré pour le module
    """
    return logging.getLogger(f'DetectCam.{module_name}')

def log_uncaught_exceptions(exctype, value, tb):
    """
    Fonction de gestion des exceptions non capturées
    À connecter à sys.excepthook
    """
    logger = logging.getLogger('DetectCam')
    logger.critical("Exception non capturée:", exc_info=(exctype, value, tb))

# Installation du handler d'exceptions non capturées
def install_exception_handler():
    """
    Installe un gestionnaire d'exceptions non capturées
    qui les envoie au système de journalisation
    """
    sys.excepthook = log_uncaught_exceptions
