#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module de configuration pour DETECTCAM
Gère les paramètres globaux et la configuration de l'application
"""

import os
import json
import logging
from typing import Dict, Any

# Constantes globales d'application
APP_NAME = "DETECTCAM"
APP_VERSION = "0.7.0"
ORGANIZATION = "Mickael BTN."

# Logger pour ce module
logger = logging.getLogger('DetectCam.Config')

# Chemins des fichiers de configuration
def get_config_paths():
    """Retourne les chemins des fichiers de configuration"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    config_paths = {
        'app_dir': base_path,
        'config_dir': os.path.join(base_path, 'config'),
        'data_dir': os.path.join(base_path, 'data'),
        'log_dir': os.path.join(base_path, 'logs'),
        'default_config': os.path.join(base_path, 'config', 'default_config.json'),
        'user_config': os.path.join(base_path, 'config', 'user_config.json'),
        'detection_config': os.path.join(base_path, 'config', 'detection_config.json'),
        'resources_dir': os.path.join(base_path, 'resources'),
    }
    
    # Créer les répertoires s'ils n'existent pas
    for path_name, path in config_paths.items():
        if path_name.endswith('_dir') and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            logger.info(f"Répertoire créé: {path}")
    
    return config_paths

def get_default_config() -> Dict[str, Any]:
    """Retourne la configuration par défaut"""
    return {
        'version': APP_VERSION,
        'general': {
            'language': 'fr',
            'theme': 'auto',
            'hardware_acceleration': True,
            'auto_start': False,
            'start_minimized': False,
        },
        'detection': {
            'model': 'yolo11m.pt',
            'conf_threshold': 0.5,
            'min_detection_interval': 2,
            'save_video': True,
            'video_duration': 5,
            'buffer_size': 150,
            'use_cuda': True,
            'iou_threshold': 0.45,
            'half_precision': True,
            'multi_scale': False,
            'object_filters': ["personne", "voiture", "moto", "sac à dos", "valise"],
            'class_thresholds': {
                "personne": 0.5,
                "sac à dos": 0.3,
                "valise": 0.3
            }
        },
        'display': {
            'resize_mode': 'fit',
            'custom_width': 640,
            'custom_height': 480,
            'resize_percent': 100,
            'auto_resize_label': True,
            'show_confidence': True,
            'show_class': True,
            'show_fps': True,
            'highlight_detections': True,
            'show_zone_numbers': True,
            'fast_resize': True,
            'detection_priority': True
        },
        'alerts': {
            'email_enabled': False,
            'email_address': '',
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'email_user': '',
            'email_password': '',
            'notification_enabled': True,
            'alert_threshold': 5
        },
        'storage': {
            'base_dir': 'detections',
            'videos_dir': 'detections/videos',
            'images_dir': 'detections/images',
            'exports_dir': 'exports',
            'max_storage_days': 30,
            'auto_cleanup': True
        },
        'zones': [],
        'zone_sensitivity': {}
    }

def load_config_file(config_file: str, default_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Charge un fichier de configuration JSON
    
    Args:
        config_file: Chemin vers le fichier de configuration
        default_config: Configuration par défaut à utiliser si le fichier n'existe pas
        
    Returns:
        Dict contenant la configuration
    """
    if default_config is None:
        default_config = get_default_config()
        
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.info(f"Configuration chargée: {config_file}")
                
                # Mettre à jour avec les clés manquantes de la configuration par défaut
                return merge_configs(default_config, config)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Erreur lors du chargement de la configuration {config_file}: {e}")
            logger.info("Utilisation de la configuration par défaut")
            return default_config
    else:
        logger.info(f"Fichier de configuration {config_file} non trouvé, création avec valeurs par défaut")
        save_config_file(config_file, default_config)
        return default_config

def save_config_file(config_file: str, config: Dict[str, Any]) -> bool:
    """
    Sauvegarde la configuration dans un fichier JSON
    
    Args:
        config_file: Chemin vers le fichier de configuration
        config: Dictionnaire de configuration à sauvegarder
        
    Returns:
        True si la sauvegarde a réussi, False sinon
    """
    try:
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        logger.info(f"Configuration sauvegardée: {config_file}")
        return True
    except (IOError, OSError) as e:
        logger.error(f"Erreur lors de la sauvegarde de la configuration {config_file}: {e}")
        return False

def merge_configs(default_config: Dict[str, Any], user_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fusionne la configuration utilisateur avec la configuration par défaut
    pour assurer que toutes les clés nécessaires sont présentes
    
    Args:
        default_config: Configuration par défaut
        user_config: Configuration utilisateur
        
    Returns:
        Configuration fusionnée
    """
    result = default_config.copy()
    
    def merge_dict(target, source):
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                merge_dict(target[key], value)
            else:
                target[key] = value
    
    merge_dict(result, user_config)
    return result

def load_app_settings() -> Dict[str, Any]:
    """
    Charge tous les paramètres de l'application
    
    Returns:
        Dict contenant tous les paramètres
    """
    paths = get_config_paths()
    default_config = get_default_config()
    
    # Charger la configuration utilisateur
    user_config = load_config_file(paths['user_config'], default_config)
    
    # Charger la configuration de détection
    detection_config = load_config_file(paths['detection_config'], default_config)
    
    # Fusionner les configurations
    final_config = merge_configs(user_config, detection_config)
    
    # S'assurer que les chemins sont complets et les dossiers existent
    storage_paths = create_storage_paths(final_config['storage'])
    final_config['storage'].update(storage_paths)
    
    return final_config

def create_storage_paths(storage_config: Dict[str, str]) -> Dict[str, str]:
    """
    Crée les chemins de stockage absolus et s'assure que les dossiers existent
    
    Args:
        storage_config: Configuration de stockage
        
    Returns:
        Dict avec les chemins absolus
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Créer les chemins absolus
    paths = {}
    for key, rel_path in storage_config.items():
        if key.endswith('_dir'):
            abs_path = os.path.join(base_dir, rel_path)
            paths[key] = abs_path
            
            # Créer le dossier s'il n'existe pas
            if not os.path.exists(abs_path):
                os.makedirs(abs_path, exist_ok=True)
                logger.info(f"Dossier de stockage créé: {abs_path}")
    
    return paths

def save_app_settings(settings: Dict[str, Any]) -> bool:
    """
    Sauvegarde tous les paramètres de l'application
    
    Args:
        settings: Paramètres à sauvegarder
        
    Returns:
        True si la sauvegarde a réussi, False sinon
    """
    paths = get_config_paths()
    
    # Séparer les configurations
    user_config = {
        'version': settings.get('version', APP_VERSION),
        'general': settings.get('general', {}),
        'display': settings.get('display', {}),
        'alerts': settings.get('alerts', {}),
        'storage': settings.get('storage', {})
    }
    
    detection_config = {
        'version': settings.get('version', APP_VERSION),
        'detection': settings.get('detection', {}),
        'zones': settings.get('zones', []),
        'zone_sensitivity': settings.get('zone_sensitivity', {})
    }
    
    # Sauvegarder les configurations
    user_success = save_config_file(paths['user_config'], user_config)
    detection_success = save_config_file(paths['detection_config'], detection_config)
    
    return user_success and detection_success
