#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module de gestion du stockage pour DETECTCAM
Gère les fichiers de détection, vidéos, images et exports.
"""

import os
import shutil
import time
import json
import csv
import logging
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union

from utils.logger import get_module_logger

class StorageManager:
    """
    Gestionnaire de stockage pour les fichiers de l'application
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le gestionnaire de stockage
        
        Args:
            config: Configuration de l'application
        """
        self.logger = get_module_logger('StorageManager')
        
        # Configuration
        self.config = config
        self.storage_config = config.get('storage', {})
        
        # Chemins de stockage
        self.base_dir = self.storage_config.get('base_dir', 'detections')
        self.videos_dir = self.storage_config.get('videos_dir', os.path.join(self.base_dir, 'videos'))
        self.images_dir = self.storage_config.get('images_dir', os.path.join(self.base_dir, 'images'))
        self.exports_dir = self.storage_config.get('exports_dir', 'exports')
        
        # Paramètres de nettoyage
        self.auto_cleanup = self.storage_config.get('auto_cleanup', True)
        self.max_storage_days = self.storage_config.get('max_storage_days', 30)
        
        # Fichier d'historique des détections
        self.history_file = os.path.join(self.base_dir, 'detection_history.json')
        
        # Créer les répertoires s'ils n'existent pas
        try:
            self._ensure_directories()
        except Exception as e:
            self.logger.error(f"Erreur lors de la création des répertoires: {str(e)}")
            # Utiliser des répertoires temporaires en cas d'erreur
            self._use_temp_directories()
        
        # Charger l'historique des détections
        self.detection_history = self._load_detection_history()
        
        self.logger.info("Gestionnaire de stockage initialisé")
        
        # Nettoyage automatique au démarrage si activé
        if self.auto_cleanup:
            try:
                self.cleanup_old_files()
            except Exception as e:
                self.logger.error(f"Erreur lors du nettoyage automatique: {str(e)}")
    
    def _ensure_directories(self):
        """Crée les répertoires de stockage s'ils n'existent pas"""
        try:
            os.makedirs(self.base_dir, exist_ok=True)
            os.makedirs(self.videos_dir, exist_ok=True)
            os.makedirs(self.images_dir, exist_ok=True)
            os.makedirs(self.exports_dir, exist_ok=True)
            
            # Vérifier que les répertoires sont accessibles en écriture
            for directory in [self.base_dir, self.videos_dir, self.images_dir, self.exports_dir]:
                test_file = os.path.join(directory, '.test_write')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
            
            self.logger.info(f"Répertoires de stockage créés/vérifiés")
        except (PermissionError, OSError) as e:
            self.logger.error(f"Erreur lors de la création/vérification des répertoires: {str(e)}")
            raise
    
    def _use_temp_directories(self):
        """Utilise des répertoires temporaires en cas d'erreur"""
        temp_dir = tempfile.gettempdir()
        app_temp_dir = os.path.join(temp_dir, 'detectcam')
        
        self.base_dir = app_temp_dir
        self.videos_dir = os.path.join(app_temp_dir, 'videos')
        self.images_dir = os.path.join(app_temp_dir, 'images')
        self.exports_dir = os.path.join(app_temp_dir, 'exports')
        self.history_file = os.path.join(app_temp_dir, 'detection_history.json')
        
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.exports_dir, exist_ok=True)
        
        self.logger.warning(f"Utilisation des répertoires temporaires: {app_temp_dir}")
    
    def _load_detection_history(self) -> List[Dict[str, Any]]:
        """
        Charge l'historique des détections depuis le fichier
        
        Returns:
            Liste des détections historiques
        """
        if not os.path.exists(self.history_file):
            return []
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
                self.logger.info(f"Historique chargé: {len(history)} entrées")
                return history
        except json.JSONDecodeError as json_error:
            self.logger.error(f"Erreur de format JSON dans l'historique: {str(json_error)}")
            # Tenter de récupérer le fichier corrompu
            self._backup_corrupted_history()
            return []
        except (IOError, PermissionError) as io_error:
            self.logger.error(f"Erreur d'accès à l'historique: {str(io_error)}")
            return []
        except Exception as e:
            self.logger.error(f"Erreur inattendue lors du chargement de l'historique: {str(e)}")
            return []
    
    def _backup_corrupted_history(self):
        """Sauvegarde un fichier d'historique corrompu"""
        try:
            if os.path.exists(self.history_file):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = f"{self.history_file}.{timestamp}.bak"
                shutil.copy2(self.history_file, backup_file)
                self.logger.info(f"Historique corrompu sauvegardé vers {backup_file}")
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde de l'historique corrompu: {str(e)}")
    
    def save_detection_history(self, max_entries: int = 10000) -> bool:
        """
        Sauvegarde l'historique des détections dans le fichier
        
        Args:
            max_entries: Nombre maximal d'entrées à conserver
            
        Returns:
            True si la sauvegarde a réussi, False sinon
        """
        # Limiter le nombre d'entrées
        if len(self.detection_history) > max_entries:
            self.detection_history = self.detection_history[-max_entries:]
        
        # Vérifier que le répertoire existe
        try:
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        except Exception as dir_error:
            self.logger.error(f"Erreur lors de la création du répertoire d'historique: {str(dir_error)}")
            return False
        
        try:
            # Écrire dans un fichier temporaire d'abord (pour éviter la corruption)
            temp_file = f"{self.history_file}.tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.detection_history, f, indent=2)
            
            # Remplacer le fichier original par le fichier temporaire
            if os.path.exists(temp_file):
                if os.path.exists(self.history_file):
                    os.replace(temp_file, self.history_file)
                else:
                    os.rename(temp_file, self.history_file)
                
            self.logger.info(f"Historique sauvegardé: {len(self.detection_history)} entrées")
            return True
        except (IOError, OSError, PermissionError) as e:
            self.logger.error(f"Erreur lors de la sauvegarde de l'historique: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Erreur inattendue lors de la sauvegarde de l'historique: {str(e)}")
            return False
    
    def add_detection(self, detection_info: Dict[str, Any]) -> bool:
        """
        Ajoute une détection à l'historique
        
        Args:
            detection_info: Informations sur la détection
            
        Returns:
            True si l'ajout a réussi, False sinon
        """
        try:
            # S'assurer que la détection a un horodatage
            if 'time' not in detection_info:
                detection_info['time'] = datetime.now().isoformat()
            
            # Nettoyer les valeurs non sérialisables
            clean_info = {}
            for key, value in detection_info.items():
                # Convertir les tableaux numpy en listes
                if hasattr(value, 'tolist'):
                    clean_info[key] = value.tolist()
                # Convertir les objets personnalisés en chaînes
                elif not isinstance(value, (str, int, float, bool, list, dict, tuple, type(None))):
                    clean_info[key] = str(value)
                else:
                    clean_info[key] = value
            
            # Ajouter à l'historique
            self.detection_history.append(clean_info)
            
            # Sauvegarder tous les 10 ajouts pour éviter les écritures fréquentes
            if len(self.detection_history) % 10 == 0:
                self.save_detection_history()
            
            return True
        except Exception as e:
            self.logger.error(f"Erreur lors de l'ajout d'une détection: {str(e)}")
            return False
    
    def save_detection_image(self, image, detection_info: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Sauvegarde une image de détection
        
        Args:
            image: Image à sauvegarder (numpy array)
            detection_info: Informations sur la détection
            
        Returns:
            Chemin de l'image sauvegardée, ou None en cas d'erreur
        """
        import cv2
        
        try:
            # Vérifier que l'image est valide
            if image is None or image.size == 0:
                self.logger.error("Image invalide ou vide")
                return None
                
            # Créer un nom de fichier avec horodatage
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Ajouter des métadonnées au nom si disponibles
            filename_parts = ["detection", timestamp]
            
            if detection_info:
                if 'class_name' in detection_info:
                    class_name = detection_info['class_name']
                    # Nettoyer le nom de classe pour éviter les problèmes de caractères spéciaux
                    class_name = ''.join(c for c in class_name if c.isalnum() or c in ['-', '_', ' '])
                    filename_parts.append(class_name)
                if 'zone' in detection_info:
                    zone_str = str(detection_info['zone']).replace('/', '_')
                    filename_parts.append(f"zone{zone_str}")
            
            # Obtenir le format d'image
            image_format = self.storage_config.get('image_format', 'jpg').lower()
            filename = "_".join(filename_parts) + f".{image_format}"
            
            # Créer le chemin complet
            try:
                # Vérifier que le répertoire existe
                os.makedirs(self.images_dir, exist_ok=True)
                image_path = os.path.join(self.images_dir, filename)
            except Exception as dir_error:
                self.logger.error(f"Erreur lors de la création du répertoire d'images: {str(dir_error)}")
                # Utiliser un répertoire temporaire
                temp_dir = os.path.join(tempfile.gettempdir(), 'detectcam', 'images')
                os.makedirs(temp_dir, exist_ok=True)
                image_path = os.path.join(temp_dir, filename)
            
            # Définir la qualité
            quality_params = []
            if image_format in ['jpg', 'jpeg']:
                quality = self.storage_config.get('image_quality', 95)
                quality_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            elif image_format == 'png':
                compression = self.storage_config.get('png_compression', 9)
                quality_params = [cv2.IMWRITE_PNG_COMPRESSION, compression]
            
            # Sauvegarder l'image
            try:
                if quality_params:
                    success = cv2.imwrite(image_path, image, quality_params)
                else:
                    success = cv2.imwrite(image_path, image)
                
                if not success:
                    raise IOError(f"Échec de l'écriture de l'image: {image_path}")
                
                # Vérifier que le fichier a bien été créé
                if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
                    raise IOError(f"Fichier image vide ou inexistant: {image_path}")
                
                self.logger.info(f"Image sauvegardée: {image_path}")
                return image_path
            except Exception as cv_error:
                self.logger.error(f"Erreur OpenCV lors de la sauvegarde de l'image: {str(cv_error)}")
                return None
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde de l'image: {str(e)}")
            return None
    
    def get_video_path(self, suffix: str = "") -> str:
        """
        Génère un chemin de fichier vidéo
        
        Args:
            suffix: Suffixe à ajouter au nom du fichier
            
        Returns:
            Chemin complet du fichier vidéo
        """
        try:
            # Créer un nom de fichier avec horodatage
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Ajouter le suffixe s'il est fourni
            filename_parts = ["detection", timestamp]
            if suffix:
                # Nettoyer le suffixe pour éviter les problèmes de caractères spéciaux
                suffix = ''.join(c for c in suffix if c.isalnum() or c in ['-', '_', ' '])
                filename_parts.append(suffix)
            
            # Obtenir le format vidéo
            video_format = self.storage_config.get('video_format', 'mp4').lower()
            filename = "_".join(filename_parts) + f".{video_format}"
            
            # Vérifier/créer le répertoire
            try:
                os.makedirs(self.videos_dir, exist_ok=True)
                return os.path.join(self.videos_dir, filename)
            except Exception as dir_error:
                self.logger.error(f"Erreur lors de la création du répertoire vidéo: {str(dir_error)}")
                # Utiliser un répertoire temporaire
                temp_dir = os.path.join(tempfile.gettempdir(), 'detectcam', 'videos')
                os.makedirs(temp_dir, exist_ok=True)
                return os.path.join(temp_dir, filename)
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération du chemin vidéo: {str(e)}")
            # Fallback en cas d'erreur
            backup_dir = os.path.join(tempfile.gettempdir(), 'detectcam', 'videos')
            os.makedirs(backup_dir, exist_ok=True)
            return os.path.join(backup_dir, f"detection_backup_{int(time.time())}.mp4")
    
    def cleanup_old_files(self, days: Optional[int] = None) -> Tuple[int, int]:
        """
        Supprime les fichiers plus anciens que le nombre de jours spécifié
        
        Args:
            days: Nombre de jours (utilise max_storage_days si None)
            
        Returns:
            Tuple (nombre de fichiers supprimés, espace libéré en octets)
        """
        if days is None:
            days = self.max_storage_days
        
        try:
            # Calculer la date limite
            cutoff_date = datetime.now() - timedelta(days=days)
            cutoff_timestamp = cutoff_date.timestamp()
            
            files_deleted = 0
            space_freed = 0
            
            # Nettoyer les vidéos
            deleted, freed = self._cleanup_directory(self.videos_dir, cutoff_timestamp)
            files_deleted += deleted
            space_freed += freed
            
            # Nettoyer les images
            deleted, freed = self._cleanup_directory(self.images_dir, cutoff_timestamp)
            files_deleted += deleted
            space_freed += freed
            
            # Nettoyer les exports
            deleted, freed = self._cleanup_directory(self.exports_dir, cutoff_timestamp)
            files_deleted += deleted
            space_freed += freed
            
            # Nettoyer l'historique des détections
            if self.detection_history:
                original_count = len(self.detection_history)
                filtered_history = []
                
                for entry in self.detection_history:
                    try:
                        entry_time = entry.get('time', '2000-01-01T00:00:00')
                        if isinstance(entry_time, str):
                            detection_date = datetime.fromisoformat(entry_time)
                            if detection_date > cutoff_date:
                                filtered_history.append(entry)
                    except (ValueError, TypeError):
                        # Conserver les entrées avec des dates invalides
                        filtered_history.append(entry)
                
                if len(filtered_history) < original_count:
                    self.detection_history = filtered_history
                    self.save_detection_history()
                    self.logger.info(f"Historique nettoyé: {original_count - len(filtered_history)} entrées supprimées")
            
            self.logger.info(f"Nettoyage terminé: {files_deleted} fichiers supprimés, {space_freed / (1024*1024):.2f} MB libérés")
            return files_deleted, space_freed
            
        except Exception as e:
            self.logger.error(f"Erreur lors du nettoyage des fichiers: {str(e)}")
            return 0, 0
    
    def _cleanup_directory(self, directory: str, cutoff_timestamp: float) -> Tuple[int, int]:
        """
        Supprime les fichiers anciens d'un répertoire
        
        Args:
            directory: Répertoire à nettoyer
            cutoff_timestamp: Horodatage limite
            
        Returns:
            Tuple (nombre de fichiers supprimés, espace libéré en octets)
        """
        if not os.path.exists(directory):
            return 0, 0
        
        files_deleted = 0
        space_freed = 0
        
        try:
            for filename in os.listdir(directory):
                filepath = os.path.join(directory, filename)
                
                try:
                    # Ignorer les sous-répertoires
                    if os.path.isdir(filepath):
                        continue
                    
                    # Vérifier l'âge du fichier (avec protection contre les erreurs)
                    try:
                        file_timestamp = os.path.getmtime(filepath)
                    except (FileNotFoundError, PermissionError, OSError):
                        continue
                    
                    if file_timestamp < cutoff_timestamp:
                        try:
                            # Récupérer la taille avant suppression
                            try:
                                file_size = os.path.getsize(filepath)
                            except:
                                file_size = 0
                            
                            # Supprimer le fichier
                            os.remove(filepath)
                            
                            files_deleted += 1
                            space_freed += file_size
                            
                            self.logger.debug(f"Fichier supprimé: {filepath}")
                        except (IOError, OSError, PermissionError) as del_error:
                            self.logger.error(f"Erreur lors de la suppression de {filepath}: {str(del_error)}")
                except Exception as file_error:
                    self.logger.error(f"Erreur de traitement du fichier {filename}: {str(file_error)}")
        except Exception as dir_error:
            self.logger.error(f"Erreur lors du nettoyage du répertoire {directory}: {str(dir_error)}")
        
        return files_deleted, space_freed
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Retourne des statistiques sur l'utilisation du stockage
        
        Returns:
            Dictionnaire des statistiques
        """
        stats = {
            'total_size': 0,
            'video_size': 0,
            'image_size': 0,
            'exports_size': 0,
            'video_count': 0,
            'image_count': 0,
            'exports_count': 0,
            'oldest_file': None,
            'newest_file': None
        }
        
        try:
            # Statistiques des vidéos
            video_stats = self._get_directory_stats(self.videos_dir, 'video')
            stats.update(video_stats)
            
            # Statistiques des images
            image_stats = self._get_directory_stats(self.images_dir, 'image')
            stats.update(image_stats)
            
            # Statistiques des exports
            exports_stats = self._get_directory_stats(self.exports_dir, 'exports')
            stats.update(exports_stats)
            
            # Calculer le total
            stats['total_size'] = stats['video_size'] + stats['image_size'] + stats['exports_size']
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul des statistiques de stockage: {str(e)}")
            return stats
    
    def _get_directory_stats(self, directory: str, prefix: str) -> Dict[str, Any]:
        """
        Calcule les statistiques d'un répertoire
        
        Args:
            directory: Répertoire à analyser
            prefix: Préfixe pour les clés du dictionnaire de résultat
            
        Returns:
            Dictionnaire des statistiques
        """
        stats = {
            f'{prefix}_size': 0,
            f'{prefix}_count': 0,
            f'{prefix}_oldest': None,
            f'{prefix}_newest': None
        }
        
        if not os.path.exists(directory):
            return stats
        
        oldest_timestamp = float('inf')
        oldest_file = None
        newest_timestamp = 0
        newest_file = None
        
        try:
            for filename in os.listdir(directory):
                filepath = os.path.join(directory, filename)
                
                try:
                    # Ignorer les sous-répertoires
                    if os.path.isdir(filepath):
                        continue
                    
                    # Taille du fichier
                    try:
                        file_size = os.path.getsize(filepath)
                    except:
                        file_size = 0
                        
                    stats[f'{prefix}_size'] += file_size
                    stats[f'{prefix}_count'] += 1
                    
                    # Horodatage du fichier
                    try:
                        file_timestamp = os.path.getmtime(filepath)
                    except:
                        continue
                    
                    # Fichier le plus ancien
                    if file_timestamp < oldest_timestamp:
                        oldest_timestamp = file_timestamp
                        oldest_file = filepath
                    
                    # Fichier le plus récent
                    if file_timestamp > newest_timestamp:
                        newest_timestamp = file_timestamp
                        newest_file = filepath
                except Exception as file_error:
                    self.logger.error(f"Erreur lors de l'analyse du fichier {filename}: {str(file_error)}")
                    continue
        except Exception as dir_error:
            self.logger.error(f"Erreur lors de l'analyse du répertoire {directory}: {str(dir_error)}")
        
        if oldest_file:
            stats[f'{prefix}_oldest'] = {
                'path': oldest_file,
                'timestamp': datetime.fromtimestamp(oldest_timestamp).isoformat()
            }
        
        if newest_file:
            stats[f'{prefix}_newest'] = {
                'path': newest_file,
                'timestamp': datetime.fromtimestamp(newest_timestamp).isoformat()
            }
        
        return stats
    
    def export_detections(self, format: str, path: Optional[str] = None, 
                         start_date: Optional[datetime] = None, 
                         end_date: Optional[datetime] = None) -> Optional[str]:
        """
        Exporte les détections dans un fichier
        
        Args:
            format: Format d'export ('csv', 'json')
            path: Chemin de sortie (auto-généré si None)
            start_date: Date de début pour le filtrage
            end_date: Date de fin pour le filtrage
            
        Returns:
            Chemin du fichier exporté, ou None en cas d'erreur
        """
        # Filtrer les détections par date si nécessaire
        filtered_history = []
        
        for detection in self.detection_history:
            try:
                detection_time_str = detection.get('time', '')
                if not detection_time_str:
                    continue
                    
                detection_time = datetime.fromisoformat(detection_time_str)
                
                if start_date and detection_time < start_date:
                    continue
                
                if end_date and detection_time > end_date:
                    continue
                
                filtered_history.append(detection)
            except (ValueError, TypeError) as date_error:
                self.logger.warning(f"Date invalide dans la détection: {detection_time_str}")
                continue
        
        # Si aucune détection après filtrage
        if not filtered_history:
            self.logger.warning("Aucune détection à exporter après filtrage")
            return None
        
        # Générer un nom de fichier si nécessaire
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detections_{timestamp}.{format}"
            
            try:
                os.makedirs(self.exports_dir, exist_ok=True)
                path = os.path.join(self.exports_dir, filename)
            except Exception as dir_error:
                self.logger.error(f"Erreur lors de la création du répertoire d'exports: {str(dir_error)}")
                # Utiliser un répertoire temporaire
                temp_dir = os.path.join(tempfile.gettempdir(), 'detectcam', 'exports')
                os.makedirs(temp_dir, exist_ok=True)
                path = os.path.join(temp_dir, filename)
        
        try:
            # Créer le répertoire parent si nécessaire
            parent_dir = os.path.dirname(path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            
            # Exporter selon le format demandé
            if format.lower() == 'csv':
                return self._export_csv(filtered_history, path)
            elif format.lower() == 'json':
                return self._export_json(filtered_history, path)
            else:
                self.logger.error(f"Format d'export non supporté: {format}")
                return None
                
        except Exception as e:
            self.logger.error(f"Erreur lors de l'export des détections: {str(e)}")
            return None
    
    def _export_csv(self, detections: List[Dict[str, Any]], path: str) -> str:
        """
        Exporte les détections au format CSV
        
        Args:
            detections: Liste des détections à exporter
            path: Chemin de sortie
            
        Returns:
            Chemin du fichier exporté
        """
        try:
            # Déterminer les colonnes
            columns = set()
            for detection in detections:
                columns.update(detection.keys())
            
            # Éliminer les colonnes complexes (listes, dictionnaires)
            columns_to_remove = []
            for col in columns:
                for detection in detections:
                    if col in detection and isinstance(detection[col], (list, dict, tuple)):
                        columns_to_remove.append(col)
                        break
            
            # Suppression des colonnes complexes
            for col in columns_to_remove:
                columns.discard(col)
            
            # Assurer que les colonnes importantes sont présentes et dans un ordre logique
            ordered_columns = ['time', 'zone', 'class_name', 'confidence']
            
            # Ajouter les autres colonnes
            for col in sorted(columns):
                if col not in ordered_columns:
                    ordered_columns.append(col)
            
            # Éliminer les colonnes inexistantes
            ordered_columns = [col for col in ordered_columns if col in columns]
            
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=ordered_columns)
                writer.writeheader()
                
                for detection in detections:
                    # Assurer que toutes les colonnes sont présentes
                    row = {}
                    for col in ordered_columns:
                        value = detection.get(col, '')
                        # Convertir en chaîne si nécessaire
                        if not isinstance(value, (str, int, float, bool, type(None))):
                            value = str(value)
                        row[col] = value
                    writer.writerow(row)
            
            self.logger.info(f"Export CSV créé: {path} ({len(detections)} détections)")
            return path
        except Exception as e:
            self.logger.error(f"Erreur lors de l'export CSV: {str(e)}")
            raise
    
    def _export_json(self, detections: List[Dict[str, Any]], path: str) -> str:
        """
        Exporte les détections au format JSON
        
        Args:
            detections: Liste des détections à exporter
            path: Chemin de sortie
            
        Returns:
            Chemin du fichier exporté
        """
        try:
            # Nettoyer les objets non sérialisables
            clean_detections = []
            for detection in detections:
                clean_detection = {}
                for key, value in detection.items():
                    # Si valeur est un objet numpy, le convertir en liste
                    if hasattr(value, 'tolist'):
                        clean_detection[key] = value.tolist()
                    # Si valeur est un objet personnalisé, le convertir en chaîne
                    elif not isinstance(value, (str, int, float, bool, list, dict, tuple, type(None))):
                        clean_detection[key] = str(value)
                    else:
                        clean_detection[key] = value
                clean_detections.append(clean_detection)
            
            # Écrire dans un fichier temporaire d'abord (pour éviter la corruption)
            temp_path = f"{path}.tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(clean_detections, f, indent=2)
            
            # Vérifier que le fichier a bien été créé
            if os.path.exists(temp_path):
                # Remplacer le fichier de destination
                if os.path.exists(path):
                    os.replace(temp_path, path)
                else:
                    os.rename(temp_path, path)
            
            self.logger.info(f"Export JSON créé: {path} ({len(detections)} détections)")
            return path
        except Exception as e:
            self.logger.error(f"Erreur lors de l'export JSON: {str(e)}")
            raise
    
    def get_file_list(self, file_type: str = 'all', 
                    start_date: Optional[datetime] = None, 
                    end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Retourne une liste de fichiers filtrée
        
        Args:
            file_type: Type de fichier ('video', 'image', 'all')
            start_date: Date de début pour le filtrage
            end_date: Date de fin pour le filtrage
            
        Returns:
            Liste des fichiers
        """
        files = []
        
        try:
            # Déterminer les répertoires à parcourir
            directories = []
            if file_type == 'video' or file_type == 'all':
                if os.path.exists(self.videos_dir):
                    directories.append((self.videos_dir, 'video'))
            if file_type == 'image' or file_type == 'all':
                if os.path.exists(self.images_dir):
                    directories.append((self.images_dir, 'image'))
            
            # Filtrer par date
            start_timestamp = start_date.timestamp() if start_date else 0
            end_timestamp = end_date.timestamp() if end_date else float('inf')
            
            for directory, type_name in directories:
                try:
                    for filename in os.listdir(directory):
                        filepath = os.path.join(directory, filename)
                        
                        try:
                            # Ignorer les sous-répertoires
                            if os.path.isdir(filepath):
                                continue
                            
                            # Obtenir l'horodatage du fichier
                            try:
                                file_timestamp = os.path.getmtime(filepath)
                            except:
                                continue
                            
                            # Filtrer par date
                            if not (start_timestamp <= file_timestamp <= end_timestamp):
                                continue
                            
                            # Obtenir la taille du fichier
                            try:
                                file_size = os.path.getsize(filepath)
                            except:
                                file_size = 0
                            
                            # Ajouter à la liste
                            file_time = datetime.fromtimestamp(file_timestamp).isoformat()
                            
                            files.append({
                                'path': filepath,
                                'name': filename,
                                'type': type_name,
                                'size': file_size,
                                'timestamp': file_time
                            })
                        except Exception as file_error:
                            self.logger.error(f"Erreur lors de l'analyse du fichier {filename}: {str(file_error)}")
                            continue
                except Exception as dir_error:
                    self.logger.error(f"Erreur lors de l'analyse du répertoire {directory}: {str(dir_error)}")
            
            # Trier par date (récent d'abord)
            files.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return files
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération de la liste des fichiers: {str(e)}")
            return []
