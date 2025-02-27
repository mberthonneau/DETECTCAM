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
        self._ensure_directories()
        
        # Charger l'historique des détections
        self.detection_history = self._load_detection_history()
        
        self.logger.info("Gestionnaire de stockage initialisé")
        
        # Nettoyage automatique au démarrage si activé
        if self.auto_cleanup:
            self.cleanup_old_files()
    
    def _ensure_directories(self):
        """Crée les répertoires de stockage s'ils n'existent pas"""
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.exports_dir, exist_ok=True)
        
        self.logger.info(f"Répertoires de stockage créés/vérifiés")
    
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
        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"Erreur lors du chargement de l'historique: {str(e)}")
            return []
    
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
        
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.detection_history, f, indent=2)
                
            self.logger.info(f"Historique sauvegardé: {len(self.detection_history)} entrées")
            return True
        except (IOError, OSError) as e:
            self.logger.error(f"Erreur lors de la sauvegarde de l'historique: {str(e)}")
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
            
            # Ajouter à l'historique
            self.detection_history.append(detection_info)
            
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
            # Créer un nom de fichier avec horodatage
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Ajouter des métadonnées au nom si disponibles
            filename_parts = ["detection", timestamp]
            
            if detection_info:
                if 'class_name' in detection_info:
                    filename_parts.append(detection_info['class_name'])
                if 'zone' in detection_info:
                    filename_parts.append(f"zone{detection_info['zone']}")
            
            # Obtenir le format d'image
            image_format = self.storage_config.get('image_format', 'jpg').lower()
            filename = "_".join(filename_parts) + f".{image_format}"
            
            # Créer le chemin complet
            image_path = os.path.join(self.images_dir, filename)
            
            # Définir la qualité
            if image_format == 'jpg' or image_format == 'jpeg':
                quality = self.storage_config.get('image_quality', 95)
                success = cv2.imwrite(image_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            elif image_format == 'png':
                compression = self.storage_config.get('png_compression', 9)
                success = cv2.imwrite(image_path, image, [cv2.IMWRITE_PNG_COMPRESSION, compression])
            else:
                success = cv2.imwrite(image_path, image)
            
            if not success:
                self.logger.error(f"Échec de l'écriture de l'image: {image_path}")
                return None
            
            self.logger.info(f"Image sauvegardée: {image_path}")
            return image_path
            
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
        # Créer un nom de fichier avec horodatage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ajouter le suffixe s'il est fourni
        filename_parts = ["detection", timestamp]
        if suffix:
            filename_parts.append(suffix)
        
        # Obtenir le format vidéo
        video_format = self.storage_config.get('video_format', 'mp4').lower()
        filename = "_".join(filename_parts) + f".{video_format}"
        
        return os.path.join(self.videos_dir, filename)
    
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
            files_deleted += self._cleanup_directory(self.videos_dir, cutoff_timestamp)
            
            # Nettoyer les images
            files_deleted += self._cleanup_directory(self.images_dir, cutoff_timestamp)
            
            # Nettoyer les exports
            files_deleted += self._cleanup_directory(self.exports_dir, cutoff_timestamp)
            
            # Nettoyer l'historique des détections
            if self.detection_history:
                original_count = len(self.detection_history)
                self.detection_history = [
                    entry for entry in self.detection_history
                    if datetime.fromisoformat(entry.get('time', '2000-01-01T00:00:00')) > cutoff_date
                ]
                
                if len(self.detection_history) < original_count:
                    self.save_detection_history()
                    self.logger.info(f"Historique nettoyé: {original_count - len(self.detection_history)} entrées supprimées")
            
            self.logger.info(f"Nettoyage terminé: {files_deleted} fichiers supprimés, {space_freed / (1024*1024):.2f} MB libérés")
            return files_deleted, space_freed
            
        except Exception as e:
            self.logger.error(f"Erreur lors du nettoyage des fichiers: {str(e)}")
            return 0, 0
    
    def _cleanup_directory(self, directory: str, cutoff_timestamp: float) -> int:
        """
        Supprime les fichiers anciens d'un répertoire
        
        Args:
            directory: Répertoire à nettoyer
            cutoff_timestamp: Horodatage limite
            
        Returns:
            Nombre de fichiers supprimés
        """
        if not os.path.exists(directory):
            return 0
        
        files_deleted = 0
        space_freed = 0
        
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            
            # Ignorer les sous-répertoires
            if os.path.isdir(filepath):
                continue
            
            # Vérifier l'âge du fichier
            file_timestamp = os.path.getmtime(filepath)
            if file_timestamp < cutoff_timestamp:
                try:
                    # Récupérer la taille avant suppression
                    file_size = os.path.getsize(filepath)
                    
                    # Supprimer le fichier
                    os.remove(filepath)
                    
                    files_deleted += 1
                    space_freed += file_size
                    
                    self.logger.debug(f"Fichier supprimé: {filepath}")
                except (IOError, OSError) as e:
                    self.logger.error(f"Erreur lors de la suppression de {filepath}: {str(e)}")
        
        return files_deleted
    
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
            stats.update(self._get_directory_stats(self.videos_dir, 'video'))
            
            # Statistiques des images
            stats.update(self._get_directory_stats(self.images_dir, 'image'))
            
            # Statistiques des exports
            stats.update(self._get_directory_stats(self.exports_dir, 'exports'))
            
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
        
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            
            # Ignorer les sous-répertoires
            if os.path.isdir(filepath):
                continue
            
            # Taille du fichier
            file_size = os.path.getsize(filepath)
            stats[f'{prefix}_size'] += file_size
            stats[f'{prefix}_count'] += 1
            
            # Horodatage du fichier
            file_timestamp = os.path.getmtime(filepath)
            
            # Fichier le plus ancien
            if file_timestamp < oldest_timestamp:
                oldest_timestamp = file_timestamp
                oldest_file = filepath
            
            # Fichier le plus récent
            if file_timestamp > newest_timestamp:
                newest_timestamp = file_timestamp
                newest_file = filepath
        
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
        filtered_history = self.detection_history
        
        if start_date or end_date:
            filtered_history = []
            
            for detection in self.detection_history:
                try:
                    detection_time = datetime.fromisoformat(detection.get('time', ''))
                    
                    if start_date and detection_time < start_date:
                        continue
                    
                    if end_date and detection_time > end_date:
                        continue
                    
                    filtered_history.append(detection)
                except (ValueError, TypeError):
                    continue
        
        # Si aucune détection après filtrage
        if not filtered_history:
            self.logger.warning("Aucune détection à exporter après filtrage")
            return None
        
        # Générer un nom de fichier si nécessaire
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detections_{timestamp}.{format}"
            path = os.path.join(self.exports_dir, filename)
        
        try:
            # Créer le répertoire parent si nécessaire
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
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
        # Déterminer les colonnes
        columns = set()
        for detection in detections:
            columns.update(detection.keys())
        
        # Assurer que les colonnes importantes sont présentes et dans un ordre logique
        ordered_columns = ['time', 'zone', 'class_name', 'confidence']
        
        # Ajouter les autres colonnes
        for col in sorted(columns):
            if col not in ordered_columns:
                ordered_columns.append(col)
        
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=ordered_columns)
            writer.writeheader()
            
            for detection in detections:
                # Assurer que toutes les colonnes sont présentes
                row = {col: detection.get(col, '') for col in ordered_columns}
                writer.writerow(row)
        
        self.logger.info(f"Export CSV créé: {path} ({len(detections)} détections)")
        return path
    
    def _export_json(self, detections: List[Dict[str, Any]], path: str) -> str:
        """
        Exporte les détections au format JSON
        
        Args:
            detections: Liste des détections à exporter
            path: Chemin de sortie
            
        Returns:
            Chemin du fichier exporté
        """
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(detections, f, indent=2)
        
        self.logger.info(f"Export JSON créé: {path} ({len(detections)} détections)")
        return path
    
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
        
        # Déterminer les répertoires à parcourir
        directories = []
        if file_type == 'video' or file_type == 'all':
            directories.append(self.videos_dir)
        if file_type == 'image' or file_type == 'all':
            directories.append(self.images_dir)
        
        # Filtrer par date
        start_timestamp = start_date.timestamp() if start_date else 0
        end_timestamp = end_date.timestamp() if end_date else float('inf')
        
        for directory in directories:
            if not os.path.exists(directory):
                continue
                
            for filename in os.listdir(directory):
                filepath = os.path.join(directory, filename)
                
                # Ignorer les sous-répertoires
                if os.path.isdir(filepath):
                    continue
                
                # Obtenir l'horodatage du fichier
                file_timestamp = os.path.getmtime(filepath)
                
                # Filtrer par date
                if not (start_timestamp <= file_timestamp <= end_timestamp):
                    continue
                
                # Ajouter à la liste
                file_type = 'video' if directory == self.videos_dir else 'image'
                file_size = os.path.getsize(filepath)
                file_time = datetime.fromtimestamp(file_timestamp).isoformat()
                
                files.append({
                    'path': filepath,
                    'name': filename,
                    'type': file_type,
                    'size': file_size,
                    'timestamp': file_time
                })
        
        # Trier par date (récent d'abord)
        files.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return files
