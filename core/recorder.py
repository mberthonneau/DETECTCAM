#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module d'enregistrement vidéo pour DETECTCAM
Gère l'enregistrement des flux vidéo avec buffer de pré-enregistrement
"""

import os
import cv2
import time
import numpy as np
from datetime import datetime
from collections import deque
from typing import Optional, Tuple, List, Deque
from PyQt6.QtCore import QObject, pyqtSignal, QTimer

from utils.logger import get_module_logger

class VideoRecorder(QObject):
    """
    Classe d'enregistrement vidéo avec support de pré-enregistrement
    et de compression intelligente.
    """
    
    # Signaux
    recording_progress = pyqtSignal(float)  # Progression de 0.0 à 1.0
    recording_finished = pyqtSignal(str)  # Chemin du fichier sauvegardé
    error_occurred = pyqtSignal(str)  # Message d'erreur
    
    def __init__(self, output_dir: str = 'detections/videos', 
                 fps: float = 30.0, 
                 frame_size: Tuple[int, int] = (640, 480),
                 codec: str = 'mp4v',
                 quality: int = 95):
        """
        Initialise l'enregistreur vidéo
        
        Args:
            output_dir: Répertoire de sortie pour les vidéos
            fps: Images par seconde
            frame_size: Taille de frame (largeur, hauteur)
            codec: Codec vidéo ('mp4v', 'avc1', 'XVID', etc.)
            quality: Qualité de 0 à 100 (100 = meilleure qualité)
        """
        super().__init__()
        self.logger = get_module_logger('VideoRecorder')
        
        # Paramètres
        self.output_dir = output_dir
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec
        self.quality = quality
        
        # État interne
        self.is_recording = False
        self.video_writer = None
        self.current_video_path = None
        self.recording_start_time = None
        self.video_duration = 5.0  # Durée par défaut en secondes
        self.frames_counter = 0
        self.total_frames = 0
        
        # Timer pour la progression
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self._update_progress)
        
        # Créer le répertoire de sortie s'il n'existe pas
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info(f"Enregistreur vidéo initialisé: {fps} FPS, {frame_size}, codec {codec}")
    
    def set_parameters(self, fps: Optional[float] = None, 
                       frame_size: Optional[Tuple[int, int]] = None,
                       codec: Optional[str] = None,
                       quality: Optional[int] = None):
        """
        Met à jour les paramètres d'enregistrement
        
        Args:
            fps: Nouvelle valeur de FPS
            frame_size: Nouvelle taille de frame
            codec: Nouveau codec
            quality: Nouvelle qualité
        """
        if fps is not None:
            self.fps = float(fps)
        
        if frame_size is not None:
            self.frame_size = frame_size
        
        if codec is not None:
            self.codec = codec
        
        if quality is not None:
            self.quality = min(100, max(0, quality))
        
        self.logger.info(f"Paramètres d'enregistrement mis à jour: {self.fps} FPS, {self.frame_size}")
    
    def set_output_directory(self, output_dir: str):
        """
        Définit le répertoire de sortie pour les vidéos
        
        Args:
            output_dir: Nouveau répertoire de sortie
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Répertoire de sortie défini: {output_dir}")
    
    def start_recording(self, duration: float = 5.0, 
                        buffer: Optional[Deque[np.ndarray]] = None) -> bool:
        """
        Démarre l'enregistrement vidéo
        
        Args:
            duration: Durée d'enregistrement en secondes
            buffer: Buffer de frames pour le pré-enregistrement
            
        Returns:
            True si l'enregistrement a démarré, False sinon
        """
        if self.is_recording:
            self.logger.warning("Enregistrement déjà en cours")
            return False
        
        try:
            # Générer un nom de fichier avec horodatage
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_video_path = os.path.join(
                self.output_dir, f"detection_{timestamp}.mp4"
            )
            
            # Vérifier la validité du chemin
            if not os.path.isdir(os.path.dirname(self.current_video_path)):
                os.makedirs(os.path.dirname(self.current_video_path), exist_ok=True)
            
            # Configuration du codec
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            
            # Créer le VideoWriter
            self.video_writer = cv2.VideoWriter(
                self.current_video_path, fourcc, self.fps, self.frame_size
            )
            
            # Vérifier que le writer est correctement initialisé
            if not self.video_writer.isOpened():
                raise IOError(f"Impossible d'ouvrir le writer pour {self.current_video_path}")
            
            # Définir la durée d'enregistrement
            self.video_duration = max(1.0, duration)
            
            # Calculer le nombre total de frames à enregistrer
            self.total_frames = int(self.fps * self.video_duration)
            self.frames_counter = 0
            
            # Écrire le buffer de pré-enregistrement si fourni
            buffer_frames = 0
            if buffer is not None and len(buffer) > 0:
                for frame in buffer:
                    if frame is None:
                        continue
                    
                    # Redimensionner si nécessaire
                    if (frame.shape[1], frame.shape[0]) != self.frame_size:
                        frame = cv2.resize(frame, self.frame_size)
                    
                    self.video_writer.write(frame)
                    buffer_frames += 1
                
                self.frames_counter = buffer_frames
                self.logger.info(f"{buffer_frames} frames du buffer écrites")
            
            # Démarrer l'enregistrement
            self.is_recording = True
            self.recording_start_time = datetime.now()
            
            # Démarrer le timer de progression
            self.progress_timer.start(100)  # Toutes les 100ms
            
            self.logger.info(f"Enregistrement démarré: {self.current_video_path}, durée: {self.video_duration}s")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors du démarrage de l'enregistrement: {str(e)}")
            self.error_occurred.emit(f"Erreur d'enregistrement: {str(e)}")
            
            # Nettoyer en cas d'erreur
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            
            return False
    
    def stop_recording(self) -> Optional[str]:
        """
        Arrête l'enregistrement vidéo
        
        Returns:
            Chemin du fichier vidéo enregistré, ou None en cas d'erreur
        """
        if not self.is_recording:
            return None
        
        try:
            # Arrêter le timer
            self.progress_timer.stop()
            
            # Finaliser l'enregistrement
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            
            self.is_recording = False
            video_path = self.current_video_path
            
            # Vérifier que le fichier existe et n'est pas vide
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                self.logger.error(f"Le fichier vidéo {video_path} est inexistant ou vide")
                self.error_occurred.emit("Erreur: Fichier vidéo vide")
                return None
            
            # Émettre le signal de fin d'enregistrement
            self.recording_finished.emit(video_path)
            self.recording_progress.emit(0.0)  # Réinitialiser la progression
            
            self.logger.info(f"Enregistrement terminé: {video_path}")
            return video_path
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'arrêt de l'enregistrement: {str(e)}")
            self.error_occurred.emit(f"Erreur: {str(e)}")
            
            self.is_recording = False
            return None
    
    def add_frame(self, frame: np.ndarray) -> bool:
        """
        Ajoute une frame à l'enregistrement en cours
        
        Args:
            frame: Frame à ajouter
            
        Returns:
            True si la frame a été ajoutée, False sinon
        """
        if not self.is_recording or self.video_writer is None or frame is None:
            return False
        
        try:
            # Redimensionner si nécessaire
            if (frame.shape[1], frame.shape[0]) != self.frame_size:
                frame = cv2.resize(frame, self.frame_size)
            
            # Écrire la frame
            self.video_writer.write(frame)
            self.frames_counter += 1
            
            # Si le nombre de frames est atteint, arrêter l'enregistrement
            if self.frames_counter >= self.total_frames:
                self.stop_recording()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'ajout de frame: {str(e)}")
            return False
    
    def _update_progress(self):
        """Met à jour la progression de l'enregistrement"""
        if not self.is_recording:
            return
        
        try:
            # Méthode 1: Basée sur le temps écoulé
            if self.recording_start_time:
                elapsed = (datetime.now() - self.recording_start_time).total_seconds()
                progress = min(1.0, elapsed / self.video_duration)
            
            # Méthode 2: Basée sur le nombre de frames
            else:
                progress = min(1.0, self.frames_counter / self.total_frames)
            
            # Émettre le signal de progression
            self.recording_progress.emit(progress)
            
            # Si l'enregistrement est terminé
            if progress >= 1.0:
                self.stop_recording()
                
        except Exception as e:
            self.logger.error(f"Erreur lors de la mise à jour de la progression: {str(e)}")
    
    def save_frame(self, frame: np.ndarray, output_dir: Optional[str] = None) -> Optional[str]:
        """
        Sauvegarde une frame en tant qu'image
        
        Args:
            frame: Frame à sauvegarder
            output_dir: Répertoire de sortie ou None pour utiliser le répertoire par défaut
            
        Returns:
            Chemin de l'image sauvegardée, ou None en cas d'erreur
        """
        if frame is None:
            return None
        
        try:
            # Utiliser le répertoire par défaut si non spécifié
            if output_dir is None:
                output_dir = os.path.join(os.path.dirname(self.output_dir), 'images')
            
            # Créer le répertoire si nécessaire
            os.makedirs(output_dir, exist_ok=True)
            
            # Générer un nom de fichier
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(output_dir, f"detection_{timestamp}.jpg")
            
            # Sauvegarder l'image
            cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            
            self.logger.info(f"Image sauvegardée: {image_path}")
            return image_path
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde de l'image: {str(e)}")
            return None
    
    def get_recording_status(self) -> Tuple[bool, float, Optional[str]]:
        """
        Retourne l'état actuel de l'enregistrement
        
        Returns:
            Tuple (is_recording, progression, path)
        """
        if not self.is_recording:
            return (False, 0.0, None)
        
        # Calculer la progression
        if self.recording_start_time:
            elapsed = (datetime.now() - self.recording_start_time).total_seconds()
            progress = min(1.0, elapsed / self.video_duration)
        else:
            progress = min(1.0, self.frames_counter / self.total_frames)
        
        return (True, progress, self.current_video_path)
