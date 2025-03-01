#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module de capture vidéo pour DETECTCAM
Gère la capture vidéo dans un thread séparé
"""
import os  # Ajouté pour la référence à os.path.exists
import cv2
import time
import logging
import numpy as np
from typing import Union, Tuple, Optional
from PyQt6.QtCore import QThread, pyqtSignal, QMutex

from utils.logger import get_module_logger

class VideoCaptureThread(QThread):
    """Thread de capture vidéo pour éviter de bloquer l'interface utilisateur"""
    
    # Signaux
    frame_captured = pyqtSignal(np.ndarray)  # Frame capturée
    error_occurred = pyqtSignal(str)  # Erreur
    source_changed = pyqtSignal(str)  # Source vidéo changée
    
    def __init__(self, source: Union[str, int] = None):
        """
        Initialise le thread de capture vidéo
        
        Args:
            source: Source vidéo (chemin de fichier, URL ou indice de caméra)
        """
        super().__init__()
        self.logger = get_module_logger('VideoCapture')
        
        # Propriétés vidéo
        self.source = source
        self.cap = None
        self.running = False
        self.paused = False
        self.interval_ms = 33  # ~30 FPS par défaut
        self.mutex = QMutex()
        self.frame_size = (640, 480)
        self.fps = 30.0
        
        # Si une source est fournie, l'initialiser
        if source is not None:
            self.set_source(source)
    
    def set_source(self, source: Union[str, int]) -> bool:
        """
        Définit la source vidéo
        
        Args:
            source: Source vidéo (chemin de fichier, URL ou indice de caméra)
            
        Returns:
            True si réussi, False sinon
        """
        self.mutex.lock()
        
        try:
            # Fermer la capture précédente si elle existe
            if self.cap is not None:
                self.cap.release()
            
            self.source = source
            self.cap = cv2.VideoCapture(source)
            
            if not self.cap.isOpened():
                self.logger.error(f"Impossible d'ouvrir la source vidéo: {source}")
                self.error_occurred.emit(f"Impossible d'ouvrir la source vidéo: {source}")
                self.mutex.unlock()
                return False
            
            # Récupérer les propriétés vidéo
            self.frame_size = (
                int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 30.0  # Valeur par défaut si non détectée
            
            # Calculer l'intervalle par défaut
            self.interval_ms = int(1000 / self.fps)
            
            self.logger.info(f"Source vidéo définie: {source}, {self.frame_size}, {self.fps} FPS")
            self.source_changed.emit(str(source))
            self.mutex.unlock()
            return True
           
        except Exception as e:
            self.logger.error(f"Erreur lors de la définition de la source: {str(e)}")
            self.error_occurred.emit(f"Erreur: {str(e)}")
            self.mutex.unlock()
            return False
    
    

    def get_frame_size(self) -> Tuple[int, int]:
        """
        Retourne la taille de frame actuelle
        
        Returns:
            Tuple (largeur, hauteur)
        """
        return self.frame_size
    
    def get_fps(self) -> float:
        """
        Retourne les FPS de la source vidéo
        
        Returns:
            FPS
        """
        return self.fps
    
    def set_interval(self, interval_ms: int):
        """
        Définit l'intervalle entre les captures en millisecondes
        
        Args:
            interval_ms: Intervalle en millisecondes
        """
        if interval_ms <= 0:
            self.logger.warning(f"Intervalle invalide: {interval_ms} ms")
            return
        
        self.mutex.lock()
        self.interval_ms = interval_ms
        self.mutex.unlock()
        self.logger.debug(f"Intervalle de capture défini à {interval_ms} ms")
    
    def configure_camera(self, width: int = None, height: int = None, 
                         fps: int = None, exposure: int = None,
                         auto_focus: bool = None, auto_wb: bool = None) -> bool:
        """
        Configure les paramètres de la caméra
        
        Args:
            width: Largeur souhaitée
            height: Hauteur souhaitée
            fps: FPS souhaités
            exposure: Valeur d'exposition (-10 à 10)
            auto_focus: Activer l'autofocus
            auto_wb: Activer la balance automatique des blancs
            
        Returns:
            True si réussi, False sinon
        """
        if self.cap is None or not self.cap.isOpened():
            self.logger.error("Aucune caméra ouverte à configurer")
            return False
        
        self.mutex.lock()
        try:
            # Définir la résolution
            if width is not None and height is not None:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                
                # Vérifier les valeurs réelles définies
                actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.frame_size = (actual_width, actual_height)
                self.logger.info(f"Résolution de caméra définie: {actual_width}x{actual_height}")
            
            # Définir les FPS
            if fps is not None:
                self.cap.set(cv2.CAP_PROP_FPS, fps)
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.fps = actual_fps
                self.logger.info(f"FPS de caméra définis: {actual_fps}")
            
            # Définir l'exposition
            if exposure is not None:
                self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
                self.logger.info(f"Exposition de caméra définie: {exposure}")
            
            # Définir l'autofocus
            if auto_focus is not None:
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if auto_focus else 0)
                self.logger.info(f"Autofocus: {'activé' if auto_focus else 'désactivé'}")
            
            # Définir la balance des blancs auto
            if auto_wb is not None:
                self.cap.set(cv2.CAP_PROP_AUTO_WB, 1 if auto_wb else 0)
                self.logger.info(f"Balance des blancs auto: {'activée' if auto_wb else 'désactivée'}")
            
            self.mutex.unlock()
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la configuration de la caméra: {str(e)}")
            self.mutex.unlock()
            return False
    
    def get_camera_properties(self) -> dict:
        """
        Retourne les propriétés actuelles de la caméra
        
        Returns:
            Dictionnaire des propriétés
        """
        if self.cap is None or not self.cap.isOpened():
            return {}
        
        props = {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'exposure': self.cap.get(cv2.CAP_PROP_EXPOSURE),
            'auto_focus': bool(self.cap.get(cv2.CAP_PROP_AUTOFOCUS)),
            'auto_wb': bool(self.cap.get(cv2.CAP_PROP_AUTO_WB)),
            'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
            'saturation': self.cap.get(cv2.CAP_PROP_SATURATION)
        }
        
        return props
    
    def start(self, interval_ms: int = None):
        """
        Démarre le thread de capture
        
        Args:
            interval_ms: Intervalle entre les captures en millisecondes
        """
        if interval_ms is not None:
            self.interval_ms = max(1, interval_ms)
        
        # Assurer que la source est initialisée
        if self.cap is None or not self.cap.isOpened():
            if self.source is not None:
                self.set_source(self.source)
            else:
                self.logger.error("Impossible de démarrer: aucune source vidéo définie")
                self.error_occurred.emit("Aucune source vidéo définie")
                return
        
        self.running = True
        self.paused = False
        super().start()
        self.logger.info(f"Thread de capture démarré avec intervalle de {self.interval_ms} ms")
    
    def stop(self):
        """Arrête le thread de capture"""
        self.running = False
        self.wait()  # Attendre la fin du thread
        
        self.mutex.lock()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.mutex.unlock()
        
        self.logger.info("Thread de capture arrêté")
    
    def pause(self):
        """Met en pause la capture"""
        self.paused = True
        self.logger.info("Capture mise en pause")
    
    def resume(self):
        """Reprend la capture après une pause"""
        self.paused = False
        self.logger.info("Capture reprise")
    
    def run(self):
        """Méthode principale du thread"""
        while self.running:
            if not self.paused:
                self.mutex.lock()
                
                if self.cap is None or not self.cap.isOpened():
                    self.mutex.unlock()
                    self.error_occurred.emit("Erreur: Source vidéo non disponible")
                    self.running = False
                    break
                
                ret, frame = self.cap.read()
                self.mutex.unlock()
                
                if not ret:
                    # Gérer différemment selon le type de source
                    if isinstance(self.source, str) and self.source.startswith(('http://', 'https://', 'rtsp://')):
                        # Pour les flux, essayer de reconnecter
                        self.logger.warning("Perte de connexion au flux, tentative de reconnexion...")
                        self.mutex.lock()
                        if self.cap is not None:
                            self.cap.release()
                        self.cap = cv2.VideoCapture(self.source)
                        self.mutex.unlock()
                        time.sleep(1)  # Attendre avant de réessayer
                        continue
                    
                    elif isinstance(self.source, str) and os.path.exists(self.source):
                        # Pour les fichiers, signaler la fin
                        self.logger.info("Fin du fichier vidéo")
                        self.error_occurred.emit("Fin du fichier vidéo")
                        self.running = False
                        break
                    
                    else:
                        # Pour les webcams, signaler l'erreur
                        self.logger.error("Erreur de lecture de la webcam")
                        self.error_occurred.emit("Erreur de lecture de la webcam")
                        time.sleep(0.5)  # Attendre avant de réessayer
                        continue
                
                # Émettre la frame capturée
                self.frame_captured.emit(frame)
            
            # Attendre l'intervalle défini
            time.sleep(self.interval_ms / 1000.0)
    
    def get_preview_frame(self) -> Optional[np.ndarray]:
        """
        Obtient une frame d'aperçu de la source vidéo
        
        Returns:
            Frame d'aperçu ou None si non disponible
        """
        if self.cap is None or not self.cap.isOpened():
            return None
        
        self.mutex.lock()
        try:
            ret, frame = self.cap.read()
            
            # Remettre la vidéo au début pour les fichiers vidéo
            if isinstance(self.source, str) and os.path.exists(self.source) and not self.source.startswith(('http://', 'https://', 'rtsp://')):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            self.mutex.unlock()
            
            if ret:
                return frame
            else:
                return None
        except Exception as e:
            self.mutex.unlock()
            self.logger.error(f"Erreur lors de l'obtention de la frame d'aperçu: {str(e)}")
            return None
    
       

