#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module de capture vidéo pour DETECTCAM
Gère la capture vidéo dans un thread séparé
"""
import os
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
                self.cap = None
            
            self.source = source
            
            # Vérifier si la source est une URL de streaming
            is_stream = False
            if isinstance(source, str):
                is_stream = source.startswith(('http://', 'https://', 'rtsp://'))
            
            # Ajouter un délai pour les flux
            if is_stream:
                time.sleep(1.0)  # Attendre avant de se connecter à des flux
            
            # Ouvrir la source
            self.cap = cv2.VideoCapture(source)
            
            # Attendre un peu que la connexion s'établisse
            if is_stream:
                time.sleep(0.5)
            
            # Vérifier si la source est ouverte
            if not self.cap.isOpened():
                self.logger.error(f"Impossible d'ouvrir la source vidéo: {source}")
                self.error_occurred.emit(f"Impossible d'ouvrir la source vidéo: {source}")
                self.mutex.unlock()
                return False
            
            # Récupérer les propriétés vidéo
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Vérifier si les dimensions sont valides
            if width <= 0 or height <= 0:
                width, height = 640, 480  # Dimensions par défaut
                self.logger.warning(f"Dimensions invalides, utilisation de valeurs par défaut: {width}x{height}")
                
                # Essayer de définir des dimensions par défaut
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            self.frame_size = (width, height)
            
            # Récupérer les FPS
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 30.0  # Valeur par défaut si non détectée
                self.logger.warning(f"FPS invalides, utilisation de la valeur par défaut: {self.fps}")
            
            # Calculer l'intervalle par défaut
            self.interval_ms = int(1000 / self.fps)
            
            self.logger.info(f"Source vidéo définie: {source}, {self.frame_size}, {self.fps} FPS")
            self.source_changed.emit(str(source))
            self.mutex.unlock()
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la définition de la source: {str(e)}")
            self.error_occurred.emit(f"Erreur: {str(e)}")
            
            # S'assurer que cap est libéré en cas d'erreur
            if hasattr(self, 'cap') and self.cap is not None:
                try:
                    self.cap.release()
                except:
                    pass
                self.cap = None
                
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
            self.logger.warning(f"Intervalle invalide: {interval_ms} ms, utilisation de 33 ms")
            interval_ms = 33  # ~30 FPS par défaut
        
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
                if width > 0 and height > 0:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    
                    # Vérifier les valeurs réelles définies
                    actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    # Vérifier si les dimensions obtenues sont valides
                    if actual_width > 0 and actual_height > 0:
                        self.frame_size = (actual_width, actual_height)
                        self.logger.info(f"Résolution de caméra définie: {actual_width}x{actual_height}")
                    else:
                        self.logger.warning(f"Échec de définition de résolution: {width}x{height}")
                else:
                    self.logger.warning(f"Dimensions invalides: {width}x{height}")
            
            # Définir les FPS
            if fps is not None and fps > 0:
                self.cap.set(cv2.CAP_PROP_FPS, fps)
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                if actual_fps > 0:
                    self.fps = actual_fps
                    self.logger.info(f"FPS de caméra définis: {actual_fps}")
                else:
                    self.logger.warning(f"Échec de définition des FPS: {fps}")
            
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
            return {
                'width': 0,
                'height': 0,
                'fps': 0,
                'exposure': 0,
                'auto_focus': False,
                'auto_wb': False,
                'brightness': 0,
                'contrast': 0,
                'saturation': 0
            }
        
        try:
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
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des propriétés: {str(e)}")
            return {
                'width': self.frame_size[0],
                'height': self.frame_size[1],
                'fps': self.fps,
                'error': str(e)
            }
    
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
                success = self.set_source(self.source)
                if not success:
                    self.logger.error("Échec d'initialisation de la source vidéo")
                    self.error_occurred.emit("Échec d'initialisation de la source vidéo")
                    return
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
        
        # Attendre que le thread se termine
        if self.isRunning():
            self.wait(2000)  # Attendre max 2 secondes
            if self.isRunning():
                self.terminate()  # Force la terminaison si nécessaire
                self.logger.warning("Thread de capture forcé à terminer")
        
        self.mutex.lock()
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                self.logger.error(f"Erreur lors de la libération de la caméra: {str(e)}")
            finally:
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
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.running:
            if not self.paused:
                self.mutex.lock()
                
                if self.cap is None or not self.cap.isOpened():
                    self.mutex.unlock()
                    self.error_occurred.emit("Erreur: Source vidéo non disponible")
                    consecutive_errors += 1
                    
                    if consecutive_errors >= max_consecutive_errors:
                        self.logger.error(f"Trop d'erreurs consécutives ({consecutive_errors}), arrêt du thread")
                        self.running = False
                        break
                    
                    time.sleep(0.5)  # Pause avant de réessayer
                    continue
                
                try:
                    ret, frame = self.cap.read()
                    self.mutex.unlock()
                    
                    if not ret or frame is None:
                        # Gérer différemment selon le type de source
                        if isinstance(self.source, str) and self.source.startswith(('http://', 'https://', 'rtsp://')):
                            # Pour les flux, essayer de reconnecter
                            self.logger.warning("Perte de connexion au flux, tentative de reconnexion...")
                            self.mutex.lock()
                            if self.cap is not None:
                                self.cap.release()
                                self.cap = None
                            self.mutex.unlock()
                            
                            time.sleep(1)  # Attendre avant de réessayer
                            
                            # Reconnecter
                            self.mutex.lock()
                            self.cap = cv2.VideoCapture(self.source)
                            self.mutex.unlock()
                            
                            consecutive_errors += 1
                            if consecutive_errors >= max_consecutive_errors:
                                self.logger.error(f"Échec de reconnexion après {consecutive_errors} tentatives")
                                self.error_occurred.emit("Échec de reconnexion au flux")
                                self.running = False
                                break
                            
                            continue
                        
                        elif isinstance(self.source, str) and os.path.exists(self.source):
                            # Pour les fichiers, boucler ou signaler la fin
                            if os.path.exists(self.source):  # Vérifier à nouveau que le fichier existe toujours
                                self.logger.info("Fin du fichier vidéo")
                                
                                # Essayer de revenir au début
                                self.mutex.lock()
                                if self.cap is not None:
                                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                    ret, frame = self.cap.read()
                                    if not ret:
                                        self.logger.error("Impossible de boucler la vidéo")
                                        self.error_occurred.emit("Fin du fichier vidéo")
                                        self.running = False
                                        self.mutex.unlock()
                                        break
                                else:
                                    self.mutex.unlock()
                                    continue
                                self.mutex.unlock()
                            else:
                                self.logger.error("Fichier vidéo devenu inaccessible")
                                self.error_occurred.emit("Fichier vidéo inaccessible")
                                self.running = False
                                break
                        
                        else:
                            # Pour les webcams, signaler l'erreur et réessayer
                            self.logger.error("Erreur de lecture de la webcam")
                            self.error_occurred.emit("Erreur de lecture de la webcam")
                            
                            consecutive_errors += 1
                            if consecutive_errors >= max_consecutive_errors:
                                self.logger.error(f"Trop d'erreurs consécutives ({consecutive_errors}), arrêt du thread")
                                self.running = False
                                break
                            
                            time.sleep(0.5)  # Attendre avant de réessayer
                            continue
                    
                    # Réinitialiser le compteur d'erreurs si on a lu une frame avec succès
                    consecutive_errors = 0
                    
                    # Vérifier que la frame est valide
                    if frame is None or frame.size == 0:
                        self.logger.warning("Frame vide reçue")
                        continue
                    
                    # Émettre la frame capturée
                    self.frame_captured.emit(frame)
                    
                except Exception as e:
                    self.mutex.unlock()  # Assurer que le mutex est déverrouillé en cas d'exception
                    self.logger.error(f"Erreur lors de la capture: {str(e)}")
                    consecutive_errors += 1
                    
                    if consecutive_errors >= max_consecutive_errors:
                        self.error_occurred.emit(f"Erreur critique: {str(e)}")
                        self.running = False
                        break
                    
                    time.sleep(0.5)  # Pause avant de réessayer
                    continue
            
            # Attendre l'intervalle défini
            try:
                sleep_time = max(0.001, self.interval_ms / 1000.0)  # Assurer un temps positif
                time.sleep(sleep_time)
            except Exception as e:
                self.logger.error(f"Erreur pendant l'attente: {str(e)}")
    
    def get_preview_frame(self) -> Optional[np.ndarray]:
        """
        Obtient une frame d'aperçu de la source vidéo
        
        Returns:
            Frame d'aperçu ou None si non disponible
        """
        if self.cap is None:
            self.logger.error("Aucune source définie pour l'aperçu")
            return None
            
        if not self.cap.isOpened():
            self.logger.error("Source non ouverte pour l'aperçu")
            return None
        
        self.mutex.lock()
        try:
            # Essayer de lire une frame
            for _ in range(3):  # Essayer 3 fois
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    break
                time.sleep(0.1)  # Petit délai entre les tentatives
            
            # Vérifier si on a réussi à lire une frame
            if not ret or frame is None or frame.size == 0:
                self.mutex.unlock()
                self.logger.error("Impossible de lire une frame d'aperçu")
                return None
            
            # Remettre la vidéo au début pour les fichiers vidéo (pas pour les streams ou webcams)
            if isinstance(self.source, str) and os.path.exists(self.source) and not self.source.startswith(('http://', 'https://', 'rtsp://')):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            self.mutex.unlock()
            return frame
            
        except Exception as e:
            self.mutex.unlock()
            self.logger.error(f"Erreur lors de l'obtention de la frame d'aperçu: {str(e)}")
            return None
