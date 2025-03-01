#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Moteur de détection principal pour DETECTCAM
Gère la détection d'objets en temps réel avec multithreading
"""

import cv2
import numpy as np
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Callable
from PyQt6.QtCore import QObject, pyqtSignal, QThread, QMutex, QWaitCondition
from collections import deque

from core.video_capture import VideoCaptureThread
from core.object_detector import ObjectDetector
from core.recorder import VideoRecorder
from utils.logger import get_module_logger

class DetectionEngine(QObject):
    """
    Moteur de détection principal qui coordonne la capture vidéo,
    la détection d'objets et l'enregistrement vidéo
    """
    # Signaux pour communiquer avec l'interface
    frame_ready = pyqtSignal(np.ndarray, dict)  # Frame et métadonnées
    detection_occurred = pyqtSignal(np.ndarray, list)  # Frame et liste de détections
    fps_updated = pyqtSignal(float)  # FPS actuels
    recording_status = pyqtSignal(bool, float)  # État d'enregistrement et progression
    error_occurred = pyqtSignal(str)  # Message d'erreur
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le moteur de détection
        
        Args:
            config: Configuration de l'application
        """
        super().__init__()
        self.logger = get_module_logger('DetectionEngine')
        self.logger.info("Initialisation du moteur de détection")
        
        # Configuration
        self.config = config
        self.detection_config = config.get('detection', {})
        self.storage_config = config.get('storage', {})
        
        # État interne
        self.is_running = False
        self.is_paused = False
        self.is_recording = False
        self.speed_multiplier = 1.0
        self.detection_zones = []
        self.zone_sensitivity = {}
        self.last_detection_time = datetime.now()
        self.detection_counter = 0
        self.fps = 30.0
        self.frame_size = (640, 480)
        self.current_frame = None
        self.mutex = QMutex()
        
        # Tampon d'enregistrement (pour pré-enregistrement)
        buffer_size = self.detection_config.get('buffer_size', 150)
        self.recording_buffer = deque(maxlen=buffer_size)
        
        # Charger les zones de détection
        self._load_detection_zones()
        
        # Initialiser les composants
        self._init_components()
    
    def _init_components(self):
        """Initialise les composants principaux du moteur de détection"""
        # Thread de capture vidéo
        self.capture_thread = VideoCaptureThread()
        self.capture_thread.frame_captured.connect(self._process_frame)
        self.capture_thread.error_occurred.connect(self._handle_error)
        
        # Détecteur d'objets
        model_path = self.detection_config.get('model', 'yolo11m.pt')
        self.detector = ObjectDetector(
            model_path=model_path,
            conf_threshold=self.detection_config.get('conf_threshold', 0.5),
            iou_threshold=self.detection_config.get('iou_threshold', 0.45),
            use_cuda=self.detection_config.get('use_cuda', True),
            half_precision=self.detection_config.get('half_precision', True)
        )
        
        # Thread d'enregistrement vidéo
        self.recorder = VideoRecorder(
            output_dir=self.storage_config.get('videos_dir', 'detections/videos'),
            fps=self.fps,
            frame_size=self.frame_size
        )
        self.recorder.recording_progress.connect(lambda p: self.recording_status.emit(True, p))
    
    def _load_detection_zones(self):
        """Charge les zones de détection depuis la configuration"""
        zones_data = self.config.get('zones', [])
        self.detection_zones = [np.array(zone) for zone in zones_data if zone]
        
        # Charger les sensibilités des zones
        self.zone_sensitivity = self.config.get('zone_sensitivity', {})
        
        self.logger.info(f"Zones de détection chargées: {len(self.detection_zones)}")
    
    def set_source(self, source):
        """
        Définit la source vidéo
        
        Args:
            source: Chemin du fichier vidéo, URL ou indice de caméra (0 pour webcam par défaut)
        """
        if self.is_running:
            self.stop()
        
        self.capture_thread.set_source(source)
        
        # Mise à jour des propriétés
        self.frame_size = self.capture_thread.get_frame_size()
        self.fps = self.capture_thread.get_fps()
        
        # Mise à jour du recorder
        self.recorder.set_parameters(self.fps, self.frame_size)
        
        # Capturer une frame d'aperçu et s'assurer qu'elle est envoyée avant de continuer
        preview_frame = self.capture_thread.get_preview_frame()
        if preview_frame is not None:
            metadata = {
                'results': None,
                'fps': self.fps,
                'is_recording': False,
                'speed': self.speed_multiplier,
                'is_preview': True  # Marquer comme preview pour traitement spécial
            }
            self.frame_ready.emit(preview_frame, metadata)
            
            # Donner du temps au système pour afficher la frame
            time.sleep(0.1)
        
        # Maintenant démarrer le thread si on est en mode "live"
        if isinstance(source, int) or (isinstance(source, str) and (source.startswith(('http://', 'https://', 'rtsp://')))):
            self.capture_thread.start(interval_ms=int(1000 / (self.fps * self.speed_multiplier)))
        
        self.logger.info(f"Source définie: {source}, taille: {self.frame_size}, FPS: {self.fps}")
    
    def start(self):
        """Démarre le moteur de détection"""
        if self.is_running:
            return
        
        self.logger.info("Démarrage du moteur de détection")
        self.is_running = True
        self.is_paused = False
        
        # Démarrer le thread de capture
        self.capture_thread.start(interval_ms=int(1000 / (self.fps * self.speed_multiplier)))
    
    def stop(self):
        """Arrête le moteur de détection"""
        if not self.is_running:
            return
        
        self.logger.info("Arrêt du moteur de détection")
        self.is_running = False
        
        # Arrêter l'enregistrement si actif
        if self.is_recording:
            self.stop_recording()
        
        # Arrêter les threads
        self.capture_thread.stop()
        
        # Petite attente pour garantir l'arrêt complet
        time.sleep(0.1)
    
    def pause(self):
        """Met en pause le moteur de détection"""
        if not self.is_running or self.is_paused:
            return
        
        self.logger.info("Pause du moteur de détection")
        self.is_paused = True
        self.capture_thread.pause()
    
    def resume(self):
        """Reprend le moteur de détection après une pause"""
        if not self.is_running or not self.is_paused:
            return
        
        self.logger.info("Reprise du moteur de détection")
        self.is_paused = False
        self.capture_thread.resume()
    
    def set_speed(self, multiplier: float):
        """
        Définit la vitesse de lecture
        
        Args:
            multiplier: Multiplicateur de vitesse (1.0 = vitesse normale)
        """
        if multiplier <= 0:
            self.logger.warning(f"Multiplicateur de vitesse invalide: {multiplier}")
            return
        
        self.speed_multiplier = multiplier
        
        # Mettre à jour l'intervalle du thread de capture
        if self.is_running:
            new_interval = int(1000 / (self.fps * self.speed_multiplier))
            self.capture_thread.set_interval(new_interval)
        
        self.logger.info(f"Vitesse définie à {multiplier}x")
    
    def set_detection_params(self, conf_threshold: float, min_interval: int):
        """
        Définit les paramètres de détection
        
        Args:
            conf_threshold: Seuil de confiance (0.0 - 1.0)
            min_interval: Intervalle minimum entre détections (secondes)
        """
        self.detection_config['conf_threshold'] = conf_threshold
        self.detection_config['min_detection_interval'] = min_interval
        
        # Mettre à jour le détecteur
        self.detector.set_conf_threshold(conf_threshold)
        
        self.logger.info(f"Paramètres de détection mis à jour: seuil={conf_threshold}, intervalle={min_interval}s")
    
    def update_zones(self, zones, sensitivities):
        """
        Met à jour les zones de détection
        
        Args:
            zones: Liste des zones (chaque zone est un np.array de points)
            sensitivities: Dictionnaire des sensibilités par zone
        """
        self.detection_zones = zones
        self.zone_sensitivity = sensitivities
        self.logger.info(f"Zones de détection mises à jour: {len(zones)}")
    
    def _process_frame(self, frame):
        """
        Traite une frame capturée
        
        Args:
            frame: Frame capturée (np.ndarray)
        """
        if frame is None or self.is_paused:
            return
        
        # Sauvegarder la frame actuelle (protégée par mutex)
        self.mutex.lock()
        self.current_frame = frame.copy()
        self.mutex.unlock()
        
        # Ajouter au buffer d'enregistrement
        self.recording_buffer.append(frame.copy())
        
        # Si enregistrement actif, écrire la frame
        if self.is_recording:
            self.recorder.add_frame(frame)
        
        # Détection d'objets en utilisant des optimisations (redimensionnement si FastBoost)
        try:
            # Appliquer FastBoost si configuré
            processing_frame = frame.copy()
            if self.detection_config.get('fast_resize', False):
                scale_factor = 0.5
                processing_frame = cv2.resize(processing_frame, None, 
                                            fx=scale_factor, fy=scale_factor, 
                                            interpolation=cv2.INTER_NEAREST)
            
            # Détecter les objets
            results = self.detector.detect(processing_frame)
            
            # Traiter les résultats de détection
            detection_occurred, detections = self._analyze_detections(processing_frame, results)
            
            # Calculer les FPS
            self._update_fps()
            
            if results is None or len(results) == 0:
                metadata = {
                    'results': None,
                    'fps': self.fps,
                    'is_recording': self.is_recording,
                    'speed': self.speed_multiplier
                }
                self.frame_ready.emit(frame, metadata)
                return

            # Émettre la frame avec les résultats
            '''metadata = {
                'results': results,
                'fps': self.fps,
                'is_recording': self.is_recording,
                'speed': self.speed_multiplier
            }
            self.frame_ready.emit(frame, metadata)'''
            
            # Si détection, déclencher l'événement correspondant
            if detection_occurred:
                self._handle_detection(frame, detections)
        
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement de frame: {str(e)}", exc_info=True)
            self.error_occurred.emit(f"Erreur: {str(e)}")
    
    def _analyze_detections(self, frame, results):
        """
        Analyse les résultats de détection
        
        Args:
            frame: Frame analysée
            results: Résultats du détecteur
            
        Returns:
            Tuple (détection survenue, liste des détections)
        """
        # Vérifier si un intervalle minimum s'est écoulé depuis la dernière détection
        current_time = datetime.now()
        min_interval = self.detection_config.get('min_detection_interval', 2)
        time_diff = (current_time - self.last_detection_time).total_seconds()
        
        if time_diff < min_interval:
            return False, []
        
        # Récupérer le seuil de confiance et les filtres d'objets
        conf_threshold = self.detection_config.get('conf_threshold', 0.5)
        object_filters = self.detection_config.get('object_filters', [])
        class_thresholds = self.detection_config.get('class_thresholds', {})
        
        detection_occurred = False
        detections = []
        
        # Parcourir les détections
        if results is None or len(results) == 0 or not hasattr(results[0], 'boxes') or results[0].boxes is None:
            return False, []
            
        for result in results[0].boxes.data:
            # Extraire les informations
            x1, y1, x2, y2, confidence, class_id = result.cpu().numpy()
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            class_id = int(class_id)
            
            # Obtenir le nom de la classe
            try:
                class_name = self.detector.get_class_name(class_id)
            except:
                class_name = f"classe_{class_id}"
            
            # Vérifier si l'objet est dans les filtres
            if object_filters and class_name not in object_filters:
                continue
            
            # Obtenir le seuil spécifique à cette classe
            threshold = class_thresholds.get(class_name, conf_threshold)
            
            # Point central
            center_point = ((bbox[0] + bbox[2])//2, (bbox[1] + bbox[3])//2)
            
            # Vérifier les zones de détection ou l'ensemble de l'image
            is_detected = False
            zone_id = None
            
            if not self.detection_zones:
                # Toute l'image est une zone
                if confidence >= threshold:
                    is_detected = True
                    zone_id = "global"
            else:
                # Vérifier chaque zone
                for i, zone in enumerate(self.detection_zones):
                    if not isinstance(zone, np.ndarray) or zone.size == 0:
                        continue
                    
                    try:
                        if cv2.pointPolygonTest(zone, center_point, False) >= 0:
                            # Ajuster la confiance avec la sensibilité de la zone
                            sensitivity = self.zone_sensitivity.get(str(i), 50) / 100.0
                            adjusted_confidence = confidence * (1 + sensitivity)
                            
                            if adjusted_confidence >= threshold:
                                is_detected = True
                                zone_id = i
                                confidence = adjusted_confidence  # Utiliser la confiance ajustée
                                break
                    except Exception as e:
                        self.logger.error(f"Erreur lors du test de zone {i}: {str(e)}")
                        continue
            
            # Si détection confirmée
            if is_detected:
                detection_occurred = True
                
                # Informations de détection
                detection_info = {
                    'zone': zone_id,
                    'confidence': float(confidence),
                    'bbox': bbox,
                    'class_id': class_id,
                    'class_name': class_name,
                    'center': center_point,
                    'time': current_time.isoformat()
                }
                
                detections.append(detection_info)
                self.detection_counter += 1
        
        return detection_occurred, detections
    
    def _handle_detection(self, frame, detections):
        """
        Gère un événement de détection
        
        Args:
            frame: Frame avec la détection
            detections: Liste des informations de détection
        """
        self.last_detection_time = datetime.now()
        
        # Sauvegarder l'image de détection
        self._save_detection_image(frame, detections)
        
        # Démarrer l'enregistrement si configuré
        if self.detection_config.get('save_video', True) and not self.is_recording:
            self.start_recording()
        
        # Émettre le signal de détection
        self.detection_occurred.emit(frame, detections)
        
        self.logger.info(f"Détection: {len(detections)} objets détectés")
    
    def _save_detection_image(self, frame, detections):
        """
        Sauvegarde une image de détection
        
        Args:
            frame: Frame à sauvegarder
            detections: Informations de détection
        """
        from datetime import datetime
        import os
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        images_dir = self.storage_config.get('images_dir', 'detections/images')
        
        # Assurer que le répertoire existe
        os.makedirs(images_dir, exist_ok=True)
        
        # Sauvegarder l'image
        image_path = os.path.join(images_dir, f"detection_{timestamp}.jpg")
        cv2.imwrite(image_path, frame)
        
        self.logger.info(f"Image de détection sauvegardée: {image_path}")
        return image_path
    
    def start_recording(self):
        """Démarre l'enregistrement vidéo avec buffer de pré-enregistrement"""
        if self.is_recording:
            return
        
        self.logger.info("Démarrage de l'enregistrement")
        
        # Configurer le recorder
        video_duration = self.detection_config.get('video_duration', 5)
        self.recorder.start_recording(video_duration, self.recording_buffer)
        
        self.is_recording = True
        self.recording_status.emit(True, 0.0)
    
    def stop_recording(self):
        """Arrête l'enregistrement vidéo"""
        if not self.is_recording:
            return
        
        self.logger.info("Arrêt de l'enregistrement")
        
        # Finaliser l'enregistrement
        video_path = self.recorder.stop_recording()
        
        self.is_recording = False
        self.recording_status.emit(False, 0.0)
        
        return video_path
    
    def _update_fps(self):
        """Met à jour le compteur de FPS"""
        current_time = time.time()
        
        if hasattr(self, 'last_time'):
            elapsed = current_time - self.last_time
            if elapsed > 0:
                current_fps = 1.0 / elapsed
                # Lissage exponentiel
                if hasattr(self, 'fps_count'):
                    self.fps_count = 0.9 * self.fps_count + 0.1 * current_fps
                else:
                    self.fps_count = current_fps
                
                # Émettre la mise à jour toutes les 10 frames
                if not hasattr(self, 'fps_update_counter'):
                    self.fps_update_counter = 0
                
                self.fps_update_counter += 1
                if self.fps_update_counter >= 10:
                    self.fps_updated.emit(self.fps_count)
                    self.fps_update_counter = 0
        
        self.last_time = current_time
    
    def _handle_error(self, error_msg):
        """
        Gère une erreur provenant des composants
        
        Args:
            error_msg: Message d'erreur
        """
        self.logger.error(f"Erreur: {error_msg}")
        self.error_occurred.emit(error_msg)
    
    def get_current_frame(self):
        """
        Retourne la frame actuelle
        
        Returns:
            Frame actuelle ou None
        """
        self.mutex.lock()
        frame = self.current_frame.copy() if self.current_frame is not None else None
        self.mutex.unlock()
        return frame
    
    def configure_detector(self, **kwargs):
        """
        Configure le détecteur d'objets
        
        Args:
            **kwargs: Paramètres de configuration
        """
        self.detector.configure(**kwargs)
        self.logger.info(f"Détecteur reconfiguré: {kwargs}")
