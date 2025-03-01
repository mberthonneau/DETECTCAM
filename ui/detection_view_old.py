#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module pour la visualisation des détections
"""

import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QSizePolicy
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QPoint
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QMouseEvent

from utils.logger import get_module_logger

class DetectionView(QWidget):
    """Widget personnalisé pour afficher la vidéo et les zones de détection"""
    
    # Signaux
    zone_updated = pyqtSignal(list, dict)  # Zones et sensibilités
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise la vue de détection
        
        Args:
            config: Configuration de l'application
        """
        super().__init__()
        self.logger = get_module_logger('UI.DetectionView')
        
        # Configuration
        self.config = config
        self.display_config = config.get('display', {})
        
        # État interne
        self.current_frame = None
        self.display_frame = None
        self.detection_zones = []
        self.zone_sensitivity = {}
        self.current_zone = []
        self.preview_point = None
        self.is_drawing = False
        self.drawing_enabled = True
        self.frame_size = (640, 480)
        
        # Charger les zones depuis la configuration
        self._load_zones()
        
        # Initialiser l'interface
        self._init_ui()
    
    def _init_ui(self):
        """Initialise l'interface utilisateur"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # QLabel pour afficher la vidéo
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setMinimumSize(640, 480)
        
        # Configuration des événements de souris
        self.video_label.setMouseTracking(True)
        
        layout.addWidget(self.video_label)
        
        # Définir les propriétés du widget
        self.setLayout(layout)
    
    def _load_zones(self):
        """Charge les zones depuis la configuration"""
        try:
            zones_data = self.config.get('zones', [])
            self.detection_zones = [np.array(zone) for zone in zones_data if zone]
            
            # Charger les sensibilités
            self.zone_sensitivity = self.config.get('zone_sensitivity', {})
            
            self.logger.info(f"Zones chargées: {len(self.detection_zones)}")
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement des zones: {str(e)}")
            self.detection_zones = []
            self.zone_sensitivity = {}
    
    def mousePressEvent(self, event: QMouseEvent):
        """
        Gère les clics de souris pour dessiner les zones
        
        Args:
            event: Événement de souris
        """
        if not self.drawing_enabled or not self.current_frame is not None:
            return
        
        # Convertir les coordonnées
        label_pos = event.position()
        label_x, label_y = int(label_pos.x()), int(label_pos.y())
        
        # Si le clic est sur le QLabel
        if self.video_label.rect().contains(label_x, label_y):
            # Convertir en coordonnées d'image
            image_x, image_y = self._label_to_image_coords(label_x, label_y)
            if image_x is None or image_y is None:
                return
            
            # Si c'est un clic droit et qu'il y a au moins 3 points, fermer la zone
            if event.button() == Qt.MouseButton.RightButton and len(self.current_zone) >= 3:
                # Vérifier si le clic est proche du premier point
                first_point = self.current_zone[0]
                distance = np.linalg.norm(np.array([image_x, image_y]) - np.array(first_point))
                
                if distance < 20:  # Tolérance de 20 pixels
                    # Fermer la zone et l'ajouter à la liste
                    self.detection_zones.append(np.array(self.current_zone))
                    zone_id = len(self.detection_zones) - 1
                    self.zone_sensitivity[str(zone_id)] = 50  # 50% par défaut
                    
                    # Émettre le signal de mise à jour
                    self.zone_updated.emit(self.detection_zones, self.zone_sensitivity)
                    
                    # Réinitialiser la zone courante
                    self.current_zone = []
                    self.update_display()
                    return
            
            # Si c'est un clic gauche, ajouter un point
            if event.button() == Qt.MouseButton.LeftButton:
                self.current_zone.append((image_x, image_y))
                self.update_display()
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """
        Gère les mouvements de souris pour prévisualiser les zones
        
        Args:
            event: Événement de souris
        """
        if not self.drawing_enabled or not self.current_frame is not None:
            return
        
        # Convertir les coordonnées
        label_pos = event.position()
        label_x, label_y = int(label_pos.x()), int(label_pos.y())
        
        # Si le mouvement est sur le QLabel
        if self.video_label.rect().contains(label_x, label_y):
            # Convertir en coordonnées d'image
            image_x, image_y = self._label_to_image_coords(label_x, label_y)
            if image_x is None or image_y is None:
                return
            
            # Si une zone est en cours de dessin, enregistrer le point de prévisualisation
            if len(self.current_zone) > 0:
                self.preview_point = (image_x, image_y)
                self.update_display()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """
        Gère les relâchements de souris
        
        Args:
            event: Événement de souris
        """
        # Les relâchements de souris sont gérés par mousePressEvent
        pass
    
    def _label_to_image_coords(self, label_x: int, label_y: int) -> Tuple[Optional[int], Optional[int]]:
        """
        Convertit les coordonnées du QLabel en coordonnées de l'image
        
        Args:
            label_x: Coordonnée X dans le QLabel
            label_y: Coordonnée Y dans le QLabel
            
        Returns:
            Tuple (x, y) dans les coordonnées de l'image, ou (None, None) si hors limites
        """
        if not self.video_label.pixmap() or self.video_label.pixmap().isNull():
            return None, None
        
        # Obtenir les dimensions du QLabel et de l'image
        label_width = self.video_label.width()
        label_height = self.video_label.height()
        
        if not hasattr(self, 'frame_size') or self.frame_size[0] <= 0 or self.frame_size[1] <= 0:
            if self.current_frame is not None:
                h, w = self.current_frame.shape[:2]
                self.frame_size = (w, h)
            else:
                return None, None
        
        image_width, image_height = self.frame_size
        
        # Récupérer les dimensions du pixmap affiché
        pixmap = self.video_label.pixmap()
        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()
        
        # Calculer les ratios
        ratio_x = image_width / pixmap_width
        ratio_y = image_height / pixmap_height
        
        # Calculer l'offset pour centrer l'image
        offset_x = (label_width - pixmap_width) / 2
        offset_y = (label_height - pixmap_height) / 2
        
        # Convertir les coordonnées
        image_x = int((label_x - offset_x) * ratio_x)
        image_y = int((label_y - offset_y) * ratio_y)
        
        # Vérifier les limites
        if 0 <= image_x < image_width and 0 <= image_y < image_height:
            return image_x, image_y
        else:
            return None, None
    
    @pyqtSlot(np.ndarray, dict)
    def update_display(self, frame: Optional[np.ndarray] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Met à jour l'affichage avec la nouvelle frame
        
        Args:
            frame: Nouvelle frame à afficher
            metadata: Métadonnées associées à la frame
        """
        # Si une frame est fournie, la sauvegarder
        if frame is not None:
            self.current_frame = frame.copy()
            
            # Mettre à jour la taille de frame
            h, w = frame.shape[:2]
            self.frame_size = (w, h)
        
        # Si aucune frame n'est disponible, retourner
        if self.current_frame is None:
            return
        
        # Créer une copie pour l'affichage
        display_frame = self.current_frame.copy()
        
        # Dessiner les zones existantes
        self._draw_zones(display_frame)
        
        # Dessiner la zone en cours
        self._draw_current_zone(display_frame)
        
        # Dessiner les détections si disponibles
        if metadata and 'results' in metadata:
            self._draw_detections(display_frame, metadata['results'])
        
        # Ajouter les informations d'overlay
        self._add_overlay(display_frame, metadata)
        
        # Afficher la frame
        self._display_frame(display_frame)
    
    def _draw_zones(self, frame: np.ndarray):
        """
        Dessine les zones de détection existantes
        
        Args:
            frame: Frame sur laquelle dessiner
        """
        for i, zone in enumerate(self.detection_zones):
            if not isinstance(zone, np.ndarray) or zone.size == 0:
                continue
            
            # Déterminer la couleur selon la sensibilité
            sensitivity = float(self.zone_sensitivity.get(str(i), 50))
            # Vert plus intense pour les zones plus sensibles
            color = (0, int(255 * (sensitivity / 100)), 0)
            
            # Dessiner le polygone
            cv2.polylines(frame, [zone.reshape((-1, 1, 2))], True, color, 2)
            
            # Dessiner les sommets
            for point in zone:
                cv2.circle(frame, (int(point[0]), int(point[1])), 4, color, -1)
            
            # Afficher le numéro de zone au centre
            if len(zone) > 0:
                center_x = int(np.mean(zone[:, 0]))
                center_y = int(np.mean(zone[:, 1]))
                cv2.putText(frame, f"Zone {i}", (center_x, center_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    def _draw_current_zone(self, frame: np.ndarray):
        """
        Dessine la zone en cours de création
        
        Args:
            frame: Frame sur laquelle dessiner
        """
        if not self.current_zone:
            return
        
        # Dessiner les lignes entre les points
        for i in range(len(self.current_zone) - 1):
            pt1 = tuple(map(int, self.current_zone[i]))
            pt2 = tuple(map(int, self.current_zone[i+1]))
            cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
        
        # Dessiner les points individuels
        for i, point in enumerate(self.current_zone):
            pt = tuple(map(int, point))
            color = (0, 255, 0) if i == 0 else (255, 0, 0)  # Premier point en vert
            cv2.circle(frame, pt, 5, color, -1)
            cv2.putText(frame, str(i), (pt[0] + 5, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Dessiner la ligne de prévisualisation
        if self.preview_point and len(self.current_zone) > 0:
            last_pt = tuple(map(int, self.current_zone[-1]))
            preview_pt = tuple(map(int, self.preview_point))
            cv2.line(frame, last_pt, preview_pt, (200, 200, 200), 1)
            
            # Si plus de 2 points, montrer une ligne vers le premier point
            if len(self.current_zone) > 2:
                first_pt = tuple(map(int, self.current_zone[0]))
                cv2.line(frame, preview_pt, first_pt, (200, 200, 200), 1)
        
        # Instructions
        if self.drawing_enabled:
            text = "Clic gauche: ajouter un point | Clic droit: fermer la zone (min 3 points)"
            cv2.putText(frame, text, (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    def _draw_detections(self, frame: np.ndarray, results):
        """
        Dessine les détections sur la frame
        
        Args:
            frame: Frame sur laquelle dessiner
            results: Résultats de détection YOLO
        """
        # Vérifier si results est None avant d'essayer d'y accéder
        if results is None:
            return
            
        # Continuer avec la vérification des boxes
        if not hasattr(results[0], 'boxes') or results[0].boxes is None:
            return
        
            # Récupérer le seuil de confiance
        conf_threshold = self.config.get('detection', {}).get('conf_threshold', 0.5)
        
        # Récupérer les paramètres d'affichage
        show_confidence = self.display_config.get('show_confidence', True)
        show_class = self.display_config.get('show_class', True)
        highlight_detections = self.display_config.get('highlight_detections', True)
        
        # Mappage des classes
        class_mapping = {
            "person": "personne",
            "bicycle": "vélo",
            "car": "voiture",
            "motorcycle": "moto",
            "airplane": "avion",
            "bus": "bus",
            "train": "train",
            "truck": "camion",
            "boat": "bateau",
            "traffic light": "feu de circulation",
            "fire hydrant": "bouche d'incendie",
            "stop sign": "panneau stop",
            "parking meter": "parcomètre",
            "bench": "banc",
            "bird": "oiseau",
            "cat": "chat",
            "dog": "chien",
            "horse": "cheval",
            "sheep": "mouton",
            "cow": "vache",
            "elephant": "éléphant",
            "bear": "ours",
            "zebra": "zèbre",
            "giraffe": "girafe",
            "backpack": "sac à dos",
            "umbrella": "parapluie",
            "handbag": "sac à main",
            "tie": "cravate",
            "suitcase": "valise",
            # Ajouter d'autres classes au besoin
        }
        
        # Parcourir toutes les détections
        for result in results[0].boxes.data:
            # Extraire les informations
            x1, y1, x2, y2, confidence, class_id = result.cpu().numpy()
            
            # Ne dessiner que si la confiance est suffisante
            if confidence < conf_threshold:
                continue
            
            # Convertir en entiers
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            class_id = int(class_id)
            
            # Obtenir le nom de la classe
            try:
                original_class_name = results[0].names[class_id]
                class_name = class_mapping.get(original_class_name, original_class_name)
            except:
                class_name = f"classe_{class_id}"
            
            # Point central pour vérifier les zones
            center_point = ((bbox[0] + bbox[2])//2, (bbox[1] + bbox[3])//2)
            
            # Vérifier si le point est dans une zone
            in_zone = False
            zone_id = None
            
            for i, zone in enumerate(self.detection_zones):
                if not isinstance(zone, np.ndarray) or zone.size == 0:
                    continue
                
                try:
                    if cv2.pointPolygonTest(zone, center_point, False) >= 0:
                        in_zone = True
                        zone_id = i
                        break
                except:
                    continue
            
            # Couleur selon la zone
            if in_zone:
                color = (0, 0, 255)  # Rouge pour objets en zone
                thickness = 3
                zone_text = f" Z{zone_id}"
            else:
                color = (0, 165, 255)  # Orange pour objets hors zone
                thickness = 2
                zone_text = ""
            
            # Dessiner le rectangle
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
            
            # Texte d'information
            info_text = ""
            if show_class:
                info_text += class_name
            if show_confidence:
                if info_text:
                    info_text += f": {confidence:.2f}"
                else:
                    info_text += f"{confidence:.2f}"
            info_text += zone_text
            
            # Fond pour le texte
            if info_text:
                text_font = cv2.FONT_HERSHEY_SIMPLEX
                text_scale = 0.6
                text_thickness = 2
                
                text_size, _ = cv2.getTextSize(info_text, text_font, text_scale, text_thickness)
                text_w, text_h = text_size
                
                # Fond semi-transparent
                overlay = frame.copy()
                cv2.rectangle(overlay, 
                            (bbox[0], bbox[1] - text_h - 10), 
                            (bbox[0] + text_w + 10, bbox[1]), 
                            (0, 0, 0), -1)
                
                alpha = 0.7
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                
                # Texte
                cv2.putText(frame, info_text, 
                        (bbox[0] + 5, bbox[1] - 5), 
                        text_font, text_scale, (255, 255, 255), text_thickness)
            
            # Point central
            if highlight_detections:
                cv2.circle(frame, center_point, 4, color, -1)
    
    def _add_overlay(self, frame: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        """
        Ajoute des informations d'overlay à la frame
        
        Args:
            frame: Frame sur laquelle ajouter l'overlay
            metadata: Métadonnées associées à la frame
        """
        # Horodatage
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # FPS
        if metadata and 'fps' in metadata and self.display_config.get('show_fps', True):
            fps_text = f"FPS: {metadata['fps']:.1f}"
            cv2.putText(frame, fps_text, (frame.shape[1] - 150, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Informations sur les zones
        zones_text = f"Zones: {len(self.detection_zones)}"
        cv2.putText(frame, zones_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Seuil de confiance
        conf_text = f"Confiance: {self.config.get('detection', {}).get('conf_threshold', 0.5):.2f}"
        cv2.putText(frame, conf_text, (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Indication d'enregistrement
        if metadata and metadata.get('is_recording', False):
            if int(datetime.now().timestamp() * 2) % 2 == 0:  # Clignotement 2Hz
                cv2.putText(frame, "⚫ ENREGISTREMENT", (frame.shape[1] - 250, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    def _display_frame(self, frame: np.ndarray):
        """
        Affiche la frame dans le QLabel
        
        Args:
            frame: Frame à afficher
        """
        if frame is None:
            return
        
        # Convertir en RGB pour Qt
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        
        # Appliquer le redimensionnement selon le mode choisi
        resize_mode = self.display_config.get('resize_mode', 'fit')
     
        # Convertir en QImage
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Mettre à jour le QLabel
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

        
        # Obtenir les dimensions du QLabel
        label_width = self.video_label.width()
        label_height = self.video_label.height()
        
        if resize_mode == 'original':
            # Taille originale
            display_w, display_h = w, h
            
        elif resize_mode == 'fit':
            # Adapter à la fenêtre en conservant le ratio
            ratio_w = label_width / w
            ratio_h = label_height / h
            ratio = min(ratio_w, ratio_h)
            display_w = int(w * ratio)
            display_h = int(h * ratio)
            
        elif resize_mode == 'fill':
            # Remplir la fenêtre
            display_w = label_width
            display_h = label_height
            
        elif resize_mode == 'custom':
            # Taille personnalisée
            display_w = self.display_config.get('custom_width', w)
            display_h = self.display_config.get('custom_height', h)
            
        elif resize_mode == 'percent':
            # Pourcentage de l'original
            percent = self.display_config.get('resize_percent', 100) / 100.0
            display_w = int(w * percent)
            display_h = int(h * percent)
            
        else:
            # Mode par défaut: adapter à la taille du QLabel
            display_w, display_h = label_width, label_height
        
        # Redimensionner l'image
        if display_w != w or display_h != h:
            interpolation = cv2.INTER_NEAREST if self.display_config.get('fast_resize', True) else cv2.INTER_AREA
            rgb_frame = cv2.resize(rgb_frame, (display_w, display_h), interpolation=interpolation)
        
        # Convertir en QImage
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Mettre à jour les dimensions du QLabel si nécessaire
        if self.display_config.get('auto_resize_label', True):
            self.video_label.setFixedSize(w, h)
        
        # Afficher l'image
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))
        
        # Sauvegarder la frame d'affichage
        self.display_frame = frame.copy()
    
    def set_zones(self, zones: List[np.ndarray], sensitivities: Dict[str, float]):
        """
        Définit les zones de détection
        
        Args:
            zones: Liste des zones
            sensitivities: Dictionnaire des sensibilités
        """
        self.detection_zones = zones
        self.zone_sensitivity = sensitivities
        self.update_display()
    
    def clear_zones(self):
        """Efface toutes les zones"""
        self.detection_zones = []
        self.zone_sensitivity = {}
        self.current_zone = []
        self.preview_point = None
        self.update_display()
    
    def set_drawing_mode(self, enabled: bool):
        """
        Active ou désactive le mode dessin
        
        Args:
            enabled: True pour activer, False pour désactiver
        """
        self.drawing_enabled = enabled
        self.update_display()
    
    def reset_view(self):
        """Réinitialise la vue"""
        self.current_frame = None
        self.display_frame = None
        self.preview_point = None
        self.video_label.clear()
    
    def update_settings(self, config: Dict[str, Any]):
        """
        Met à jour les paramètres d'affichage
        
        Args:
            config: Nouvelle configuration
        """
        self.config = config
        self.display_config = config.get('display', {})
        self.update_display()
