#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module d'édition des zones de détection pour DETECTCAM
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from copy import deepcopy

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QListWidget, QSlider, QGroupBox, QCheckBox, QSpinBox,
    QInputDialog, QMessageBox, QMenu, QSplitter, QFrame,
    QSizePolicy, QScrollArea, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QPoint, QSize
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QMouseEvent, QAction

from utils.logger import get_module_logger

class ZoneEditor(QDialog):
    """
    Éditeur de zones de détection
    Permet de créer, modifier et supprimer des zones
    """
    
    def __init__(self, frame: np.ndarray, 
                 zones: List[np.ndarray] = None, 
                 sensitivities: Dict[str, float] = None,
                 parent=None):
        """
        Initialise l'éditeur de zones
        
        Args:
            frame: Image de fond pour l'édition
            zones: Liste des zones existantes
            sensitivities: Dictionnaire des sensibilités par zone
            parent: Widget parent
        """
        super().__init__(parent)
        self.logger = get_module_logger('UI.ZoneEditor')
        
        # Vérifier et copier le frame
        if frame is None or frame.size == 0:
            self.logger.warning("Frame vide fournie à l'éditeur de zones")
            # Créer une frame de remplacement vide
            height, width = 480, 640
            self.original_frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(
                self.original_frame, "Pas d'image disponible", (50, height//2), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
            )
        else:
            self.original_frame = frame.copy()
            
        self.display_frame = self.original_frame.copy()
        
        # Copier et vérifier les zones
        self.zones = []
        if zones is not None:
            for zone in zones:
                if isinstance(zone, np.ndarray) and zone.size >= 6:  # Au moins 3 points (x,y)
                    self.zones.append(zone.copy())
        
        # Copier et vérifier les sensibilités
        self.sensitivities = {}
        if sensitivities is not None:
            for key, value in sensitivities.items():
                if isinstance(value, (int, float)):
                    self.sensitivities[key] = float(value)
        
        # État interne
        self.current_zone = []
        self.preview_point = None
        self.is_drawing = False
        self.selected_zone_index = -1
        self.draw_grid = False
        self.zone_names = {}
        
        # Initialiser l'interface utilisateur
        self._init_ui()
        
        # Mettre à jour l'affichage initial
        self.update_display()
    
    def _init_ui(self):
        """Initialise l'interface utilisateur"""
        self.setWindowTitle("Éditeur de zones de détection")
        self.setMinimumSize(1000, 700)
        self.setModal(True)
        
        # Layout principal
        main_layout = QHBoxLayout(self)
        
        # Zone d'image à gauche (2/3 de l'espace)
        image_area = QVBoxLayout()
        
        # Image label
        self.image_label = QLabel()
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        
        # Configuration des événements de souris
        self.image_label.setMouseTracking(True)
        self.image_label.mousePressEvent = self.mouse_press_event
        self.image_label.mouseMoveEvent = self.mouse_move_event
        self.image_label.mouseReleaseEvent = self.mouse_release_event
        
        # Scroll area pour l'image
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)
        image_area.addWidget(scroll_area)
        
        # Options d'affichage
        display_options = QHBoxLayout()
        
        self.grid_check = QCheckBox("Afficher grille")
        self.grid_check.setChecked(self.draw_grid)
        self.grid_check.stateChanged.connect(self.toggle_grid)
        
        self.highlight_check = QCheckBox("Mettre en valeur la zone sélectionnée")
        self.highlight_check.setChecked(True)
        self.highlight_check.stateChanged.connect(self.update_display)
        
        display_options.addWidget(self.grid_check)
        display_options.addWidget(self.highlight_check)
        image_area.addLayout(display_options)
        
        # Zone de contrôle à droite (1/3 de l'espace)
        control_area = QVBoxLayout()
        
        # Liste des zones
        zones_group = QGroupBox("Zones de détection")
        zones_layout = QVBoxLayout()
        
        self.zones_list = QListWidget()
        self.zones_list.currentRowChanged.connect(self.select_zone)
        self.update_zones_list()
        
        zones_layout.addWidget(self.zones_list)
        
        # Boutons pour la liste de zones
        zones_buttons = QHBoxLayout()
        
        add_btn = QPushButton("Ajouter")
        add_btn.clicked.connect(self.start_drawing)
        
        edit_btn = QPushButton("Modifier")
        edit_btn.clicked.connect(self.edit_zone)
        
        delete_btn = QPushButton("Supprimer")
        delete_btn.clicked.connect(self.delete_zone)
        
        zones_buttons.addWidget(add_btn)
        zones_buttons.addWidget(edit_btn)
        zones_buttons.addWidget(delete_btn)
        
        zones_layout.addLayout(zones_buttons)
        zones_group.setLayout(zones_layout)
        
        # Propriétés de la zone sélectionnée
        properties_group = QGroupBox("Propriétés")
        properties_layout = QVBoxLayout()
        
        # Nom de la zone
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Nom:"))
        self.name_edit = QPushButton("Nommer la zone")
        self.name_edit.clicked.connect(self.rename_zone)
        name_layout.addWidget(self.name_edit)
        
        properties_layout.addLayout(name_layout)
        
        # Sensibilité
        sensitivity_layout = QVBoxLayout()
        sensitivity_layout.addWidget(QLabel("Sensibilité:"))
        
        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity_slider.setRange(0, 100)
        self.sensitivity_slider.setValue(50)
        self.sensitivity_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sensitivity_slider.setTickInterval(10)
        self.sensitivity_slider.valueChanged.connect(self.update_sensitivity)
        
        self.sensitivity_label = QLabel("50%")
        
        sensitivity_layout.addWidget(self.sensitivity_slider)
        sensitivity_layout.addWidget(self.sensitivity_label)
        
        properties_layout.addLayout(sensitivity_layout)
        
        # Types d'objets à détecter dans cette zone
        objects_layout = QVBoxLayout()
        objects_layout.addWidget(QLabel("Types d'objets à détecter:"))
        
        self.objects_combo = QComboBox()
        self.objects_combo.addItem("Tous les objets")
        self.objects_combo.addItem("Personnes uniquement")
        self.objects_combo.addItem("Véhicules uniquement")
        self.objects_combo.addItem("Animaux uniquement")
        self.objects_combo.addItem("Objets personnalisés...")
        
        objects_layout.addWidget(self.objects_combo)
        
        properties_layout.addLayout(objects_layout)
        
        # Actions automatiques
        actions_layout = QVBoxLayout()
        actions_layout.addWidget(QLabel("Actions:"))
        
        self.record_check = QCheckBox("Enregistrer vidéo")
        self.record_check.setChecked(True)
        
        self.alert_check = QCheckBox("Envoyer alerte")
        self.alert_check.setChecked(True)
        
        actions_layout.addWidget(self.record_check)
        actions_layout.addWidget(self.alert_check)
        
        properties_layout.addLayout(actions_layout)
        
        properties_group.setLayout(properties_layout)
        
        # Instructions
        instructions_group = QGroupBox("Instructions")
        instructions_layout = QVBoxLayout()
        
        instructions_text = """
        <b>Création de zone:</b>
        - Cliquez sur "Ajouter" puis sur l'image pour placer des points
        - Cliquez sur le premier point avec le bouton droit pour fermer la zone
        - La zone doit avoir au moins 3 points
        
        <b>Modification de zone:</b>
        - Sélectionnez une zone dans la liste puis cliquez sur "Modifier"
        - Ajustez les points en les faisant glisser
        
        <b>Sensibilité:</b>
        - Une sensibilité plus élevée détecte plus facilement les objets
        - Ajustez selon l'importance de la zone
        """
        
        instructions_label = QLabel(instructions_text)
        instructions_label.setWordWrap(True)
        instructions_layout.addWidget(instructions_label)
        
        instructions_group.setLayout(instructions_layout)
        
        # Ajouter les groupes au contrôle
        control_area.addWidget(zones_group)
        control_area.addWidget(properties_group)
        control_area.addWidget(instructions_group)
        control_area.addStretch()
        
        # Boutons de confirmation
        buttons_layout = QHBoxLayout()
        
        cancel_btn = QPushButton("Annuler")
        cancel_btn.clicked.connect(self.reject)
        
        save_btn = QPushButton("Enregistrer")
        save_btn.clicked.connect(self.accept)
        save_btn.setDefault(True)
        
        buttons_layout.addWidget(cancel_btn)
        buttons_layout.addWidget(save_btn)
        
        control_area.addLayout(buttons_layout)
        
        # Ajouter les zones à la mise en page principale
        main_layout.addLayout(image_area, 2)  # 2/3 de l'espace
        main_layout.addLayout(control_area, 1)  # 1/3 de l'espace
        
        self.setLayout(main_layout)
        
        # Désactiver les contrôles de propriétés au démarrage
        self.sensitivity_slider.setEnabled(False)
        self.name_edit.setEnabled(False)
    
    def update_zones_list(self):
        """Met à jour la liste des zones"""
        # Sauvegarder l'index sélectionné
        selected_index = self.zones_list.currentRow()
        
        # Effacer la liste
        self.zones_list.clear()
        
        # Ajouter les zones valides
        for i, zone in enumerate(self.zones):
            if zone is None or len(zone) < 3:
                continue
                
            # Nom de la zone (par défaut "Zone X")
            zone_name = self.zone_names.get(i, f"Zone {i+1}")
            
            # Ajouter à la liste
            self.zones_list.addItem(zone_name)
        
        # Restaurer la sélection si possible
        if selected_index >= 0 and selected_index < self.zones_list.count():
            self.zones_list.setCurrentRow(selected_index)
    
    def select_zone(self, index: int):
        """
        Sélectionne une zone
        
        Args:
            index: Index de la zone dans la liste
        """
        if index < 0 or index >= len(self.zones):
            self.selected_zone_index = -1
            self.sensitivity_slider.setEnabled(False)
            self.name_edit.setEnabled(False)
            return
        
        self.selected_zone_index = index
        
        # Mettre à jour les contrôles
        self.sensitivity_slider.setEnabled(True)
        self.name_edit.setEnabled(True)
        
        # Définir la sensibilité
        sensitivity = self.sensitivities.get(str(index), 50)
        self.sensitivity_slider.setValue(int(sensitivity))
        self.sensitivity_label.setText(f"{sensitivity}%")
        
        # Mettre à jour l'affichage
        self.update_display()
    
    def start_drawing(self):
        """Démarre le mode dessin de zone"""
        self.is_drawing = True
        self.current_zone = []
        self.update_display()
    
    def edit_zone(self):
        """Entre en mode édition pour la zone sélectionnée"""
        if self.selected_zone_index < 0:
            return
        
        # TODO: Implémenter l'édition de zone (déplacer des points)
        QMessageBox.information(self, "Édition de zone", 
                              "L'édition de zone n'est pas encore implémentée.")
    
    def delete_zone(self):
        """Supprime la zone sélectionnée"""
        if self.selected_zone_index < 0:
            return
        
        # Confirmation
        reply = QMessageBox.question(
            self, "Confirmer la suppression",
            f"Êtes-vous sûr de vouloir supprimer la zone {self.selected_zone_index+1}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                # Supprimer la zone
                if 0 <= self.selected_zone_index < len(self.zones):
                    del self.zones[self.selected_zone_index]
                    
                    # Mettre à jour les sensibilités et noms
                    new_sensitivities = {}
                    new_zone_names = {}
                    
                    for i in range(len(self.zones)):
                        old_i = i if i < self.selected_zone_index else i + 1
                        new_sensitivities[str(i)] = self.sensitivities.get(str(old_i), 50)
                        if old_i in self.zone_names:
                            new_zone_names[i] = self.zone_names[old_i]
                    
                    self.sensitivities = new_sensitivities
                    self.zone_names = new_zone_names
                    
                    # Mettre à jour l'interface
                    self.selected_zone_index = -1
                    self.update_zones_list()
                    self.update_display()
                else:
                    self.logger.error(f"Index de zone invalide: {self.selected_zone_index}")
            except Exception as e:
                self.logger.error(f"Erreur lors de la suppression de zone: {str(e)}")
                QMessageBox.critical(self, "Erreur", f"Erreur lors de la suppression: {str(e)}")
    
    def rename_zone(self):
        """Renomme la zone sélectionnée"""
        if self.selected_zone_index < 0:
            return
        
        try:
            # Obtenir le nom actuel
            current_name = self.zone_names.get(self.selected_zone_index, f"Zone {self.selected_zone_index+1}")
            
            # Boîte de dialogue pour le nouveau nom
            new_name, ok = QInputDialog.getText(
                self, "Renommer la zone", "Nouveau nom:",
                text=current_name
            )
            
            if ok and new_name:
                # Mettre à jour le nom
                self.zone_names[self.selected_zone_index] = new_name
                
                # Mettre à jour l'interface
                self.update_zones_list()
                
                # Resélectionner la zone
                self.zones_list.setCurrentRow(self.selected_zone_index)
        except Exception as e:
            self.logger.error(f"Erreur lors du renommage de zone: {str(e)}")
            QMessageBox.critical(self, "Erreur", f"Erreur lors du renommage: {str(e)}")
    
    def update_sensitivity(self, value: int):
        """
        Met à jour la sensibilité de la zone sélectionnée
        
        Args:
            value: Nouvelle valeur de sensibilité (0-100)
        """
        if self.selected_zone_index < 0:
            return
        
        try:
            # Mettre à jour la sensibilité
            self.sensitivities[str(self.selected_zone_index)] = value
            
            # Mettre à jour le label
            self.sensitivity_label.setText(f"{value}%")
            
            # Mettre à jour l'affichage
            self.update_display()
        except Exception as e:
            self.logger.error(f"Erreur lors de la mise à jour de la sensibilité: {str(e)}")
    
    def toggle_grid(self, state: int):
        """
        Active/désactive l'affichage de la grille
        
        Args:
            state: État de la case à cocher
        """
        try:
            self.draw_grid = state == Qt.CheckState.Checked
            self.update_display()
        except Exception as e:
            self.logger.error(f"Erreur lors du basculement de la grille: {str(e)}")
    
    def mouse_press_event(self, event: QMouseEvent):
        """
        Gère les événements de clic de souris
        
        Args:
            event: Événement de souris
        """
        if not self.is_drawing:
            return
        
        try:
            # Obtenir les coordonnées dans l'image
            label_pos = event.position()
            image_x, image_y = self._label_to_image_coords(int(label_pos.x()), int(label_pos.y()))
            
            if image_x is None or image_y is None:
                return
            
            # Si c'est un clic droit avec au moins 3 points, fermer la zone
            if event.button() == Qt.MouseButton.RightButton and len(self.current_zone) >= 3:
                # Vérifier si on est proche du premier point
                first_point = self.current_zone[0]
                distance = np.linalg.norm(np.array([image_x, image_y]) - np.array(first_point))
                
                if distance < 20:  # Tolérance de 20 pixels
                    # Créer un nouveau tableau numpy pour la zone
                    zone_array = np.array(self.current_zone)
                    
                    # Vérifier que la zone est valide
                    if len(zone_array) >= 3:
                        # Ajouter la zone
                        self.zones.append(zone_array)
                        new_index = len(self.zones) - 1
                        
                        # Définir la sensibilité par défaut
                        self.sensitivities[str(new_index)] = 50
                        
                        # Mettre à jour l'interface
                        self.update_zones_list()
                        self.zones_list.setCurrentRow(new_index)
                        self.selected_zone_index = new_index
                        
                        # Réinitialiser la zone courante
                        self.current_zone = []
                        self.is_drawing = False
                        
                        self.update_display()
                    return
            
            # Si c'est un clic gauche, ajouter un point
            if event.button() == Qt.MouseButton.LeftButton:
                self.current_zone.append((image_x, image_y))
                self.update_display()
        except Exception as e:
            self.logger.error(f"Erreur lors de l'événement de clic: {str(e)}")
    
    def mouse_move_event(self, event: QMouseEvent):
        """
        Gère les événements de mouvement de souris
        
        Args:
            event: Événement de souris
        """
        if not self.is_drawing or len(self.current_zone) == 0:
            return
        
        try:
            # Obtenir les coordonnées dans l'image
            label_pos = event.position()
            image_x, image_y = self._label_to_image_coords(int(label_pos.x()), int(label_pos.y()))
            
            if image_x is None or image_y is None:
                return
            
            # Mettre à jour le point de prévisualisation
            self.preview_point = (image_x, image_y)
            self.update_display()
        except Exception as e:
            self.logger.error(f"Erreur lors de l'événement de mouvement: {str(e)}")
    
    def mouse_release_event(self, event: QMouseEvent):
        """
        Gère les événements de relâchement de souris
        
        Args:
            event: Événement de souris
        """
        # Les relâchements sont gérés par mouse_press_event
        pass
    
    def _label_to_image_coords(self, label_x: int, label_y: int) -> Tuple[Optional[int], Optional[int]]:
        """
        Convertit les coordonnées du QLabel en coordonnées d'image
        
        Args:
            label_x: Coordonnée X dans le QLabel
            label_y: Coordonnée Y dans le QLabel
            
        Returns:
            Tuple (x, y) dans les coordonnées de l'image, ou (None, None) si hors limites
        """
        try:
            if not self.image_label.pixmap() or self.image_label.pixmap().isNull():
                return None, None
            
            # Obtenir les dimensions du QLabel et de l'image
            label_width = self.image_label.width()
            label_height = self.image_label.height()
            
            if label_width <= 0 or label_height <= 0:
                return None, None
                
            image_width = self.original_frame.shape[1]
            image_height = self.original_frame.shape[0]
            
            # Récupérer les dimensions du pixmap affiché
            pixmap = self.image_label.pixmap()
            pixmap_width = pixmap.width()
            pixmap_height = pixmap.height()
            
            # Vérifier les dimensions du pixmap
            if pixmap_width <= 0 or pixmap_height <= 0:
                return None, None
            
            # Calculer les ratios
            ratio_x = image_width / pixmap_width
            ratio_y = image_height / pixmap_height
            
            # Calculer l'offset pour centrer l'image
            offset_x = (label_width - pixmap_width) / 2
            offset_y = (label_height - pixmap_height) / 2
            
            # Coordonnées relatives au pixmap
            pixmap_x = label_x - offset_x
            pixmap_y = label_y - offset_y
            
            # Si les coordonnées sont hors du pixmap, retourner None
            if pixmap_x < 0 or pixmap_x >= pixmap_width or pixmap_y < 0 or pixmap_y >= pixmap_height:
                return None, None
            
            # Convertir en coordonnées d'image
            image_x = int(pixmap_x * ratio_x)
            image_y = int(pixmap_y * ratio_y)
            
            # Vérifier les limites dans l'image
            if 0 <= image_x < image_width and 0 <= image_y < image_height:
                return image_x, image_y
            else:
                return None, None
        except Exception as e:
            self.logger.error(f"Erreur lors de la conversion des coordonnées: {str(e)}")
            return None, None
    
    def update_display(self):
        """Met à jour l'affichage de l'image avec les zones"""
        try:
            if self.original_frame is None:
                return
            
            # Copier l'image originale
            display_frame = self.original_frame.copy()
            
            # Dessiner la grille si activée
            if self.draw_grid:
                self._draw_grid(display_frame)
            
            # Dessiner les zones existantes
            for i, zone in enumerate(self.zones):
                if not isinstance(zone, np.ndarray) or zone.size == 0 or len(zone) < 3:
                    continue
                
                # Déterminer la couleur selon la sensibilité et si la zone est sélectionnée
                is_selected = (i == self.selected_zone_index)
                
                # Sensibilité de 0 à 100
                sensitivity = int(self.sensitivities.get(str(i), 50))
                
                # Couleur de base: plus la sensibilité est élevée, plus la zone est verte
                # Vert pour les zones normales, bleu pour la zone sélectionnée
                if is_selected and self.highlight_check.isChecked():
                    # Bleu pour la zone sélectionnée
                    color = (255, 0, 0)  # BGR
                    thickness = 3
                else:
                    # Vert avec intensité basée sur la sensibilité
                    green_intensity = int(100 + (sensitivity / 100) * 155)  # 100 à 255
                    color = (0, green_intensity, 0)  # BGR
                    thickness = 2
                
                # Vérifier que la zone a le bon format
                try:
                    # Dessiner le polygone
                    zone_points = zone.reshape((-1, 1, 2)).astype(np.int32)
                    cv2.polylines(display_frame, [zone_points], True, color, thickness)
                    
                    # Dessiner les sommets
                    for point in zone:
                        cv2.circle(display_frame, (int(point[0]), int(point[1])), 4, color, -1)
                    
                    # Afficher le numéro et le nom de la zone
                    if len(zone) > 0:
                        center_x = int(np.mean(zone[:, 0]))
                        center_y = int(np.mean(zone[:, 1]))
                        
                        # Obtenir le nom de la zone
                        zone_name = self.zone_names.get(i, f"Zone {i+1}")
                        
                        # Vérifier que le centre est dans l'image
                        h, w = display_frame.shape[:2]
                        if 0 <= center_x < w and 0 <= center_y < h:
                            # Dessiner un fond pour le texte
                            text_size, _ = cv2.getTextSize(zone_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                            cv2.rectangle(display_frame, 
                                        (center_x - 5, center_y - text_size[1] - 5), 
                                        (center_x + text_size[0] + 5, center_y + 5), 
                                        (0, 0, 0), -1)
                            
                            # Dessiner le texte
                            cv2.putText(display_frame, zone_name, (center_x, center_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                except Exception as zone_error:
                    self.logger.error(f"Erreur lors du dessin de la zone {i}: {str(zone_error)}")
                    continue
            
            # Dessiner la zone en cours
            if self.is_drawing and len(self.current_zone) > 0:
                try:
                    # Dessiner les lignes entre les points
                    for i in range(len(self.current_zone) - 1):
                        pt1 = tuple(map(int, self.current_zone[i]))
                        pt2 = tuple(map(int, self.current_zone[i+1]))
                        cv2.line(display_frame, pt1, pt2, (255, 0, 0), 2)
                    
                    # Dessiner les points individuels
                    for i, point in enumerate(self.current_zone):
                        pt = tuple(map(int, point))
                        color = (0, 255, 0) if i == 0 else (255, 0, 0)  # Premier point en vert
                        cv2.circle(display_frame, pt, 5, color, -1)
                        cv2.putText(display_frame, str(i+1), (pt[0] + 5, pt[1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Dessiner la ligne de prévisualisation
                    if self.preview_point and len(self.current_zone) > 0:
                        last_pt = tuple(map(int, self.current_zone[-1]))
                        preview_pt = tuple(map(int, self.preview_point))
                        cv2.line(display_frame, last_pt, preview_pt, (200, 200, 200), 1)
                        
                        # Si plus de 2 points, montrer une ligne vers le premier point
                        if len(self.current_zone) > 2:
                            first_pt = tuple(map(int, self.current_zone[0]))
                            cv2.line(display_frame, preview_pt, first_pt, (200, 200, 200), 1, cv2.LINE_AA)
                    
                    # Instructions pour fermer la zone
                    if len(self.current_zone) >= 3:
                        h, w = display_frame.shape[:2]
                        text = "Cliquez sur le premier point avec le bouton droit pour fermer la zone"
                        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        
                        # Dessiner un fond pour le texte
                        cv2.rectangle(display_frame, 
                                    (10, h - 30 - text_size[1]), 
                                    (10 + text_size[0], h - 30 + text_size[1]), 
                                    (0, 0, 0), -1)
                        
                        cv2.putText(display_frame, text, (10, h - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                except Exception as current_error:
                    self.logger.error(f"Erreur lors du dessin de la zone en cours: {str(current_error)}")
            
            # Sauvegarder la frame modifiée
            self.display_frame = display_frame.copy()
            
            # Afficher l'image
            self._display_frame(display_frame)
        except Exception as e:
            self.logger.error(f"Erreur lors de la mise à jour de l'affichage: {str(e)}")
    
    def _draw_grid(self, frame: np.ndarray):
        """
        Dessine une grille sur l'image
        
        Args:
            frame: Image sur laquelle dessiner
        """
        try:
            if frame is None or frame.size == 0:
                return
                
            height, width = frame.shape[:2]
            
            # Espacement de la grille
            grid_spacing = 50
            
            # Couleur et épaisseur
            color = (200, 200, 200)  # Gris clair
            thickness = 1
            
            # Dessiner les lignes verticales
            for x in range(0, width, grid_spacing):
                cv2.line(frame, (x, 0), (x, height), color, thickness)
            
            # Dessiner les lignes horizontales
            for y in range(0, height, grid_spacing):
                cv2.line(frame, (0, y), (width, y), color, thickness)
        except Exception as e:
            self.logger.error(f"Erreur lors du dessin de la grille: {str(e)}")
    
    def _display_frame(self, frame: np.ndarray):
        """
        Affiche la frame dans le QLabel
        
        Args:
            frame: Frame à afficher
        """
        try:
            if frame is None or frame.size == 0:
                return
            
            # Convertir en RGB pour Qt
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            
            # Convertir en QImage
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            # Mettre à jour le QLabel
            self.image_label.setPixmap(QPixmap.fromImage(qt_image))
        except Exception as e:
            self.logger.error(f"Erreur lors de l'affichage de la frame: {str(e)}")
    
    def get_zones(self) -> List[np.ndarray]:
        """
        Retourne les zones définies
        
        Returns:
            Liste des zones
        """
        # Ne retourner que les zones valides
        valid_zones = []
        for zone in self.zones:
            if isinstance(zone, np.ndarray) and zone.size >= 6:  # Au moins 3 points (x,y)
                valid_zones.append(zone)
        return valid_zones
    
    def get_sensitivities(self) -> Dict[str, float]:
        """
        Retourne les sensibilités des zones
        
        Returns:
            Dictionnaire des sensibilités
        """
        return self.sensitivities
    
    def get_zone_names(self) -> Dict[int, str]:
        """
        Retourne les noms des zones
        
        Returns:
            Dictionnaire des noms
        """
        return self.zone_names
