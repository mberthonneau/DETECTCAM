#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fenêtre principale de l'application DETECTCAM
"""

import os
import cv2
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QComboBox, QSpinBox, QDoubleSpinBox, 
    QFileDialog, QCheckBox, QGroupBox, QInputDialog, QProgressBar, 
    QDialog, QLineEdit, QSlider, QMessageBox, QTabWidget, QScrollArea, 
    QMenu, QTextEdit, QRadioButton, QFormLayout, QApplication,
    QStatusBar, QToolBar, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot, QSize
from PyQt6.QtGui import QImage, QPixmap, QAction, QIcon, QKeySequence

from core.detection_engine import DetectionEngine
from ui.detection_view import DetectionView
from ui.settings_dialog import SettingsDialog
from ui.zone_editor import ZoneEditor
from ui.stats_view import StatsView
from utils.logger import get_module_logger
from config.settings import save_app_settings

class MainWindow(QMainWindow):
    """Fenêtre principale de l'application"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise la fenêtre principale
        
        Args:
            config: Configuration de l'application
        """
        super().__init__()
        self.logger = get_module_logger('UI.MainWindow')
        self.config = config
        
        # État interne
        self.is_running = False
        
        # Initialiser l'interface utilisateur
        self._init_ui()
        
        # Initialiser le moteur de détection
        self._init_detection_engine()
        
        # Configurer les connexions de signaux
        self._setup_connections()
        
        self.logger.info("Fenêtre principale initialisée")
    
    def _init_ui(self):
        """Initialise l'interface utilisateur"""
        # Configuration de base de la fenêtre
        self.setWindowTitle(f"DETECTCAM v{self.config.get('version', '0.7.0')} - Détection de mouvement")
        self.setGeometry(100, 100, 1200, 800)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Barre d'outils
        self._create_toolbar()
        
        # Panneau supérieur avec les contrôles
        top_panel = self._create_top_panel()
        main_layout.addLayout(top_panel)
        
        # Zone principale avec la vue de détection
        self.detection_view = DetectionView(self.config)
        main_layout.addWidget(self.detection_view, 1)  # Stretch factor 1
        
        # Barre de progression pour l'enregistrement vidéo
        self.recording_progress = QProgressBar()
        self.recording_progress.setVisible(False)
        main_layout.addWidget(self.recording_progress)
        
        # Barre d'état
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("Prêt")
        self.status_bar.addWidget(self.status_label, 1)
        
        # Affichage FPS
        self.fps_label = QLabel("FPS: --")
        self.status_bar.addPermanentWidget(self.fps_label)
        
        # Timer pour mise à jour des stats
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self._update_stats)
        self.stats_timer.start(1000)  # Mise à jour toutes les secondes

        # Ajouter des contrôles de lecture pour les fichiers vidéo
        self.video_controls = QWidget()
        controls_layout = QHBoxLayout(self.video_controls)
        
        self.play_btn = QPushButton("▶️ Lecture")
        self.pause_btn = QPushButton("⏸️ Pause")
        self.stop_btn = QPushButton("⏹️ Stop")
        
        self.play_btn.clicked.connect(lambda: self._control_video("play"))
        self.pause_btn.clicked.connect(lambda: self._control_video("pause"))
        self.stop_btn.clicked.connect(lambda: self._control_video("stop"))
        
        controls_layout.addWidget(self.play_btn)
        controls_layout.addWidget(self.pause_btn)
        controls_layout.addWidget(self.stop_btn)
        
        self.video_controls.setVisible(False)  # Caché par défaut
        main_layout.addWidget(self.video_controls)

    def _control_video(self, action):
        """
        Contrôle la lecture de la vidéo
        
        Args:
            action: Action à effectuer ("play", "pause", "stop")
        """
        if not hasattr(self.engine, 'capture_thread') or not self.engine.capture_thread.cap:
            return
        
        if action == "play":
            # Démarrer la lecture si elle n'est pas déjà en cours
            if not self.is_running:
                self.engine.start()
                self.is_running = True
                self.start_btn.setText("Arrêter")
                self.start_action.setText("Arrêter")
                self.status_label.setText("Lecture en cours...")
                self.detection_view.set_drawing_mode(False)
        
        elif action == "pause":
            # Mettre en pause ou reprendre
            if self.is_running:
                if self.engine.is_paused:
                    self.engine.resume()
                    self.status_label.setText("Lecture reprise")
                    self.pause_btn.setText("⏸️ Pause")
                else:
                    self.engine.pause()
                    self.status_label.setText("Lecture en pause")
                    self.pause_btn.setText("▶️ Reprendre")
        
        elif action == "stop":
            # Arrêter la lecture et revenir au début
            if self.is_running:
                self.engine.stop()
                self.is_running = False
                self.start_btn.setText("Démarrer")
                self.start_action.setText("Démarrer")
                self.status_label.setText("Lecture arrêtée")
                self.detection_view.set_drawing_mode(True)
                
                # Remettre la vidéo au début et afficher la première frame
                if hasattr(self.engine, 'source') and isinstance(self.engine.source, str):
                    # Reset la source pour revenir au début du fichier
                    self.engine.set_source(self.engine.source)


                    
    def _update_ui_for_source(self, source):
        """Met à jour l'interface selon le type de source"""
        # Afficher les contrôles vidéo uniquement pour les fichiers vidéo
        is_file = isinstance(source, str) and os.path.exists(source)
        self.video_controls.setVisible(is_file)
        
    def _create_toolbar(self):
        """Crée la barre d'outils"""
        toolbar = QToolBar("Barre d'outils principale")
        toolbar.setIconSize(QSize(32, 32))
        self.addToolBar(toolbar)
        
        # Actions principales
        # Note: Les icônes devraient être chargées depuis resources/icons/
        # Pour simplifier, nous utilisons des textes pour l'instant
        
        # Action Start/Stop
        self.start_action = QAction("Démarrer", self)
        self.start_action.setShortcut(QKeySequence("F5"))
        self.start_action.triggered.connect(self._toggle_detection)
        toolbar.addAction(self.start_action)
        
        toolbar.addSeparator()
        
        # Action pour la webcam
        webcam_action = QAction("Webcam", self)
        webcam_action.triggered.connect(self._open_webcam)
        toolbar.addAction(webcam_action)
        
        # Action pour ouvrir un fichier vidéo
        file_action = QAction("Fichier", self)
        file_action.setShortcut(QKeySequence.StandardKey.Open)
        file_action.triggered.connect(self._open_video_file)
        toolbar.addAction(file_action)
        
        toolbar.addSeparator()
        
        # Action pour les zones
        zones_action = QAction("Zones", self)
        zones_action.triggered.connect(self._edit_zones)
        toolbar.addAction(zones_action)
        
        # Action pour les paramètres
        settings_action = QAction("Paramètres", self)
        settings_action.setShortcut(QKeySequence.StandardKey.Preferences)
        settings_action.triggered.connect(self._open_settings)
        toolbar.addAction(settings_action)
        
        toolbar.addSeparator()
        
        # Action pour les statistiques
        stats_action = QAction("Statistiques", self)
        stats_action.triggered.connect(self._show_stats)
        toolbar.addAction(stats_action)
        
        # Action pour ouvrir le dossier des détections
        folder_action = QAction("Dossier Détections", self)
        folder_action.triggered.connect(self._open_detections_folder)
        toolbar.addAction(folder_action)
    
    def _create_top_panel(self):
        """
        Crée le panneau supérieur avec les contrôles
        
        Returns:
            Layout contenant les contrôles
        """
        top_panel = QHBoxLayout()
        
        # Groupe Source Vidéo
        source_group = QGroupBox("Source Vidéo")
        source_layout = QVBoxLayout()
        
        # Boutons pour la source
        self.webcam_btn = QPushButton("Webcam")
        self.webcam_btn.clicked.connect(self._open_webcam)
        
        self.file_btn = QPushButton("Fichier Vidéo")
        self.file_btn.clicked.connect(self._open_video_file)
        
        self.webcam_config_btn = QPushButton("Config. Webcam")
        self.webcam_config_btn.clicked.connect(self._configure_webcam)
        
        source_layout.addWidget(self.webcam_btn)
        source_layout.addWidget(self.file_btn)
        source_layout.addWidget(self.webcam_config_btn)
        
        source_group.setLayout(source_layout)
        
        # Groupe Paramètres
        settings_group = QGroupBox("Paramètres")
        settings_layout = QVBoxLayout()
        
        # Seuil de confiance
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confiance:"))
        
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.1, 1.0)
        self.conf_spin.setSingleStep(0.1)
        self.conf_spin.setValue(self.config.get('detection', {}).get('conf_threshold', 0.5))
        self.conf_spin.valueChanged.connect(self._update_detection_params)
        conf_layout.addWidget(self.conf_spin)
        
        settings_layout.addLayout(conf_layout)
        
        # Intervalle de détection
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("Intervalle (s):"))
        
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 60)
        self.interval_spin.setValue(self.config.get('detection', {}).get('min_detection_interval', 2))
        self.interval_spin.valueChanged.connect(self._update_detection_params)
        interval_layout.addWidget(self.interval_spin)
        
        settings_layout.addLayout(interval_layout)
        
        # Option d'enregistrement vidéo
        self.video_check = QCheckBox("Enregistrer vidéo")
        self.video_check.setChecked(self.config.get('detection', {}).get('save_video', True))
        
        video_duration_layout = QHBoxLayout()
        video_duration_layout.addWidget(QLabel("Durée (s):"))
        
        self.video_duration_spin = QSpinBox()
        self.video_duration_spin.setRange(1, 60)
        self.video_duration_spin.setValue(self.config.get('detection', {}).get('video_duration', 5))
        video_duration_layout.addWidget(self.video_duration_spin)
        
        settings_layout.addWidget(self.video_check)
        settings_layout.addLayout(video_duration_layout)
        
        settings_group.setLayout(settings_layout)
        
        # Groupe Contrôles
        controls_group = QGroupBox("Contrôles")
        controls_layout = QVBoxLayout()
        
        # Bouton pour démarrer/arrêter la détection
        self.start_btn = QPushButton("Démarrer")
        self.start_btn.clicked.connect(self._toggle_detection)
        
        # Autres boutons
        self.zones_btn = QPushButton("Zones")
        self.zones_btn.clicked.connect(self._edit_zones)
        
        self.settings_btn = QPushButton("Paramètres")
        self.settings_btn.clicked.connect(self._open_settings)
        
        self.stats_btn = QPushButton("Statistiques")
        self.stats_btn.clicked.connect(self._show_stats)
        
        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(self.zones_btn)
        controls_layout.addWidget(self.settings_btn)
        controls_layout.addWidget(self.stats_btn)
        
        controls_group.setLayout(controls_layout)
        
        # Groupe Vitesse
        speed_group = QGroupBox("Vitesse")
        speed_layout = QVBoxLayout()
        
        # Slider de vitesse
        speed_slider_layout = QHBoxLayout()
        speed_slider_layout.addWidget(QLabel("x0.5"))
        
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(5, 40)  # 0.5x à 4.0x
        self.speed_slider.setValue(10)  # 1.0x par défaut
        self.speed_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.speed_slider.setTickInterval(5)
        self.speed_slider.valueChanged.connect(self._update_speed)
        speed_slider_layout.addWidget(self.speed_slider)
        
        speed_slider_layout.addWidget(QLabel("x4.0"))
        
        self.speed_label = QLabel("Vitesse: 1.0x")
        
        # Option FastBoost
        self.fastboost_check = QCheckBox("FastBoost (moins précis, plus rapide)")
        self.fastboost_check.setChecked(self.config.get('detection', {}).get('fast_resize', False))
        self.fastboost_check.stateChanged.connect(self._toggle_fastboost)
        
        speed_layout.addLayout(speed_slider_layout)
        speed_layout.addWidget(self.speed_label)
        speed_layout.addWidget(self.fastboost_check)
        
        speed_group.setLayout(speed_layout)
        
        # Ajouter tous les groupes au panneau supérieur
        top_panel.addWidget(source_group)
        top_panel.addWidget(settings_group)
        top_panel.addWidget(controls_group)
        top_panel.addWidget(speed_group)
        
        return top_panel
    
    def _init_detection_engine(self):
        """Initialise le moteur de détection"""
        self.engine = DetectionEngine(self.config)
        
        # Connecter les signaux du moteur
        self.engine.frame_ready.connect(self.detection_view.update_display)
        self.engine.detection_occurred.connect(self._handle_detection)
        self.engine.fps_updated.connect(self._update_fps)
        self.engine.recording_status.connect(self._update_recording_status)
        self.engine.error_occurred.connect(self._handle_error)
    
    def _setup_connections(self):
        """Configure les connexions de signaux"""
        # Connexions pour la mise à jour des paramètres
        self.conf_spin.valueChanged.connect(self._update_detection_params)
        self.interval_spin.valueChanged.connect(self._update_detection_params)
        self.video_check.stateChanged.connect(self._update_detection_params)
        self.video_duration_spin.valueChanged.connect(self._update_detection_params)
        
        # Connexion pour la vue de détection
        self.detection_view.zone_updated.connect(self._update_detection_zones)
        
        # Autres connexions au besoin
    
    @pyqtSlot()
    def _toggle_detection(self):
        """Active ou désactive la détection"""
        if self.is_running:
            # Arrêter la détection
            self.engine.stop()
            self.is_running = False
            self.start_btn.setText("Démarrer")
            self.start_action.setText("Démarrer")
            self.status_label.setText("Détection arrêtée")
            self.detection_view.set_drawing_mode(True)  # Activer le dessin des zones
        else:
            # Démarrer la détection
            if not hasattr(self.engine, 'capture_thread') or not self.engine.capture_thread.cap:
                QMessageBox.warning(self, "Erreur", "Aucune source vidéo sélectionnée")
                return
            
            self.engine.start()
            self.is_running = True
            self.start_btn.setText("Arrêter")
            self.start_action.setText("Arrêter")
            self.status_label.setText("Détection en cours...")
            self.detection_view.set_drawing_mode(False)  # Désactiver le dessin des zones
    
    @pyqtSlot()
    def _open_webcam(self):
        """Ouvre la webcam comme source vidéo"""
        # Arrêter la détection si elle est en cours
        if self.is_running:
            self._toggle_detection()
        
        # Définir la webcam comme source
        success = self.engine.set_source(0)  # Webcam par défaut
        
        if success:
            self.status_label.setText("Webcam connectée")
            self._update_ui_for_source(0)  # 0 = webcam

            # Démarrer immédiatement la capture pour afficher l'image
            #self.engine.capture_thread.start()
        else:
            self.status_label.setText("Échec de connexion à la webcam")
            self.detection_view.reset_view()  # Reset seulement en cas d'échec
        
    
    @pyqtSlot()
    def _open_video_file(self):
        """Ouvre un fichier vidéo comme source"""
        # Arrêter la détection si elle est en cours
        if self.is_running:
            self._toggle_detection()
        
        # Ouvrir le sélecteur de fichier
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Sélectionner vidéo", "", "Videos (*.mp4 *.avi *.mkv *.mov)"
        )
        
        if file_name:
            # SUPPRIMEZ reset_view() ici car il efface la frame d'aperçu
            
            # Définir le fichier comme source
            self.engine.set_source(file_name)
            self.status_label.setText(f"Vidéo chargée: {os.path.basename(file_name)}")
            self._update_ui_for_source(file_name)
        else:
            # Réinitialiser seulement si aucun fichier n'est sélectionné
            self.detection_view.reset_view()
    
    @pyqtSlot()
    def _configure_webcam(self):
        """Ouvre la boîte de dialogue de configuration de la webcam"""
        if not hasattr(self.engine, 'capture_thread') or not self.engine.capture_thread.cap:
            QMessageBox.warning(self, "Erreur", "Aucune webcam connectée")
            return
        
        # Récupérer les propriétés actuelles
        props = self.engine.capture_thread.get_camera_properties()
        
        # Créer la boîte de dialogue
        dialog = QDialog(self)
        dialog.setWindowTitle("Configuration de la webcam")
        layout = QVBoxLayout()
        
        # Résolution
        res_group = QGroupBox("Résolution")
        res_layout = QVBoxLayout()
        
        resolutions = ["640x480", "1280x720", "1920x1080"]
        res_combo = QComboBox()
        res_combo.addItems(resolutions)
        
        # Sélectionner la résolution actuelle
        current_res = f"{props.get('width', 640)}x{props.get('height', 480)}"
        index = res_combo.findText(current_res)
        if index >= 0:
            res_combo.setCurrentIndex(index)
            
        res_layout.addWidget(res_combo)
        res_group.setLayout(res_layout)
        
        # FPS
        fps_group = QGroupBox("FPS")
        fps_layout = QVBoxLayout()
        
        fps_options = ["15", "30", "60"]
        fps_combo = QComboBox()
        fps_combo.addItems(fps_options)
        
        # Sélectionner les FPS actuels
        current_fps = str(int(props.get('fps', 30)))
        index = fps_combo.findText(current_fps)
        if index >= 0:
            fps_combo.setCurrentIndex(index)
            
        fps_layout.addWidget(fps_combo)
        fps_group.setLayout(fps_layout)
        
        # Exposition
        exposure_group = QGroupBox("Exposition")
        exposure_layout = QVBoxLayout()
        
        exposure_slider = QSlider(Qt.Orientation.Horizontal)
        exposure_slider.setRange(-10, 10)
        exposure_slider.setValue(int(props.get('exposure', 0)))
        exposure_layout.addWidget(exposure_slider)
        
        exposure_group.setLayout(exposure_layout)
        
        # Boutons
        buttons = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(dialog.accept)
        cancel_button = QPushButton("Annuler")
        cancel_button.clicked.connect(dialog.reject)
        
        buttons.addWidget(cancel_button)
        buttons.addWidget(ok_button)
        
        # Assemblage
        layout.addWidget(res_group)
        layout.addWidget(fps_group)
        layout.addWidget(exposure_group)
        layout.addLayout(buttons)
        
        dialog.setLayout(layout)
        
        # Exécuter la boîte de dialogue
        if dialog.exec():
            # Appliquer les paramètres
            selected_res = res_combo.currentText().split('x')
            width, height = int(selected_res[0]), int(selected_res[1])
            
            fps = int(fps_combo.currentText())
            exposure = exposure_slider.value()
            
            # Configurer la webcam
            self.engine.capture_thread.configure_camera(
                width=width, height=height, 
                fps=fps, exposure=exposure,
                auto_focus=True, auto_wb=True
            )
            
            self.status_label.setText(f"Webcam configurée: {width}x{height} @ {fps}fps")
    
    @pyqtSlot()
    def _edit_zones(self):
        """Ouvre l'éditeur de zones"""
        # Si la détection est en cours, la désactiver temporairement
        was_running = self.is_running
        if was_running:
            self._toggle_detection()
        
        # Récupérer la frame actuelle pour l'éditeur
        frame = self.engine.get_current_frame()
        if frame is None:
            QMessageBox.warning(self, "Erreur", "Aucune image disponible. Veuillez connecter une source vidéo.")
            return
        
        # Créer l'éditeur de zones
        zone_editor = ZoneEditor(
            frame, 
            self.config.get('zones', []), 
            self.config.get('zone_sensitivity', {})
        )
        
        if zone_editor.exec():
            # Récupérer les zones et sensibilités mises à jour
            zones = zone_editor.get_zones()
            sensitivities = zone_editor.get_sensitivities()
            
            # Mettre à jour la configuration
            self.config['zones'] = [zone.tolist() for zone in zones]
            self.config['zone_sensitivity'] = sensitivities
            
            # Sauvegarder la configuration
            save_app_settings(self.config)
            
            # Mettre à jour le moteur de détection
            self.engine.update_zones(zones, sensitivities)
            
            # Mettre à jour la vue
            self.detection_view.set_zones(zones, sensitivities)
            self.status_label.setText(f"Zones mises à jour: {len(zones)} zones définies")
        
        # Redémarrer la détection si elle était active
        if was_running:
            self._toggle_detection()
    
    @pyqtSlot()
    def _open_settings(self):
        """Ouvre la boîte de dialogue des paramètres"""
        settings_dialog = SettingsDialog(self.config, self)
        
        if settings_dialog.exec():
            # Récupérer la configuration mise à jour
            updated_config = settings_dialog.get_updated_config()
            
            # Mettre à jour la configuration
            self.config.update(updated_config)
            
            # Sauvegarder la configuration
            save_app_settings(self.config)
            
            # Mettre à jour les composants
            self._update_detection_params()
            self.detection_view.update_settings(self.config)
            
            self.status_label.setText("Paramètres mis à jour")
    
    @pyqtSlot()
    def _show_stats(self):
        """Affiche les statistiques de détection"""
        stats_view = StatsView(self.config, self)
        stats_view.exec()
    
    @pyqtSlot()
    def _open_detections_folder(self):
        """Ouvre le dossier des détections"""
        try:
            folder_path = self.config.get('storage', {}).get('base_dir', 'detections')
            
            # Assurer que le chemin est absolu
            if not os.path.isabs(folder_path):
                base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                folder_path = os.path.join(base_path, folder_path)
            
            # Ouvrir le dossier selon le système d'exploitation
            if os.path.exists(folder_path):
                if os.name == 'nt':  # Windows
                    os.startfile(folder_path)
                elif os.name == 'posix':  # Linux/Mac
                    import subprocess
                    subprocess.run(['xdg-open', folder_path])
            else:
                QMessageBox.warning(self, "Erreur", f"Dossier introuvable: {folder_path}")
                
        except Exception as e:
            self.logger.error(f"Erreur lors de l'ouverture du dossier: {str(e)}")
            QMessageBox.warning(self, "Erreur", f"Impossible d'ouvrir le dossier: {str(e)}")
    
    @pyqtSlot(int)
    def _update_speed(self, value):
        """
        Met à jour la vitesse de lecture
        
        Args:
            value: Valeur du slider (5-40, représentant 0.5x à 4.0x)
        """
        # Convertir la valeur du slider en multiplicateur
        multiplier = value / 10.0
        
        # Mise à jour de l'interface
        self.speed_label.setText(f"Vitesse: {multiplier:.1f}x")
        
        # Mise à jour du moteur
        if hasattr(self, 'engine'):
            self.engine.set_speed(multiplier)
    
    @pyqtSlot()
    def _update_detection_params(self):
        """Met à jour les paramètres de détection"""
        # Récupérer les valeurs de l'interface
        conf_threshold = self.conf_spin.value()
        min_interval = self.interval_spin.value()
        save_video = self.video_check.isChecked()
        video_duration = self.video_duration_spin.value()
        
        # Mettre à jour la configuration
        if 'detection' not in self.config:
            self.config['detection'] = {}
        
        self.config['detection']['conf_threshold'] = conf_threshold
        self.config['detection']['min_detection_interval'] = min_interval
        self.config['detection']['save_video'] = save_video
        self.config['detection']['video_duration'] = video_duration
        
        # Mettre à jour le moteur
        if hasattr(self, 'engine'):
            self.engine.set_detection_params(conf_threshold, min_interval)
    
    @pyqtSlot(int)
    def _toggle_fastboost(self, state):
        """
        Active/désactive le mode FastBoost
        
        Args:
            state: État de la case à cocher (Qt.CheckState)
        """
        is_enabled = state == Qt.CheckState.Checked
        
        # Mettre à jour la configuration
        if 'detection' not in self.config:
            self.config['detection'] = {}
        
        self.config['detection']['fast_resize'] = is_enabled
        
        # Sauvegarder la configuration
        save_app_settings(self.config)
        
        # Mettre à jour l'interface
        self.status_label.setText(f"FastBoost {'activé' if is_enabled else 'désactivé'}")
    
    @pyqtSlot(np.ndarray, list)
    def _handle_detection(self, frame, detections):
        """
        Gère un événement de détection
        
        Args:
            frame: Frame avec détection
            detections: Liste des informations de détection
        """
        # Mettre à jour le statut
        objects_text = ", ".join([d.get('class_name', 'objet') for d in detections[:3]])
        if len(detections) > 3:
            objects_text += f" et {len(detections) - 3} autres"
            
        self.status_label.setText(f"Détection: {objects_text}")
    
    @pyqtSlot(list, dict)
    def _update_detection_zones(self, zones, sensitivities):
        """
        Met à jour les zones de détection
        
        Args:
            zones: Liste des zones
            sensitivities: Dictionnaire des sensibilités
        """
        # Mettre à jour la configuration
        self.config['zones'] = [zone.tolist() for zone in zones]
        self.config['zone_sensitivity'] = sensitivities
        
        # Sauvegarder la configuration
        save_app_settings(self.config)
        
        # Mettre à jour le moteur
        if hasattr(self, 'engine'):
            self.engine.update_zones(zones, sensitivities)
    
    @pyqtSlot(float)
    def _update_fps(self, fps_value):
        """
        Met à jour l'affichage des FPS
        
        Args:
            fps_value: Valeur actuelle des FPS
        """
        self.fps_label.setText(f"FPS: {fps_value:.1f}")
    
    @pyqtSlot(bool, float)
    def _update_recording_status(self, is_recording, progress):
        """
        Met à jour l'affichage du statut d'enregistrement
        
        Args:
            is_recording: True si enregistrement en cours
            progress: Progression de l'enregistrement (0.0 - 1.0)
        """
        if is_recording:
            # Afficher et mettre à jour la barre de progression
            self.recording_progress.setVisible(True)
            self.recording_progress.setValue(int(progress * 100))
            
            # Statut clignotant pour l'enregistrement
            self.status_label.setText("⚫ ENREGISTREMENT EN COURS")
        else:
            # Masquer la barre de progression
            self.recording_progress.setVisible(False)
    
    @pyqtSlot(str)
    def _handle_error(self, error_msg):
        """
        Gère une erreur provenant du moteur
        
        Args:
            error_msg: Message d'erreur
        """
        self.logger.error(f"Erreur signalée par le moteur: {error_msg}")
        self.status_label.setText(f"Erreur: {error_msg}")
        
        # Afficher un message uniquement pour les erreurs critiques
        if "critique" in error_msg.lower() or "fatal" in error_msg.lower():
            QMessageBox.critical(self, "Erreur", error_msg)
    
    def _update_stats(self):
        """Met à jour les statistiques en temps réel"""
        # Cette méthode peut être enrichie pour afficher plus de statistiques
        pass
    
    def closeEvent(self, event):
        """Gère l'événement de fermeture de la fenêtre"""
        # Arrêter le moteur de détection
        if hasattr(self, 'engine'):
            self.engine.stop()
        
        # Sauvegarder la configuration
        save_app_settings(self.config)
        
        # Accepter l'événement de fermeture
        event.accept()
