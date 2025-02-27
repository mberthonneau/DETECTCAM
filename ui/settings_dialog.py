#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module de boîte de dialogue des paramètres pour DETECTCAM
"""

import os
import cv2
import torch
from typing import Dict, List, Any, Optional, Tuple

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QTabWidget, QWidget, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QGroupBox, QListWidget, QListWidgetItem, QLineEdit,
    QFileDialog, QMessageBox, QFormLayout, QSlider, QColorDialog,
    QScrollArea, QFrame, QGridLayout
)
from PyQt6.QtCore import Qt, QSettings, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QIcon, QFont

from utils.logger import get_module_logger

class SettingsDialog(QDialog):
    """Boîte de dialogue des paramètres de l'application"""
    
    def __init__(self, config: Dict[str, Any], parent=None):
        """
        Initialise la boîte de dialogue des paramètres
        
        Args:
            config: Configuration actuelle de l'application
            parent: Widget parent
        """
        super().__init__(parent)
        self.logger = get_module_logger('UI.SettingsDialog')
        
        # Copier la configuration pour éviter de modifier l'original
        self.config = config.copy()
        
        # Initialiser l'interface utilisateur
        self._init_ui()
        
        # Charger les paramètres actuels
        self._load_current_settings()
    
    def _init_ui(self):
        """Initialise l'interface utilisateur"""
        self.setWindowTitle("Paramètres")
        self.setMinimumSize(800, 600)
        
        # Layout principal
        main_layout = QVBoxLayout(self)
        
        # Onglets pour les différentes sections
        self.tabs = QTabWidget()
        
        # Onglet Général
        general_tab = self._create_general_tab()
        self.tabs.addTab(general_tab, "Général")
        
        # Onglet Détection
        detection_tab = self._create_detection_tab()
        self.tabs.addTab(detection_tab, "Détection")
        
        # Onglet Affichage
        display_tab = self._create_display_tab()
        self.tabs.addTab(display_tab, "Affichage")
        
        # Onglet Alertes
        alerts_tab = self._create_alerts_tab()
        self.tabs.addTab(alerts_tab, "Alertes")
        
        # Onglet Stockage
        storage_tab = self._create_storage_tab()
        self.tabs.addTab(storage_tab, "Stockage")
        
        # Onglet Avancé
        advanced_tab = self._create_advanced_tab()
        self.tabs.addTab(advanced_tab, "Avancé")
        
        # Ajouter les onglets au layout principal
        main_layout.addWidget(self.tabs)
        
        # Boutons OK/Annuler
        buttons_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("Réinitialiser")
        self.reset_btn.clicked.connect(self._reset_settings)
        
        self.cancel_btn = QPushButton("Annuler")
        self.cancel_btn.clicked.connect(self.reject)
        
        self.ok_btn = QPushButton("OK")
        self.ok_btn.clicked.connect(self.accept)
        self.ok_btn.setDefault(True)
        
        buttons_layout.addWidget(self.reset_btn)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.cancel_btn)
        buttons_layout.addWidget(self.ok_btn)
        
        main_layout.addLayout(buttons_layout)
        
        self.setLayout(main_layout)
    
    def _create_general_tab(self) -> QWidget:
        """
        Crée l'onglet des paramètres généraux
        
        Returns:
            Widget de l'onglet
        """
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Groupe Application
        app_group = QGroupBox("Application")
        app_layout = QFormLayout()
        
        # Langue
        self.language_combo = QComboBox()
        self.language_combo.addItems(["Français", "English", "Español", "Deutsch"])
        app_layout.addRow("Langue:", self.language_combo)
        
        # Thème
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Automatique", "Clair", "Sombre"])
        app_layout.addRow("Thème:", self.theme_combo)
        
        # Démarrage automatique
        self.auto_start_check = QCheckBox("Lancer au démarrage de l'ordinateur")
        app_layout.addRow("", self.auto_start_check)
        
        # Démarrer minimisé
        self.start_minimized_check = QCheckBox("Démarrer minimisé dans la barre des tâches")
        app_layout.addRow("", self.start_minimized_check)
        
        app_group.setLayout(app_layout)
        layout.addWidget(app_group)
        
        # Groupe Performance
        perf_group = QGroupBox("Performance")
        perf_layout = QFormLayout()
        
        # Accélération matérielle
        self.hardware_accel_check = QCheckBox("Utiliser l'accélération matérielle (CUDA/MPS)")
        self.hardware_accel_check.setChecked(True)
        perf_layout.addRow("", self.hardware_accel_check)
        
        # Mode FastBoost
        self.fastboost_check = QCheckBox("Activer FastBoost par défaut (moins précis, plus rapide)")
        perf_layout.addRow("", self.fastboost_check)
        
        # Priorité
        self.priority_combo = QComboBox()
        self.priority_combo.addItems(["Normale", "Haute", "Basse"])
        perf_layout.addRow("Priorité CPU:", self.priority_combo)
        
        # Multithreading
        self.multithreading_check = QCheckBox("Utiliser le multithreading")
        self.multithreading_check.setChecked(True)
        perf_layout.addRow("", self.multithreading_check)
        
        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)
        
        # Groupe À propos
        about_group = QGroupBox("À propos")
        about_layout = QVBoxLayout()
        
        # Version
        version_label = QLabel(f"DETECTCAM v{self.config.get('version', '0.7.0')}")
        version_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        about_layout.addWidget(version_label)
        
        # Description
        description = QLabel("Système avancé de détection d'objets et de mouvement\n"
                            "Développé par Mickael BTN.")
        description.setWordWrap(True)
        about_layout.addWidget(description)
        
        about_group.setLayout(about_layout)
        layout.addWidget(about_group)
        
        # Ajouter du stretch pour remplir l'espace restant
        layout.addStretch()
        
        tab.setLayout(layout)
        return tab
    
    def _create_detection_tab(self) -> QWidget:
        """
        Crée l'onglet des paramètres de détection
        
        Returns:
            Widget de l'onglet
        """
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Groupe Modèle
        model_group = QGroupBox("Modèle de détection")
        model_layout = QFormLayout()
        
        # Sélection du modèle
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "YOLO11n (petit, rapide)",
            "YOLO11s (petit-moyen)",
            "YOLO11m (moyen, équilibré)",
            "YOLO11l (large)",
            "YOLO11x (extra large, précis)",
            "Modèle personnalisé..."
        ])
        model_layout.addRow("Modèle:", self.model_combo)
        
        # Chemin du modèle personnalisé
        self.custom_model_layout = QHBoxLayout()
        self.custom_model_path = QLineEdit()
        self.custom_model_path.setReadOnly(True)
        self.custom_model_path.setPlaceholderText("Chemin du modèle personnalisé")
        
        self.browse_model_btn = QPushButton("Parcourir...")
        self.browse_model_btn.clicked.connect(self._browse_model_file)
        
        self.custom_model_layout.addWidget(self.custom_model_path)
        self.custom_model_layout.addWidget(self.browse_model_btn)
        model_layout.addRow("", self.custom_model_layout)
        
        # Connecter l'événement de changement de modèle
        self.model_combo.currentIndexChanged.connect(self._update_model_visibility)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Groupe Paramètres de détection
        detection_group = QGroupBox("Paramètres de détection")
        detection_layout = QFormLayout()
        
        # Seuil de confiance
        self.conf_threshold_spin = QDoubleSpinBox()
        self.conf_threshold_spin.setRange(0.1, 1.0)
        self.conf_threshold_spin.setSingleStep(0.05)
        self.conf_threshold_spin.setDecimals(2)
        self.conf_threshold_spin.setValue(0.5)
        detection_layout.addRow("Seuil de confiance:", self.conf_threshold_spin)
        
        # Intervalle minimum entre les détections
        self.min_interval_spin = QSpinBox()
        self.min_interval_spin.setRange(0, 60)
        self.min_interval_spin.setSingleStep(1)
        self.min_interval_spin.setValue(2)
        self.min_interval_spin.setSuffix(" secondes")
        detection_layout.addRow("Intervalle minimum:", self.min_interval_spin)
        
        # Seuil IoU (Intersection over Union)
        self.iou_threshold_spin = QDoubleSpinBox()
        self.iou_threshold_spin.setRange(0.1, 1.0)
        self.iou_threshold_spin.setSingleStep(0.05)
        self.iou_threshold_spin.setDecimals(2)
        self.iou_threshold_spin.setValue(0.45)
        detection_layout.addRow("Seuil IoU:", self.iou_threshold_spin)
        
        # Classe d'objets à détecter
        self.objects_group = QGroupBox("Classes d'objets à détecter")
        objects_layout = QVBoxLayout()
        
        # Liste des classes principales
        self.person_check = QCheckBox("Personnes")
        self.vehicle_check = QCheckBox("Véhicules (voiture, moto, vélo, camion)")
        self.animal_check = QCheckBox("Animaux (chat, chien, oiseau, etc.)")
        self.bag_check = QCheckBox("Sacs et valises")
        
        objects_layout.addWidget(self.person_check)
        objects_layout.addWidget(self.vehicle_check)
        objects_layout.addWidget(self.animal_check)
        objects_layout.addWidget(self.bag_check)
        
        # Bouton "Plus d'objets"
        self.more_objects_btn = QPushButton("Configurer toutes les classes...")
        self.more_objects_btn.clicked.connect(self._configure_object_classes)
        objects_layout.addWidget(self.more_objects_btn)
        
        self.objects_group.setLayout(objects_layout)
        detection_layout.addRow("", self.objects_group)
        
        detection_group.setLayout(detection_layout)
        layout.addWidget(detection_group)
        
        # Groupe Enregistrement vidéo
        video_group = QGroupBox("Enregistrement vidéo")
        video_layout = QFormLayout()
        
        # Activer l'enregistrement vidéo
        self.save_video_check = QCheckBox("Enregistrer une vidéo à chaque détection")
        video_layout.addRow("", self.save_video_check)
        
        # Durée de la vidéo
        self.video_duration_spin = QSpinBox()
        self.video_duration_spin.setRange(1, 60)
        self.video_duration_spin.setValue(5)
        self.video_duration_spin.setSuffix(" secondes")
        video_layout.addRow("Durée de la vidéo:", self.video_duration_spin)
        
        # Taille du buffer de pré-enregistrement
        self.buffer_size_spin = QSpinBox()
        self.buffer_size_spin.setRange(0, 600)
        self.buffer_size_spin.setValue(150)
        self.buffer_size_spin.setSuffix(" frames")
        video_layout.addRow("Buffer de pré-enregistrement:", self.buffer_size_spin)
        
        # Enregistrer avec audio
        self.record_audio_check = QCheckBox("Enregistrer l'audio (si disponible)")
        video_layout.addRow("", self.record_audio_check)
        
        video_group.setLayout(video_layout)
        layout.addWidget(video_group)
        
        # Options avancées
        advanced_group = QGroupBox("Options avancées")
        advanced_layout = QFormLayout()
        
        # Utiliser la demi-précision
        self.half_precision_check = QCheckBox("Utiliser la demi-précision (plus rapide, moins précis)")
        advanced_layout.addRow("", self.half_precision_check)
        
        # Détection multi-échelle
        self.multi_scale_check = QCheckBox("Détection multi-échelle (plus précis, plus lent)")
        advanced_layout.addRow("", self.multi_scale_check)
        
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)
        
        tab.setLayout(layout)
        return tab
    
    def _create_display_tab(self) -> QWidget:
        """
        Crée l'onglet des paramètres d'affichage
        
        Returns:
            Widget de l'onglet
        """
        tab = QWidget()
        
        # Créer un widget défilant pour contenir tous les paramètres
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        
        # Groupe Affichage vidéo
        video_group = QGroupBox("Affichage vidéo")
        video_layout = QFormLayout()
        
        # Mode de redimensionnement
        self.resize_mode_combo = QComboBox()
        self.resize_mode_combo.addItems([
            "Taille originale (plus rapide)",
            "Adapter à la fenêtre (conserver ratio)",
            "Remplir la fenêtre",
            "Taille personnalisée",
            "Pourcentage de l'original"
        ])
        video_layout.addRow("Mode d'affichage:", self.resize_mode_combo)
        
        # Taille personnalisée
        self.custom_size_layout = QHBoxLayout()
        
        self.custom_width_spin = QSpinBox()
        self.custom_width_spin.setRange(160, 3840)
        self.custom_width_spin.setValue(640)
        
        self.custom_height_spin = QSpinBox()
        self.custom_height_spin.setRange(120, 2160)
        self.custom_height_spin.setValue(480)
        
        self.custom_size_layout.addWidget(QLabel("Largeur:"))
        self.custom_size_layout.addWidget(self.custom_width_spin)
        self.custom_size_layout.addWidget(QLabel("Hauteur:"))
        self.custom_size_layout.addWidget(self.custom_height_spin)
        
        video_layout.addRow("Taille personnalisée:", self.custom_size_layout)
        
        # Pourcentage
        self.percent_layout = QHBoxLayout()
        
        self.resize_percent_spin = QSpinBox()
        self.resize_percent_spin.setRange(10, 200)
        self.resize_percent_spin.setValue(100)
        self.resize_percent_spin.setSuffix("%")
        
        self.percent_layout.addWidget(self.resize_percent_spin)
        
        video_layout.addRow("Pourcentage:", self.percent_layout)
        
        # Redimensionnement automatique
        self.auto_resize_check = QCheckBox("Redimensionner automatiquement la fenêtre vidéo")
        video_layout.addRow("", self.auto_resize_check)
        
        # Connecter les événements
        self.resize_mode_combo.currentIndexChanged.connect(self._update_size_controls_visibility)
        
        video_group.setLayout(video_layout)
        layout.addWidget(video_group)
        
        # Groupe Éléments d'affichage
        elements_group = QGroupBox("Éléments d'affichage")
        elements_layout = QVBoxLayout()
        
        # Montrer le score de confiance
        self.show_confidence_check = QCheckBox("Afficher les scores de confiance")
        elements_layout.addWidget(self.show_confidence_check)
        
        # Montrer la classe
        self.show_class_check = QCheckBox("Afficher les noms des classes")
        elements_layout.addWidget(self.show_class_check)
        
        # Montrer les FPS
        self.show_fps_check = QCheckBox("Afficher les FPS")
        elements_layout.addWidget(self.show_fps_check)
        
        # Mettre en valeur les détections
        self.highlight_check = QCheckBox("Mettre en valeur les détections")
        elements_layout.addWidget(self.highlight_check)
        
        # Afficher les numéros de zone
        self.show_zone_numbers_check = QCheckBox("Afficher les numéros de zone")
        elements_layout.addWidget(self.show_zone_numbers_check)
        
        elements_group.setLayout(elements_layout)
        layout.addWidget(elements_group)
        
        # Groupe Performance d'affichage
        perf_group = QGroupBox("Performance d'affichage")
        perf_layout = QVBoxLayout()
        
        # Redimensionnement rapide
        self.fast_resize_check = QCheckBox("Utiliser un redimensionnement rapide (moins précis)")
        self.fast_resize_check.setToolTip("Utilise un algorithme plus rapide mais de moindre qualité")
        perf_layout.addWidget(self.fast_resize_check)
        
        # Priorité à la détection
        self.detection_priority_check = QCheckBox("Priorité à la détection (réduire qualité d'affichage si nécessaire)")
        self.detection_priority_check.setToolTip("Optimise la vitesse de détection au détriment de la qualité d'affichage")
        perf_layout.addWidget(self.detection_priority_check)
        
        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)
        
        # Groupe Couleurs
        colors_group = QGroupBox("Couleurs")
        colors_layout = QGridLayout()
        
        # Couleur des boîtes de détection
        self.detection_color_btn = QPushButton("Boîtes de détection")
        self.detection_color_btn.setStyleSheet("background-color: #FF0000;")
        self.detection_color_btn.clicked.connect(lambda: self._select_color(self.detection_color_btn))
        colors_layout.addWidget(QLabel("Détections:"), 0, 0)
        colors_layout.addWidget(self.detection_color_btn, 0, 1)
        
        # Couleur des zones
        self.zone_color_btn = QPushButton("Zones")
        self.zone_color_btn.setStyleSheet("background-color: #00FF00;")
        self.zone_color_btn.clicked.connect(lambda: self._select_color(self.zone_color_btn))
        colors_layout.addWidget(QLabel("Zones:"), 1, 0)
        colors_layout.addWidget(self.zone_color_btn, 1, 1)
        
        # Couleur du texte
        self.text_color_btn = QPushButton("Texte")
        self.text_color_btn.setStyleSheet("background-color: #FFFFFF;")
        self.text_color_btn.clicked.connect(lambda: self._select_color(self.text_color_btn))
        colors_layout.addWidget(QLabel("Texte:"), 2, 0)
        colors_layout.addWidget(self.text_color_btn, 2, 1)
        
        # Bouton de réinitialisation des couleurs
        self.reset_colors_btn = QPushButton("Réinitialiser les couleurs")
        self.reset_colors_btn.clicked.connect(self._reset_colors)
        colors_layout.addWidget(self.reset_colors_btn, 3, 0, 1, 2)
        
        colors_group.setLayout(colors_layout)
        layout.addWidget(colors_group)
        
        # Finaliser le scroll area
        scroll_area.setWidget(scroll_content)
        
        # Layout principal
        main_layout = QVBoxLayout(tab)
        main_layout.addWidget(scroll_area)
        
        return tab
    
    def _create_alerts_tab(self) -> QWidget:
        """
        Crée l'onglet des paramètres d'alerte
        
        Returns:
            Widget de l'onglet
        """
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Groupe Email
        email_group = QGroupBox("Alertes par e-mail")
        email_layout = QVBoxLayout()
        
        # Activer les alertes email
        self.email_enabled_check = QCheckBox("Activer les alertes par e-mail")
        email_layout.addWidget(self.email_enabled_check)
        
        # Forme avec les paramètres email
        email_form = QFormLayout()
        
        # Destinataire
        self.email_address_edit = QLineEdit()
        self.email_address_edit.setPlaceholderText("adresse@exemple.com")
        email_form.addRow("Destinataire:", self.email_address_edit)
        
        # Serveur SMTP
        self.smtp_server_edit = QLineEdit()
        self.smtp_server_edit.setPlaceholderText("smtp.gmail.com")
        self.smtp_server_edit.setText("smtp.gmail.com")
        email_form.addRow("Serveur SMTP:", self.smtp_server_edit)
        
        # Port SMTP
        self.smtp_port_spin = QSpinBox()
        self.smtp_port_spin.setRange(1, 65535)
        self.smtp_port_spin.setValue(587)
        email_form.addRow("Port SMTP:", self.smtp_port_spin)
        
        # Utilisateur SMTP
        self.email_user_edit = QLineEdit()
        self.email_user_edit.setPlaceholderText("votre_email@gmail.com")
        email_form.addRow("Utilisateur:", self.email_user_edit)
        
        # Mot de passe SMTP
        self.email_password_edit = QLineEdit()
        self.email_password_edit.setPlaceholderText("mot de passe ou clé d'application")
        self.email_password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        email_form.addRow("Mot de passe:", self.email_password_edit)
        
        # Bouton de test
        self.test_email_btn = QPushButton("Tester la configuration")
        self.test_email_btn.clicked.connect(self._test_email_config)
        email_form.addRow("", self.test_email_btn)
        
        email_layout.addLayout(email_form)
        email_group.setLayout(email_layout)
        layout.addWidget(email_group)
        
        # Groupe Notifications
        notif_group = QGroupBox("Notifications système")
        notif_layout = QVBoxLayout()
        
        # Activer les notifications
        self.notification_enabled_check = QCheckBox("Activer les notifications système")
        notif_layout.addWidget(self.notification_enabled_check)
        
        # Forme avec les paramètres de notification
        notif_form = QFormLayout()
        
        # Seuil d'alerte
        self.alert_threshold_spin = QSpinBox()
        self.alert_threshold_spin.setRange(1, 100)
        self.alert_threshold_spin.setValue(5)
        self.alert_threshold_spin.setToolTip("Nombre de détections requises pour déclencher une alerte")
        notif_form.addRow("Seuil d'alerte:", self.alert_threshold_spin)
        
        # Son d'alerte
        self.sound_alert_check = QCheckBox("Jouer un son lors des alertes")
        notif_form.addRow("", self.sound_alert_check)
        
        # Sélection du son
        self.sound_layout = QHBoxLayout()
        
        self.sound_file_edit = QLineEdit()
        self.sound_file_edit.setReadOnly(True)
        self.sound_file_edit.setPlaceholderText("Fichier son (.mp3, .wav)")
        
        self.browse_sound_btn = QPushButton("Parcourir...")
        self.browse_sound_btn.clicked.connect(self._browse_sound_file)
        
        self.test_sound_btn = QPushButton("Test")
        self.test_sound_btn.clicked.connect(self._test_sound)
        
        self.sound_layout.addWidget(self.sound_file_edit)
        self.sound_layout.addWidget(self.browse_sound_btn)
        self.sound_layout.addWidget(self.test_sound_btn)
        
        notif_form.addRow("Fichier son:", self.sound_layout)
        
        notif_layout.addLayout(notif_form)
        notif_group.setLayout(notif_layout)
        layout.addWidget(notif_group)
        
        # Groupe Autres alertes
        other_alerts_group = QGroupBox("Autres alertes")
        other_alerts_layout = QVBoxLayout()
        
        # Webhook (pour intégration avec d'autres services)
        self.webhook_check = QCheckBox("Utiliser un webhook")
        other_alerts_layout.addWidget(self.webhook_check)
        
        # URL du webhook
        self.webhook_layout = QHBoxLayout()
        
        self.webhook_url_edit = QLineEdit()
        self.webhook_url_edit.setPlaceholderText("https://example.com/webhook")
        
        self.test_webhook_btn = QPushButton("Test")
        self.test_webhook_btn.clicked.connect(self._test_webhook)
        
        self.webhook_layout.addWidget(self.webhook_url_edit)
        self.webhook_layout.addWidget(self.test_webhook_btn)
        
        other_alerts_form = QFormLayout()
        other_alerts_form.addRow("URL du webhook:", self.webhook_layout)
        
        other_alerts_layout.addLayout(other_alerts_form)
        other_alerts_group.setLayout(other_alerts_layout)
        layout.addWidget(other_alerts_group)
        
        # Ajouter l'espace restant
        layout.addStretch()
        
        tab.setLayout(layout)
        return tab
    
    def _create_storage_tab(self) -> QWidget:
        """
        Crée l'onglet des paramètres de stockage
        
        Returns:
            Widget de l'onglet
        """
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Groupe Dossiers
        folders_group = QGroupBox("Dossiers de stockage")
        folders_layout = QVBoxLayout()
        
        # Formulaire pour les dossiers
        folders_form = QFormLayout()
        
        # Dossier de base
        self.base_dir_layout = QHBoxLayout()
        
        self.base_dir_edit = QLineEdit()
        self.base_dir_edit.setReadOnly(True)
        self.base_dir_edit.setText("detections")
        
        self.browse_base_dir_btn = QPushButton("Parcourir...")
        self.browse_base_dir_btn.clicked.connect(lambda: self._browse_folder(self.base_dir_edit))
        
        self.base_dir_layout.addWidget(self.base_dir_edit)
        self.base_dir_layout.addWidget(self.browse_base_dir_btn)
        
        folders_form.addRow("Dossier de base:", self.base_dir_layout)
        
        # Dossier des vidéos
        self.videos_dir_layout = QHBoxLayout()
        
        self.videos_dir_edit = QLineEdit()
        self.videos_dir_edit.setReadOnly(True)
        self.videos_dir_edit.setText("detections/videos")
        
        self.browse_videos_dir_btn = QPushButton("Parcourir...")
        self.browse_videos_dir_btn.clicked.connect(lambda: self._browse_folder(self.videos_dir_edit))
        
        self.videos_dir_layout.addWidget(self.videos_dir_edit)
        self.videos_dir_layout.addWidget(self.browse_videos_dir_btn)
        
        folders_form.addRow("Dossier des vidéos:", self.videos_dir_layout)
        
        # Dossier des images
        self.images_dir_layout = QHBoxLayout()
        
        self.images_dir_edit = QLineEdit()
        self.images_dir_edit.setReadOnly(True)
        self.images_dir_edit.setText("detections/images")
        
        self.browse_images_dir_btn = QPushButton("Parcourir...")
        self.browse_images_dir_btn.clicked.connect(lambda: self._browse_folder(self.images_dir_edit))
        
        self.images_dir_layout.addWidget(self.images_dir_edit)
        self.images_dir_layout.addWidget(self.browse_images_dir_btn)
        
        folders_form.addRow("Dossier des images:", self.images_dir_layout)
        
        # Dossier des exports
        self.exports_dir_layout = QHBoxLayout()
        
        self.exports_dir_edit = QLineEdit()
        self.exports_dir_edit.setReadOnly(True)
        self.exports_dir_edit.setText("exports")
        
        self.browse_exports_dir_btn = QPushButton("Parcourir...")
        self.browse_exports_dir_btn.clicked.connect(lambda: self._browse_folder(self.exports_dir_edit))
        
        self.exports_dir_layout.addWidget(self.exports_dir_edit)
        self.exports_dir_layout.addWidget(self.browse_exports_dir_btn)
        
        folders_form.addRow("Dossier des exports:", self.exports_dir_layout)
        
        folders_layout.addLayout(folders_form)
        folders_group.setLayout(folders_layout)
        layout.addWidget(folders_group)
        
        # Groupe Gestion des fichiers
        files_group = QGroupBox("Gestion des fichiers")
        files_layout = QVBoxLayout()
        
        # Nettoyage automatique
        self.auto_cleanup_check = QCheckBox("Nettoyage automatique des fichiers anciens")
        files_layout.addWidget(self.auto_cleanup_check)
        
        # Durée de conservation
        files_form = QFormLayout()
        
        self.max_storage_days_spin = QSpinBox()
        self.max_storage_days_spin.setRange(1, 365)
        self.max_storage_days_spin.setValue(30)
        self.max_storage_days_spin.setSuffix(" jours")
        files_form.addRow("Durée de conservation maximale:", self.max_storage_days_spin)
        
        files_layout.addLayout(files_form)
        
        # Bouton de nettoyage manuel
        self.cleanup_btn = QPushButton("Nettoyer maintenant")
        self.cleanup_btn.clicked.connect(self._manual_cleanup)
        files_layout.addWidget(self.cleanup_btn)
        
        files_group.setLayout(files_layout)
        layout.addWidget(files_group)
        
        # Groupe Format d'enregistrement
        format_group = QGroupBox("Format d'enregistrement")
        format_layout = QFormLayout()
        
        # Format vidéo
        self.video_format_combo = QComboBox()
        self.video_format_combo.addItems(["MP4 (H.264)", "AVI", "MKV", "MOV"])
        format_layout.addRow("Format vidéo:", self.video_format_combo)
        
        # Qualité vidéo
        self.video_quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.video_quality_slider.setRange(1, 100)
        self.video_quality_slider.setValue(80)
        self.video_quality_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.video_quality_slider.setTickInterval(10)
        
        self.video_quality_label = QLabel("80%")
        self.video_quality_slider.valueChanged.connect(
            lambda v: self.video_quality_label.setText(f"{v}%")
        )
        
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(self.video_quality_slider)
        quality_layout.addWidget(self.video_quality_label)
        
        format_layout.addRow("Qualité vidéo:", quality_layout)
        
        # Format image
        self.image_format_combo = QComboBox()
        self.image_format_combo.addItems(["JPG", "PNG", "BMP", "TIFF"])
        format_layout.addRow("Format image:", self.image_format_combo)
        
        format_group.setLayout(format_layout)
        layout.addWidget(format_group)
        
        # Espace restant
        layout.addStretch()
        
        tab.setLayout(layout)
        return tab
    
    def _create_advanced_tab(self) -> QWidget:
        """
        Crée l'onglet des paramètres avancés
        
        Returns:
            Widget de l'onglet
        """
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Avertissement
        warning_label = QLabel("⚠️ Attention: Ces paramètres avancés peuvent affecter "
                              "les performances et la stabilité de l'application.")
        warning_label.setStyleSheet("color: red; font-weight: bold;")
        warning_label.setWordWrap(True)
        layout.addWidget(warning_label)
        
        # Groupe IA
        ai_group = QGroupBox("Paramètres IA")
        ai_layout = QFormLayout()
        
        # Taille de batch
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 64)
        self.batch_size_spin.setValue(1)
        ai_layout.addRow("Taille du batch:", self.batch_size_spin)
        
        # Taille d'inférence
        self.inference_size_layout = QHBoxLayout()
        
        self.inference_width_spin = QSpinBox()
        self.inference_width_spin.setRange(32, 1280)
        self.inference_width_spin.setValue(640)
        self.inference_width_spin.setSingleStep(32)
        
        self.inference_height_spin = QSpinBox()
        self.inference_height_spin.setRange(32, 1280)
        self.inference_height_spin.setValue(640)
        self.inference_height_spin.setSingleStep(32)
        
        self.inference_size_layout.addWidget(self.inference_width_spin)
        self.inference_size_layout.addWidget(QLabel("×"))
        self.inference_size_layout.addWidget(self.inference_height_spin)
        
        ai_layout.addRow("Taille d'inférence:", self.inference_size_layout)
        
        # Mode de tracking
        self.tracking_combo = QComboBox()
        self.tracking_combo.addItems(["Aucun", "ByteTrack", "DeepSORT"])
        ai_layout.addRow("Méthode de tracking:", self.tracking_combo)
        
        # Afficher trackID
        self.show_trackid_check = QCheckBox("Afficher les ID de tracking")
        ai_layout.addRow("", self.show_trackid_check)
        
        ai_group.setLayout(ai_layout)
        layout.addWidget(ai_group)
        
        # Groupe Système
        system_group = QGroupBox("Système")
        system_layout = QFormLayout()
        
        # Niveau de log
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.log_level_combo.setCurrentText("INFO")
        system_layout.addRow("Niveau de log:", self.log_level_combo)
        
        # Cache maximal
        self.max_cache_spin = QSpinBox()
        self.max_cache_spin.setRange(50, 10000)
        self.max_cache_spin.setValue(500)
        self.max_cache_spin.setSuffix(" MB")
        system_layout.addRow("Cache maximal:", self.max_cache_spin)
        
        # Délai de démarrage
        self.startup_delay_spin = QSpinBox()
        self.startup_delay_spin.setRange(0, 60)
        self.startup_delay_spin.setValue(0)
        self.startup_delay_spin.setSuffix(" secondes")
        system_layout.addRow("Délai de démarrage:", self.startup_delay_spin)
        
        system_group.setLayout(system_layout)
        layout.addWidget(system_group)
        
        # Groupe Développement
        dev_group = QGroupBox("Développement")
        dev_layout = QVBoxLayout()
        
        # Mode debug
        self.debug_mode_check = QCheckBox("Mode débogage (plus de logs et d'informations)")
        dev_layout.addWidget(self.debug_mode_check)
        
        # Mode expérimental
        self.experimental_check = QCheckBox("Activer les fonctionnalités expérimentales")
        dev_layout.addWidget(self.experimental_check)
        
        # Éditeur de configuration
        self.edit_config_btn = QPushButton("Éditer manuellement la configuration")
        self.edit_config_btn.clicked.connect(self._edit_config_manually)
        dev_layout.addWidget(self.edit_config_btn)
        
        dev_group.setLayout(dev_layout)
        layout.addWidget(dev_group)
        
        # Ajouter l'espace restant
        layout.addStretch()
        
        tab.setLayout(layout)
        return tab
    
    def _load_current_settings(self):
        """Charge les paramètres actuels dans l'interface"""
        # Paramètres généraux
        general_config = self.config.get('general', {})
        
        # Langue
        language = general_config.get('language', 'fr')
        if language == 'fr':
            self.language_combo.setCurrentText("Français")
        elif language == 'en':
            self.language_combo.setCurrentText("English")
        elif language == 'es':
            self.language_combo.setCurrentText("Español")
        elif language == 'de':
            self.language_combo.setCurrentText("Deutsch")
        
        # Thème
        theme = general_config.get('theme', 'auto')
        if theme == 'auto':
            self.theme_combo.setCurrentText("Automatique")
        elif theme == 'light':
            self.theme_combo.setCurrentText("Clair")
        elif theme == 'dark':
            self.theme_combo.setCurrentText("Sombre")
        
        # Démarrage
        self.auto_start_check.setChecked(general_config.get('auto_start', False))
        self.start_minimized_check.setChecked(general_config.get('start_minimized', False))
        
        # Accélération matérielle
        self.hardware_accel_check.setChecked(general_config.get('hardware_acceleration', True))
        
        # Paramètres de détection
        detection_config = self.config.get('detection', {})
        
        # Modèle
        model_path = detection_config.get('model', 'yolo11m.pt')
        if model_path == 'yolo11n.pt' or model_path == 'yolov8n.pt':
            self.model_combo.setCurrentText("YOLOv8n (petit, rapide)")
        elif model_path == 'yolo11s.pt' or model_path == 'yolov8s.pt':
            self.model_combo.setCurrentText("YOLOv8s (petit-moyen)")
        elif model_path == 'yolo11m.pt' or model_path == 'yolov8m.pt':
            self.model_combo.setCurrentText("YOLOv8m (moyen, équilibré)")
        elif model_path == 'yolo11l.pt' or model_path == 'yolov8l.pt':
            self.model_combo.setCurrentText("YOLOv8l (large)")
        elif model_path == 'yolo11x.pt' or model_path == 'yolov8x.pt':
            self.model_combo.setCurrentText("YOLOv8x (extra large, précis)")
        else:
            self.model_combo.setCurrentText("Modèle personnalisé...")
            self.custom_model_path.setText(model_path)
        
        # Mettre à jour la visibilité du champ de modèle personnalisé
        self._update_model_visibility()
        
        # Seuils
        self.conf_threshold_spin.setValue(detection_config.get('conf_threshold', 0.5))
        self.min_interval_spin.setValue(detection_config.get('min_detection_interval', 2))
        self.iou_threshold_spin.setValue(detection_config.get('iou_threshold', 0.45))
        
        # Classes d'objets
        object_filters = detection_config.get('object_filters', [])
        self.person_check.setChecked("personne" in object_filters)
        self.vehicle_check.setChecked(any(v in object_filters for v in ["voiture", "moto", "vélo", "camion", "bus"]))
        self.animal_check.setChecked(any(a in object_filters for a in ["chat", "chien", "oiseau"]))
        self.bag_check.setChecked(any(b in object_filters for b in ["sac à dos", "valise"]))
        
        # Vidéo
        self.save_video_check.setChecked(detection_config.get('save_video', True))
        self.video_duration_spin.setValue(detection_config.get('video_duration', 5))
        self.buffer_size_spin.setValue(detection_config.get('buffer_size', 150))
        self.record_audio_check.setChecked(detection_config.get('record_audio', False))
        
        # Options avancées
        self.half_precision_check.setChecked(detection_config.get('half_precision', True))
        self.multi_scale_check.setChecked(detection_config.get('multi_scale', False))
        self.fastboost_check.setChecked(detection_config.get('fast_resize', False))
        
        # Paramètres d'affichage
        display_config = self.config.get('display', {})
        
        # Mode de redimensionnement
        resize_mode = display_config.get('resize_mode', 'fit')
        if resize_mode == 'original':
            self.resize_mode_combo.setCurrentText("Taille originale (plus rapide)")
        elif resize_mode == 'fit':
            self.resize_mode_combo.setCurrentText("Adapter à la fenêtre (conserver ratio)")
        elif resize_mode == 'fill':
            self.resize_mode_combo.setCurrentText("Remplir la fenêtre")
        elif resize_mode == 'custom':
            self.resize_mode_combo.setCurrentText("Taille personnalisée")
        elif resize_mode == 'percent':
            self.resize_mode_combo.setCurrentText("Pourcentage de l'original")
        
        # Taille personnalisée
        self.custom_width_spin.setValue(display_config.get('custom_width', 640))
        self.custom_height_spin.setValue(display_config.get('custom_height', 480))
        
        # Pourcentage
        self.resize_percent_spin.setValue(display_config.get('resize_percent', 100))
        
        # Redimensionnement automatique
        self.auto_resize_check.setChecked(display_config.get('auto_resize_label', True))
        
        # Mettre à jour la visibilité des contrôles de taille
        self._update_size_controls_visibility()
        
        # Éléments d'affichage
        self.show_confidence_check.setChecked(display_config.get('show_confidence', True))
        self.show_class_check.setChecked(display_config.get('show_class', True))
        self.show_fps_check.setChecked(display_config.get('show_fps', True))
        self.highlight_check.setChecked(display_config.get('highlight_detections', True))
        self.show_zone_numbers_check.setChecked(display_config.get('show_zone_numbers', True))
        
        # Performance d'affichage
        self.fast_resize_check.setChecked(display_config.get('fast_resize', True))
        self.detection_priority_check.setChecked(display_config.get('detection_priority', True))
        
        # Paramètres d'alerte
        alerts_config = self.config.get('alerts', {})
        
        # Email
        self.email_enabled_check.setChecked(alerts_config.get('email_enabled', False))
        self.email_address_edit.setText(alerts_config.get('email_address', ''))
        self.smtp_server_edit.setText(alerts_config.get('smtp_server', 'smtp.gmail.com'))
        self.smtp_port_spin.setValue(alerts_config.get('smtp_port', 587))
        self.email_user_edit.setText(alerts_config.get('email_user', ''))
        self.email_password_edit.setText(alerts_config.get('email_password', ''))
        
        # Notifications
        self.notification_enabled_check.setChecked(alerts_config.get('notification_enabled', True))
        self.alert_threshold_spin.setValue(alerts_config.get('alert_threshold', 5))
        
        # Son
        self.sound_alert_check.setChecked(alerts_config.get('sound_alert', False))
        self.sound_file_edit.setText(alerts_config.get('sound_file', ''))
        
        # Webhook
        self.webhook_check.setChecked(alerts_config.get('webhook_enabled', False))
        self.webhook_url_edit.setText(alerts_config.get('webhook_url', ''))
        
        # Paramètres de stockage
        storage_config = self.config.get('storage', {})
        
        # Dossiers
        self.base_dir_edit.setText(storage_config.get('base_dir', 'detections'))
        self.videos_dir_edit.setText(storage_config.get('videos_dir', 'detections/videos'))
        self.images_dir_edit.setText(storage_config.get('images_dir', 'detections/images'))
        self.exports_dir_edit.setText(storage_config.get('exports_dir', 'exports'))
        
        # Gestion des fichiers
        self.auto_cleanup_check.setChecked(storage_config.get('auto_cleanup', True))
        self.max_storage_days_spin.setValue(storage_config.get('max_storage_days', 30))
        
        # Format d'enregistrement
        # Correspondance des formats vidéo
        video_format = storage_config.get('video_format', 'mp4')
        if video_format == 'mp4':
            self.video_format_combo.setCurrentText("MP4 (H.264)")
        elif video_format == 'avi':
            self.video_format_combo.setCurrentText("AVI")
        elif video_format == 'mkv':
            self.video_format_combo.setCurrentText("MKV")
        elif video_format == 'mov':
            self.video_format_combo.setCurrentText("MOV")
        
        # Qualité vidéo
        self.video_quality_slider.setValue(storage_config.get('video_quality', 80))
        
        # Format image
        image_format = storage_config.get('image_format', 'jpg')
        self.image_format_combo.setCurrentText(image_format.upper())
        
        # Paramètres avancés
        advanced_config = self.config.get('advanced', {})
        
        # IA
        self.batch_size_spin.setValue(advanced_config.get('batch_size', 1))
        self.inference_width_spin.setValue(advanced_config.get('inference_width', 640))
        self.inference_height_spin.setValue(advanced_config.get('inference_height', 640))
        
        # Tracking
        tracking_method = advanced_config.get('tracking_method', 'none')
        if tracking_method == 'none':
            self.tracking_combo.setCurrentText("Aucun")
        elif tracking_method == 'bytetrack':
            self.tracking_combo.setCurrentText("ByteTrack")
        elif tracking_method == 'deepsort':
            self.tracking_combo.setCurrentText("DeepSORT")
        
        self.show_trackid_check.setChecked(advanced_config.get('show_trackid', False))
        
        # Système
        log_level = advanced_config.get('log_level', 'INFO')
        self.log_level_combo.setCurrentText(log_level)
        
        self.max_cache_spin.setValue(advanced_config.get('max_cache', 500))
        self.startup_delay_spin.setValue(advanced_config.get('startup_delay', 0))
        
        # Développement
        self.debug_mode_check.setChecked(advanced_config.get('debug_mode', False))
        self.experimental_check.setChecked(advanced_config.get('experimental', False))
    
    def _update_model_visibility(self):
        """Met à jour la visibilité du champ de modèle personnalisé"""
        is_custom = self.model_combo.currentText() == "Modèle personnalisé..."
        self.custom_model_path.setVisible(is_custom)
        self.browse_model_btn.setVisible(is_custom)
    
    def _update_size_controls_visibility(self):
        """Met à jour la visibilité des contrôles de taille d'affichage"""
        mode = self.resize_mode_combo.currentText()
        
        # Afficher les contrôles de taille personnalisée seulement si ce mode est sélectionné
        is_custom = mode == "Taille personnalisée"
        self.custom_width_spin.setVisible(is_custom)
        self.custom_height_spin.setVisible(is_custom)
        
        # Afficher le contrôle de pourcentage seulement si ce mode est sélectionné
        is_percent = mode == "Pourcentage de l'original"
        self.resize_percent_spin.setVisible(is_percent)
    
    def _browse_model_file(self):
        """Parcourir pour sélectionner un fichier de modèle"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Sélectionner un modèle", "", "Modèles PyTorch (*.pt);;Tous les fichiers (*.*)"
        )
        if file_path:
            self.custom_model_path.setText(file_path)
    
    def _browse_folder(self, line_edit):
        """
        Parcourir pour sélectionner un dossier
        
        Args:
            line_edit: QLineEdit à mettre à jour
        """
        folder_path = QFileDialog.getExistingDirectory(self, "Sélectionner un dossier")
        if folder_path:
            line_edit.setText(folder_path)
    
    def _browse_sound_file(self):
        """Parcourir pour sélectionner un fichier son"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Sélectionner un fichier son", "", "Fichiers audio (*.mp3 *.wav *.ogg);;Tous les fichiers (*.*)"
        )
        if file_path:
            self.sound_file_edit.setText(file_path)
    
    def _select_color(self, button):
        """
        Ouvre le sélecteur de couleur
        
        Args:
            button: Bouton à mettre à jour
        """
        # Extraire la couleur actuelle du style
        current_style = button.styleSheet()
        current_color = QColor(Qt.GlobalColor.white)
        
        if "background-color:" in current_style:
            color_str = current_style.split("background-color:")[1].strip().split(";")[0]
            current_color = QColor(color_str)
        
        # Ouvrir le sélecteur de couleur
        color = QColorDialog.getColor(current_color, self, "Sélectionner une couleur")
        
        if color.isValid():
            # Mettre à jour le style du bouton
            button.setStyleSheet(f"background-color: {color.name()};")
    
    def _reset_colors(self):
        """Réinitialise les couleurs par défaut"""
        self.detection_color_btn.setStyleSheet("background-color: #FF0000;")
        self.zone_color_btn.setStyleSheet("background-color: #00FF00;")
        self.text_color_btn.setStyleSheet("background-color: #FFFFFF;")
    
    def _test_email_config(self):
        """Teste la configuration email"""
        # Récupérer les paramètres
        email_address = self.email_address_edit.text()
        smtp_server = self.smtp_server_edit.text()
        smtp_port = self.smtp_port_spin.value()
        user = self.email_user_edit.text()
        password = self.email_password_edit.text()
        
        # Vérifier que les champs sont remplis
        if not email_address or not smtp_server or not user or not password:
            QMessageBox.warning(self, "Configuration incomplète", 
                               "Veuillez remplir tous les champs pour tester la configuration email.")
            return
        
        # Afficher un message de progression
        QMessageBox.information(self, "Test de configuration", 
                              "Test de la configuration email en cours...\n\n"
                              "Cette fonctionnalité sera implémentée dans la version finale.")
    
    def _test_sound(self):
        """Teste la lecture du son d'alerte"""
        sound_file = self.sound_file_edit.text()
        
        if not sound_file or not os.path.exists(sound_file):
            QMessageBox.warning(self, "Fichier son introuvable", 
                               "Veuillez sélectionner un fichier son valide.")
            return
        
        # Afficher un message
        QMessageBox.information(self, "Test de son", 
                              "Lecture du son...\n\n"
                              "Cette fonctionnalité sera implémentée dans la version finale.")
    
    def _test_webhook(self):
        """Teste le webhook"""
        webhook_url = self.webhook_url_edit.text()
        
        if not webhook_url:
            QMessageBox.warning(self, "URL vide", 
                               "Veuillez entrer une URL de webhook.")
            return
        
        # Afficher un message
        QMessageBox.information(self, "Test de webhook", 
                              "Test du webhook en cours...\n\n"
                              "Cette fonctionnalité sera implémentée dans la version finale.")
    
    def _manual_cleanup(self):
        """Déclenche le nettoyage manuel des fichiers anciens"""
        max_age = self.max_storage_days_spin.value()
        
        reply = QMessageBox.question(
            self, "Confirmation",
            f"Voulez-vous supprimer tous les fichiers plus anciens que {max_age} jours?\n"
            "Cette opération est irréversible.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Afficher un message
            QMessageBox.information(self, "Nettoyage", 
                                  "Nettoyage en cours...\n\n"
                                  "Cette fonctionnalité sera implémentée dans la version finale.")
    
    def _edit_config_manually(self):
        """Ouvre un éditeur de texte pour modifier la configuration manuellement"""
        # Afficher un message
        QMessageBox.information(self, "Édition manuelle", 
                              "L'édition manuelle de la configuration n'est pas disponible dans cette version.")
    
    def _configure_object_classes(self):
        """Ouvre la configuration complète des classes d'objets"""
        # Afficher un message
        QMessageBox.information(self, "Configuration des classes", 
                              "La configuration complète des classes d'objets sera disponible dans la version finale.")
    
    def _reset_settings(self):
        """Réinitialise tous les paramètres à leurs valeurs par défaut"""
        reply = QMessageBox.question(
            self, "Confirmation",
            "Voulez-vous réinitialiser tous les paramètres à leurs valeurs par défaut?\n"
            "Cette action ne peut pas être annulée.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Réinitialiser la configuration
            # Pour simplifier, nous rechargeons juste la page
            self._load_current_settings()
            
            QMessageBox.information(self, "Paramètres réinitialisés", 
                                  "Tous les paramètres ont été réinitialisés à leurs valeurs par défaut.")
    
    def get_updated_config(self) -> Dict[str, Any]:
        """
        Retourne la configuration mise à jour
        
        Returns:
            Configuration mise à jour
        """
        # Copier la configuration actuelle
        updated_config = self.config.copy()
        
        # Mettre à jour les paramètres généraux
        if 'general' not in updated_config:
            updated_config['general'] = {}
        
        # Langue
        language = self.language_combo.currentText()
        if language == "Français":
            updated_config['general']['language'] = 'fr'
        elif language == "English":
            updated_config['general']['language'] = 'en'
        elif language == "Español":
            updated_config['general']['language'] = 'es'
        elif language == "Deutsch":
            updated_config['general']['language'] = 'de'
        
        # Thème
        theme = self.theme_combo.currentText()
        if theme == "Automatique":
            updated_config['general']['theme'] = 'auto'
        elif theme == "Clair":
            updated_config['general']['theme'] = 'light'
        elif theme == "Sombre":
            updated_config['general']['theme'] = 'dark'
        
        # Démarrage
        updated_config['general']['auto_start'] = self.auto_start_check.isChecked()
        updated_config['general']['start_minimized'] = self.start_minimized_check.isChecked()
        
        # Accélération matérielle
        updated_config['general']['hardware_acceleration'] = self.hardware_accel_check.isChecked()
        
        # Mettre à jour les paramètres de détection
        if 'detection' not in updated_config:
            updated_config['detection'] = {}
        
        # Modèle
        model = self.model_combo.currentText()
        if model == "YOLOv8n (petit, rapide)":
            updated_config['detection']['model'] = 'yolov8n.pt'
        elif model == "YOLOv8s (petit-moyen)":
            updated_config['detection']['model'] = 'yolov8s.pt'
        elif model == "YOLOv8m (moyen, équilibré)":
            updated_config['detection']['model'] = 'yolov8m.pt'
        elif model == "YOLOv8l (large)":
            updated_config['detection']['model'] = 'yolov8l.pt'
        elif model == "YOLOv8x (extra large, précis)":
            updated_config['detection']['model'] = 'yolov8x.pt'
        elif model == "Modèle personnalisé...":
            updated_config['detection']['model'] = self.custom_model_path.text()
        
        # Seuils
        updated_config['detection']['conf_threshold'] = self.conf_threshold_spin.value()
        updated_config['detection']['min_detection_interval'] = self.min_interval_spin.value()
        updated_config['detection']['iou_threshold'] = self.iou_threshold_spin.value()
        
        # Classes d'objets
        object_filters = []
        if self.person_check.isChecked():
            object_filters.append("personne")
        
        if self.vehicle_check.isChecked():
            object_filters.extend(["voiture", "moto", "vélo", "camion", "bus"])
        
        if self.animal_check.isChecked():
            object_filters.extend(["chat", "chien", "oiseau", "cheval", "vache"])
        
        if self.bag_check.isChecked():
            object_filters.extend(["sac à dos", "valise"])
        
        updated_config['detection']['object_filters'] = object_filters
        
        # Vidéo
        updated_config['detection']['save_video'] = self.save_video_check.isChecked()
        updated_config['detection']['video_duration'] = self.video_duration_spin.value()
        updated_config['detection']['buffer_size'] = self.buffer_size_spin.value()
        updated_config['detection']['record_audio'] = self.record_audio_check.isChecked()
        
        # Options avancées
        updated_config['detection']['half_precision'] = self.half_precision_check.isChecked()
        updated_config['detection']['multi_scale'] = self.multi_scale_check.isChecked()
        updated_config['detection']['fast_resize'] = self.fastboost_check.isChecked()
        
        # Mettre à jour les paramètres d'affichage
        if 'display' not in updated_config:
            updated_config['display'] = {}
        
        # Mode de redimensionnement
        mode = self.resize_mode_combo.currentText()
        if mode == "Taille originale (plus rapide)":
            updated_config['display']['resize_mode'] = 'original'
        elif mode == "Adapter à la fenêtre (conserver ratio)":
            updated_config['display']['resize_mode'] = 'fit'
        elif mode == "Remplir la fenêtre":
            updated_config['display']['resize_mode'] = 'fill'
        elif mode == "Taille personnalisée":
            updated_config['display']['resize_mode'] = 'custom'
        elif mode == "Pourcentage de l'original":
            updated_config['display']['resize_mode'] = 'percent'
        
        # Taille personnalisée
        updated_config['display']['custom_width'] = self.custom_width_spin.value()
        updated_config['display']['custom_height'] = self.custom_height_spin.value()
        
        # Pourcentage
        updated_config['display']['resize_percent'] = self.resize_percent_spin.value()
        
        # Redimensionnement automatique
        updated_config['display']['auto_resize_label'] = self.auto_resize_check.isChecked()
        
        # Éléments d'affichage
        updated_config['display']['show_confidence'] = self.show_confidence_check.isChecked()
        updated_config['display']['show_class'] = self.show_class_check.isChecked()
        updated_config['display']['show_fps'] = self.show_fps_check.isChecked()
        updated_config['display']['highlight_detections'] = self.highlight_check.isChecked()
        updated_config['display']['show_zone_numbers'] = self.show_zone_numbers_check.isChecked()
        
        # Performance d'affichage
        updated_config['display']['fast_resize'] = self.fast_resize_check.isChecked()
        updated_config['display']['detection_priority'] = self.detection_priority_check.isChecked()
        
        # Couleurs
        updated_config['display']['detection_color'] = self.detection_color_btn.styleSheet().split("background-color:")[1].strip().split(";")[0]
        updated_config['display']['zone_color'] = self.zone_color_btn.styleSheet().split("background-color:")[1].strip().split(";")[0]
        updated_config['display']['text_color'] = self.text_color_btn.styleSheet().split("background-color:")[1].strip().split(";")[0]
        
        # Mettre à jour les paramètres d'alerte
        if 'alerts' not in updated_config:
            updated_config['alerts'] = {}
        
        # Email
        updated_config['alerts']['email_enabled'] = self.email_enabled_check.isChecked()
        updated_config['alerts']['email_address'] = self.email_address_edit.text()
        updated_config['alerts']['smtp_server'] = self.smtp_server_edit.text()
        updated_config['alerts']['smtp_port'] = self.smtp_port_spin.value()
        updated_config['alerts']['email_user'] = self.email_user_edit.text()
        updated_config['alerts']['email_password'] = self.email_password_edit.text()
        
        # Notifications
        updated_config['alerts']['notification_enabled'] = self.notification_enabled_check.isChecked()
        updated_config['alerts']['alert_threshold'] = self.alert_threshold_spin.value()
        
        # Son
        updated_config['alerts']['sound_alert'] = self.sound_alert_check.isChecked()
        updated_config['alerts']['sound_file'] = self.sound_file_edit.text()
        
        # Webhook
        updated_config['alerts']['webhook_enabled'] = self.webhook_check.isChecked()
        updated_config['alerts']['webhook_url'] = self.webhook_url_edit.text()
        
        # Mettre à jour les paramètres de stockage
        if 'storage' not in updated_config:
            updated_config['storage'] = {}
        
        # Dossiers
        updated_config['storage']['base_dir'] = self.base_dir_edit.text()
        updated_config['storage']['videos_dir'] = self.videos_dir_edit.text()
        updated_config['storage']['images_dir'] = self.images_dir_edit.text()
        updated_config['storage']['exports_dir'] = self.exports_dir_edit.text()
        
        # Gestion des fichiers
        updated_config['storage']['auto_cleanup'] = self.auto_cleanup_check.isChecked()
        updated_config['storage']['max_storage_days'] = self.max_storage_days_spin.value()
        
        # Format d'enregistrement
        video_format = self.video_format_combo.currentText()
        if video_format == "MP4 (H.264)":
            updated_config['storage']['video_format'] = 'mp4'
        elif video_format == "AVI":
            updated_config['storage']['video_format'] = 'avi'
        elif video_format == "MKV":
            updated_config['storage']['video_format'] = 'mkv'
        elif video_format == "MOV":
            updated_config['storage']['video_format'] = 'mov'
        
        updated_config['storage']['video_quality'] = self.video_quality_slider.value()
        updated_config['storage']['image_format'] = self.image_format_combo.currentText().lower()
        
        # Mettre à jour les paramètres avancés
        if 'advanced' not in updated_config:
            updated_config['advanced'] = {}
        
        # IA
        updated_config['advanced']['batch_size'] = self.batch_size_spin.value()
        updated_config['advanced']['inference_width'] = self.inference_width_spin.value()
        updated_config['advanced']['inference_height'] = self.inference_height_spin.value()
        
        # Tracking
        tracking = self.tracking_combo.currentText()
        if tracking == "Aucun":
            updated_config['advanced']['tracking_method'] = 'none'
        elif tracking == "ByteTrack":
            updated_config['advanced']['tracking_method'] = 'bytetrack'
        elif tracking == "DeepSORT":
            updated_config['advanced']['tracking_method'] = 'deepsort'
        
        updated_config['advanced']['show_trackid'] = self.show_trackid_check.isChecked()
        
        # Système
        updated_config['advanced']['log_level'] = self.log_level_combo.currentText()
        updated_config['advanced']['max_cache'] = self.max_cache_spin.value()
        updated_config['advanced']['startup_delay'] = self.startup_delay_spin.value()
        
        # Développement
        updated_config['advanced']['debug_mode'] = self.debug_mode_check.isChecked()
        updated_config['advanced']['experimental'] = self.experimental_check.isChecked()
        
        return updated_config
