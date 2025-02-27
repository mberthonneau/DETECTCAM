#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module d'affichage des statistiques pour DETECTCAM
"""

import os
import json
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QTabWidget, QWidget, QTableWidget, QTableWidgetItem,
    QComboBox, QDateEdit, QGroupBox, QCheckBox, QFileDialog,
    QMessageBox, QSplitter, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, QDate, pyqtSignal, pyqtSlot, QSize
from PyQt6.QtGui import QColor, QFont, QIcon

# Import conditionnel de matplotlib pour la visualisation
try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from utils.logger import get_module_logger

class DetectionHistogram(QWidget):
    """Widget pour afficher un histogramme des détections"""
    
    def __init__(self, parent=None):
        """Initialise le widget d'histogramme"""
        super().__init__(parent)
        
        if not HAS_MATPLOTLIB:
            layout = QVBoxLayout(self)
            msg = QLabel("Matplotlib non disponible. Installez-le pour voir les graphiques.")
            msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(msg)
            self.setLayout(layout)
            return
        
        layout = QVBoxLayout(self)
        
        # Créer la figure matplotlib
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(300)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
    
    def plot_data(self, data: Dict[str, List[int]], title: str = "Détections par période"):
        """
        Trace un histogramme à partir des données
        
        Args:
            data: Dictionnaire des données (clé: étiquette, valeur: liste de valeurs)
            title: Titre du graphique
        """
        if not HAS_MATPLOTLIB:
            return
        
        # Effacer la figure
        self.figure.clear()
        
        # Créer un axe
        ax = self.figure.add_subplot(111)
        
        # Tracer les données
        for label, values in data.items():
            x = range(len(values))
            ax.bar(x, values, label=label, alpha=0.7)
        
        # Configurer l'axe
        ax.set_title(title)
        ax.set_xlabel("Période")
        ax.set_ylabel("Nombre de détections")
        ax.legend()
        
        # Mettre à jour le canvas
        self.canvas.draw()
    
    def plot_time_distribution(self, hours: List[int], title: str = "Distribution par heure"):
        """
        Trace la distribution des détections par heure
        
        Args:
            hours: Liste des heures de détection
            title: Titre du graphique
        """
        if not HAS_MATPLOTLIB:
            return
        
        # Effacer la figure
        self.figure.clear()
        
        # Créer un axe
        ax = self.figure.add_subplot(111)
        
        # Compter les occurrences de chaque heure
        hour_counts = [0] * 24
        for hour in hours:
            if 0 <= hour < 24:
                hour_counts[hour] += 1
        
        # Tracer le graphique
        x = range(24)
        ax.bar(x, hour_counts, color='steelblue', alpha=0.8)
        
        # Ajouter des étiquettes d'axe
        ax.set_title(title)
        ax.set_xlabel("Heure")
        ax.set_ylabel("Nombre de détections")
        ax.set_xticks(range(0, 24, 2))
        
        # Mettre à jour le canvas
        self.canvas.draw()
    
    def plot_zone_stats(self, zone_data: Dict[str, int], title: str = "Détections par zone"):
        """
        Trace les statistiques par zone
        
        Args:
            zone_data: Dictionnaire des données (clé: nom de zone, valeur: nombre de détections)
            title: Titre du graphique
        """
        if not HAS_MATPLOTLIB:
            return
        
        # Effacer la figure
        self.figure.clear()
        
        # Créer un axe
        ax = self.figure.add_subplot(111)
        
        # Trier les données par valeur décroissante
        sorted_data = sorted(zone_data.items(), key=lambda x: x[1], reverse=True)
        zones = [item[0] for item in sorted_data]
        values = [item[1] for item in sorted_data]
        
        # Générer des couleurs
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(zones)))
        
        # Tracer le graphique
        ax.barh(zones, values, color=colors, alpha=0.8)
        
        # Ajouter des étiquettes d'axe
        ax.set_title(title)
        ax.set_xlabel("Nombre de détections")
        ax.set_ylabel("Zone")
        
        # Inverser l'axe y pour que la zone avec le plus de détections soit en haut
        ax.invert_yaxis()
        
        # Ajuster les étiquettes
        ax.tick_params(axis='y', labelsize=8)
        
        # Mettre à jour le canvas
        self.canvas.draw()
    
    def plot_class_stats(self, class_data: Dict[str, int], title: str = "Détections par classe"):
        """
        Trace les statistiques par classe d'objet
        
        Args:
            class_data: Dictionnaire des données (clé: nom de classe, valeur: nombre de détections)
            title: Titre du graphique
        """
        if not HAS_MATPLOTLIB:
            return
        
        # Effacer la figure
        self.figure.clear()
        
        # Créer un axe pour le camembert
        ax = self.figure.add_subplot(111)
        
        # Filtrer les classes avec peu de détections
        threshold = max(class_data.values()) * 0.01  # 1% du maximum
        filtered_data = {k: v for k, v in class_data.items() if v >= threshold}
        
        # Si plus de 10 classes, regrouper les moins fréquentes
        if len(filtered_data) > 10:
            sorted_data = sorted(filtered_data.items(), key=lambda x: x[1], reverse=True)
            top_10 = dict(sorted_data[:10])
            others_sum = sum(dict(sorted_data[10:]).values())
            if others_sum > 0:
                top_10["Autres"] = others_sum
            filtered_data = top_10
        
        # Tracer le camembert
        labels = filtered_data.keys()
        sizes = filtered_data.values()
        
        # Générer des couleurs
        colors = plt.cm.tab10(range(len(filtered_data)))
        
        # Créer le camembert
        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels=None,  # Pas d'étiquettes sur le graphique
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            shadow=False
        )
        
        # Améliorer l'apparence des pourcentages
        for autotext in autotexts:
            autotext.set_fontsize(8)
            autotext.set_weight('bold')
        
        # Ajouter la légende
        ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        # Égaliser les axes pour un camembert circulaire
        ax.axis('equal')
        
        # Ajouter un titre
        ax.set_title(title)
        
        # Mettre à jour le canvas
        self.canvas.draw()


class StatsView(QDialog):
    """Dialogue d'affichage des statistiques de détection"""
    
    def __init__(self, config: Dict[str, Any], parent=None):
        """
        Initialise la vue des statistiques
        
        Args:
            config: Configuration de l'application
            parent: Widget parent
        """
        super().__init__(parent)
        self.logger = get_module_logger('UI.StatsView')
        self.config = config
        
        # Charger les données existantes
        self.detection_history = []
        self.load_detection_history()
        
        # Initialiser l'interface utilisateur
        self._init_ui()
        
        # Mettre à jour les statistiques
        self.update_stats()
    
    def _init_ui(self):
        """Initialise l'interface utilisateur"""
        self.setWindowTitle("Statistiques de détection")
        self.setMinimumSize(900, 600)
        
        # Layout principal
        main_layout = QVBoxLayout(self)
        
        # Filtres en haut
        filters_group = QGroupBox("Filtres")
        filters_layout = QHBoxLayout()
        
        # Période
        period_label = QLabel("Période:")
        self.period_combo = QComboBox()
        self.period_combo.addItems([
            "Aujourd'hui", "7 derniers jours", "30 derniers jours", 
            "Mois dernier", "Année dernière", "Tout"
        ])
        self.period_combo.currentIndexChanged.connect(self.update_stats)
        
        # Date de début et fin
        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDate(QDate.currentDate().addDays(-7))
        
        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDate(QDate.currentDate())
        
        date_label = QLabel("Du")
        to_label = QLabel("au")
        
        # Zone
        zone_label = QLabel("Zone:")
        self.zone_combo = QComboBox()
        self.zone_combo.addItem("Toutes les zones")
        
        # Ajouter les zones existantes
        for i, _ in enumerate(self.config.get('zones', [])):
            zone_name = self.config.get('zone_names', {}).get(str(i), f"Zone {i+1}")
            self.zone_combo.addItem(zone_name)
        
        self.zone_combo.currentIndexChanged.connect(self.update_stats)
        
        # Classe d'objet
        class_label = QLabel("Classe:")
        self.class_combo = QComboBox()
        self.class_combo.addItem("Toutes les classes")
        self.class_combo.addItems([
            "personne", "véhicule", "animal", "objet"
        ])
        self.class_combo.currentIndexChanged.connect(self.update_stats)
        
        # Bouton d'actualisation
        refresh_btn = QPushButton("Actualiser")
        refresh_btn.clicked.connect(self.update_stats)
        
        # Ajouter les filtres au layout
        filters_layout.addWidget(period_label)
        filters_layout.addWidget(self.period_combo)
        filters_layout.addWidget(date_label)
        filters_layout.addWidget(self.start_date)
        filters_layout.addWidget(to_label)
        filters_layout.addWidget(self.end_date)
        filters_layout.addWidget(zone_label)
        filters_layout.addWidget(self.zone_combo)
        filters_layout.addWidget(class_label)
        filters_layout.addWidget(self.class_combo)
        filters_layout.addWidget(refresh_btn)
        
        filters_group.setLayout(filters_layout)
        
        # Onglets pour les différentes vues
        self.tabs = QTabWidget()
        
        # Onglet de résumé
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        
        # Statistiques globales
        global_stats_group = QGroupBox("Statistiques globales")
        global_stats_layout = QVBoxLayout()
        
        self.total_detections_label = QLabel("Nombre total de détections: 0")
        self.avg_per_day_label = QLabel("Moyenne par jour: 0")
        self.peak_day_label = QLabel("Jour de pointe: -")
        self.peak_hour_label = QLabel("Heure de pointe: -")
        
        global_stats_layout.addWidget(self.total_detections_label)
        global_stats_layout.addWidget(self.avg_per_day_label)
        global_stats_layout.addWidget(self.peak_day_label)
        global_stats_layout.addWidget(self.peak_hour_label)
        
        global_stats_group.setLayout(global_stats_layout)
        summary_layout.addWidget(global_stats_group)
        
        # Histogramme pour le résumé
        if HAS_MATPLOTLIB:
            self.summary_histogram = DetectionHistogram()
            summary_layout.addWidget(self.summary_histogram)
        
        self.tabs.addTab(summary_tab, "Résumé")
        
        # Onglet par zone
        zones_tab = QWidget()
        zones_layout = QVBoxLayout(zones_tab)
        
        # Tableau des statistiques par zone
        self.zones_table = QTableWidget()
        self.zones_table.setColumnCount(3)
        self.zones_table.setHorizontalHeaderLabels(["Zone", "Détections", "% du total"])
        self.zones_table.horizontalHeader().setStretchLastSection(True)
        
        zones_layout.addWidget(self.zones_table)
        
        # Histogramme pour les zones
        if HAS_MATPLOTLIB:
            self.zones_histogram = DetectionHistogram()
            zones_layout.addWidget(self.zones_histogram)
        
        self.tabs.addTab(zones_tab, "Par Zone")
        
        # Onglet par heure
        time_tab = QWidget()
        time_layout = QVBoxLayout(time_tab)
        
        # Tableau des statistiques par heure
        self.time_table = QTableWidget()
        self.time_table.setColumnCount(3)
        self.time_table.setHorizontalHeaderLabels(["Heure", "Détections", "% du total"])
        self.time_table.horizontalHeader().setStretchLastSection(True)
        
        time_layout.addWidget(self.time_table)
        
        # Histogramme pour les heures
        if HAS_MATPLOTLIB:
            self.time_histogram = DetectionHistogram()
            time_layout.addWidget(self.time_histogram)
        
        self.tabs.addTab(time_tab, "Par Heure")
        
        # Onglet par classe d'objet
        class_tab = QWidget()
        class_layout = QVBoxLayout(class_tab)
        
        # Tableau des statistiques par classe
        self.class_table = QTableWidget()
        self.class_table.setColumnCount(3)
        self.class_table.setHorizontalHeaderLabels(["Classe", "Détections", "% du total"])
        self.class_table.horizontalHeader().setStretchLastSection(True)
        
        class_layout.addWidget(self.class_table)
        
        # Histogramme pour les classes
        if HAS_MATPLOTLIB:
            self.class_histogram = DetectionHistogram()
            class_layout.addWidget(self.class_histogram)
        
        self.tabs.addTab(class_tab, "Par Classe")
        
        # Onglet des données brutes
        data_tab = QWidget()
        data_layout = QVBoxLayout(data_tab)
        
        # Tableau des données brutes
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(5)
        self.data_table.setHorizontalHeaderLabels(["Date", "Heure", "Zone", "Classe", "Confiance"])
        self.data_table.horizontalHeader().setStretchLastSection(True)
        
        data_layout.addWidget(self.data_table)
        
        self.tabs.addTab(data_tab, "Données")
        
        # Boutons d'exportation
        export_layout = QHBoxLayout()
        
        export_csv_btn = QPushButton("Exporter CSV")
        export_csv_btn.clicked.connect(self.export_csv)
        
        export_json_btn = QPushButton("Exporter JSON")
        export_json_btn.clicked.connect(self.export_json)
        
        # Si matplotlib est disponible
        if HAS_MATPLOTLIB:
            export_image_btn = QPushButton("Exporter graphique")
            export_image_btn.clicked.connect(self.export_current_chart)
            export_layout.addWidget(export_image_btn)
        
        # Bouton de fermeture
        close_btn = QPushButton("Fermer")
        close_btn.clicked.connect(self.close)
        
        export_layout.addWidget(export_csv_btn)
        export_layout.addWidget(export_json_btn)
        export_layout.addStretch()
        export_layout.addWidget(close_btn)
        
        # Assemblage du layout principal
        main_layout.addWidget(filters_group)
        main_layout.addWidget(self.tabs, 1)  # 1 = stretch factor
        main_layout.addLayout(export_layout)
        
        self.setLayout(main_layout)
    
    def load_detection_history(self):
        """Charge l'historique des détections"""
        try:
            # Charger depuis un fichier JSON ou une base de données
            history_file = os.path.join(
                self.config.get('storage', {}).get('base_dir', 'detections'),
                'detection_history.json'
            )
            
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    self.detection_history = json.load(f)
                    
                    self.logger.info(f"Historique de détection chargé: {len(self.detection_history)} entrées")
            else:
                self.logger.warning(f"Fichier d'historique non trouvé: {history_file}")
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de l'historique: {str(e)}")
            self.detection_history = []
    
    def update_stats(self):
        """Met à jour les statistiques affichées selon les filtres"""
        # Filtrer les données selon la période
        filtered_data = self._filter_detection_history()
        
        # Aucune donnée à afficher
        if not filtered_data:
            self._show_empty_stats()
            return
        
        # Mettre à jour les statistiques globales
        self._update_global_stats(filtered_data)
        
        # Mettre à jour les statistiques par zone
        self._update_zone_stats(filtered_data)
        
        # Mettre à jour les statistiques par heure
        self._update_time_stats(filtered_data)
        
        # Mettre à jour les statistiques par classe
        self._update_class_stats(filtered_data)
        
        # Mettre à jour les données brutes
        self._update_raw_data(filtered_data)
    
    def _filter_detection_history(self) -> List[Dict[str, Any]]:
        """
        Filtre l'historique des détections selon les critères définis
        
        Returns:
            Liste filtrée des détections
        """
        result = []
        
        # Déterminer les dates de début et de fin
        period = self.period_combo.currentText()
        
        if period == "Aujourd'hui":
            start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = datetime.now()
        elif period == "7 derniers jours":
            start_date = (datetime.now() - timedelta(days=7)).replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = datetime.now()
        elif period == "30 derniers jours":
            start_date = (datetime.now() - timedelta(days=30)).replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = datetime.now()
        elif period == "Mois dernier":
            today = datetime.now()
            if today.month == 1:
                start_date = datetime(today.year - 1, 12, 1)
            else:
                start_date = datetime(today.year, today.month - 1, 1)
            if today.month == 1:
                end_date = datetime(today.year, 1, 1)
            else:
                end_date = datetime(today.year, today.month, 1)
        elif period == "Année dernière":
            today = datetime.now()
            start_date = datetime(today.year - 1, 1, 1)
            end_date = datetime(today.year, 1, 1)
        else:  # "Tout"
            start_date = datetime.fromtimestamp(0)  # Début des temps UNIX
            end_date = datetime.now()
        
        # Filtrer par zone
        zone_filter = self.zone_combo.currentIndex() - 1  # -1 = Toutes les zones
        
        # Filtrer par classe
        class_filter = self.class_combo.currentText()
        if class_filter == "Toutes les classes":
            class_filter = None
        
        # Parcourir l'historique
        for detection in self.detection_history:
            try:
                # Vérifier la date
                detection_time = datetime.fromisoformat(detection['time'])
                if not (start_date <= detection_time <= end_date):
                    continue
                
                # Vérifier la zone
                if zone_filter >= 0 and detection.get('zone', -1) != zone_filter:
                    continue
                
                # Vérifier la classe
                if class_filter and detection.get('class_name', '') != class_filter:
                    # Gestion des groupes de classes
                    if class_filter == "véhicule" and detection.get('class_name') not in [
                        "voiture", "moto", "camion", "bus", "vélo"
                    ]:
                        continue
                    elif class_filter == "animal" and detection.get('class_name') not in [
                        "chat", "chien", "oiseau", "cheval", "vache", "mouton", "éléphant", 
                        "ours", "zèbre", "girafe"
                    ]:
                        continue
                    elif class_filter == "objet" and detection.get('class_name') not in [
                        "sac à dos", "valise", "téléphone", "ordinateur portable", "chaise",
                        "table", "bouteille", "tasse", "livre"
                    ]:
                        continue
                
                # Ajouter à la liste filtrée
                result.append(detection)
                
            except (ValueError, KeyError):
                continue
        
        return result
    
    def _show_empty_stats(self):
        """Affiche des statistiques vides"""
        # Statistiques globales
        self.total_detections_label.setText("Nombre total de détections: 0")
        self.avg_per_day_label.setText("Moyenne par jour: 0")
        self.peak_day_label.setText("Jour de pointe: -")
        self.peak_hour_label.setText("Heure de pointe: -")
        
        # Tableaux
        for table in [self.zones_table, self.time_table, self.class_table, self.data_table]:
            table.setRowCount(0)
        
        # Histogrammes
        if HAS_MATPLOTLIB:
            data = {"Aucune donnée": [0]}
            for histogram in [self.summary_histogram, self.zones_histogram, 
                             self.time_histogram, self.class_histogram]:
                histogram.plot_data(data, "Aucune donnée pour la période sélectionnée")
    
    def _update_global_stats(self, filtered_data: List[Dict[str, Any]]):
        """
        Met à jour les statistiques globales
        
        Args:
            filtered_data: Liste des détections filtrées
        """
        # Nombre total de détections
        total_detections = len(filtered_data)
        self.total_detections_label.setText(f"Nombre total de détections: {total_detections}")
        
        # Regrouper par jour
        detections_by_day = {}
        for detection in filtered_data:
            try:
                detection_time = datetime.fromisoformat(detection['time'])
                day_key = detection_time.strftime("%Y-%m-%d")
                
                if day_key not in detections_by_day:
                    detections_by_day[day_key] = 0
                
                detections_by_day[day_key] += 1
            except (ValueError, KeyError):
                continue
        
        # Moyenne par jour
        if detections_by_day:
            avg_per_day = total_detections / len(detections_by_day)
            self.avg_per_day_label.setText(f"Moyenne par jour: {avg_per_day:.1f}")
            
            # Jour de pointe
            peak_day = max(detections_by_day.items(), key=lambda x: x[1])
            peak_day_date = datetime.strptime(peak_day[0], "%Y-%m-%d").strftime("%d/%m/%Y")
            self.peak_day_label.setText(f"Jour de pointe: {peak_day_date} ({peak_day[1]} détections)")
        else:
            self.avg_per_day_label.setText("Moyenne par jour: 0")
            self.peak_day_label.setText("Jour de pointe: -")
        
        # Regrouper par heure
        detections_by_hour = {}
        for detection in filtered_data:
            try:
                detection_time = datetime.fromisoformat(detection['time'])
                hour_key = detection_time.hour
                
                if hour_key not in detections_by_hour:
                    detections_by_hour[hour_key] = 0
                
                detections_by_hour[hour_key] += 1
            except (ValueError, KeyError):
                continue
        
        # Heure de pointe
        if detections_by_hour:
            peak_hour = max(detections_by_hour.items(), key=lambda x: x[1])
            self.peak_hour_label.setText(f"Heure de pointe: {peak_hour[0]}h ({peak_hour[1]} détections)")
        else:
            self.peak_hour_label.setText("Heure de pointe: -")
        
        # Histogramme de résumé (détections par jour)
        if HAS_MATPLOTLIB:
            # Trier les jours par ordre chronologique
            sorted_days = sorted(detections_by_day.items(), key=lambda x: x[0])
            
            # Si trop de jours, regrouper par semaine ou mois
            if len(sorted_days) > 31:
                # Regrouper par mois
                detections_by_month = {}
                for day, count in sorted_days:
                    month_key = day[:7]  # YYYY-MM
                    if month_key not in detections_by_month:
                        detections_by_month[month_key] = 0
                    detections_by_month[month_key] += count
                
                # Formater les étiquettes de mois
                formatted_data = {}
                for month, count in detections_by_month.items():
                    year, month = month.split('-')
                    formatted_key = f"{int(month)}/{year[2:]}"  # MM/YY
                    formatted_data[formatted_key] = count
                
                data = {"Mensuel": list(formatted_data.values())}
                self.summary_histogram.plot_data(data, "Détections par mois")
                
            elif len(sorted_days) > 14:
                # Regrouper par semaine
                detections_by_week = {}
                for day, count in sorted_days:
                    date = datetime.strptime(day, "%Y-%m-%d")
                    week_num = date.isocalendar()[1]
                    week_key = f"{date.year}-W{week_num:02d}"
                    
                    if week_key not in detections_by_week:
                        detections_by_week[week_key] = 0
                    detections_by_week[week_key] += count
                
                data = {"Hebdomadaire": list(detections_by_week.values())}
                self.summary_histogram.plot_data(data, "Détections par semaine")
                
            else:
                # Afficher par jour
                data = {"Quotidien": [count for _, count in sorted_days]}
                self.summary_histogram.plot_data(data, "Détections par jour")
    
    def _update_zone_stats(self, filtered_data: List[Dict[str, Any]]):
        """
        Met à jour les statistiques par zone
        
        Args:
            filtered_data: Liste des détections filtrées
        """
        # Regrouper par zone
        detections_by_zone = {}
        for detection in filtered_data:
            try:
                zone = detection.get('zone', "inconnu")
                zone_key = str(zone)
                
                if zone_key not in detections_by_zone:
                    detections_by_zone[zone_key] = 0
                
                detections_by_zone[zone_key] += 1
            except (ValueError, KeyError):
                continue
        
        # Mise à jour du tableau
        self.zones_table.setRowCount(len(detections_by_zone))
        
        total_detections = len(filtered_data)
        row = 0
        
        zone_names = {}
        for i, _ in enumerate(self.config.get('zones', [])):
            zone_names[str(i)] = self.config.get('zone_names', {}).get(str(i), f"Zone {i+1}")
        
        # Trier par nombre de détections (décroissant)
        sorted_zones = sorted(detections_by_zone.items(), key=lambda x: x[1], reverse=True)
        
        for zone_key, count in sorted_zones:
            # Nom de la zone
            if zone_key == "global" or zone_key == "inconnu":
                zone_name = "Toute l'image"
            else:
                # Obtenir le nom de la zone depuis la configuration
                zone_name = zone_names.get(zone_key, f"Zone {zone_key}")
            
            # Calculer le pourcentage
            percent = (count / total_detections * 100) if total_detections > 0 else 0
            
            # Remplir le tableau
            self.zones_table.setItem(row, 0, QTableWidgetItem(zone_name))
            self.zones_table.setItem(row, 1, QTableWidgetItem(str(count)))
            self.zones_table.setItem(row, 2, QTableWidgetItem(f"{percent:.1f}%"))
            
            row += 1
        
        # Ajuster les colonnes
        self.zones_table.resizeColumnsToContents()
        
        # Histogramme des zones
        if HAS_MATPLOTLIB and detections_by_zone:
            # Préparer les données pour l'histogramme
            zone_data = {}
            for zone_key, count in sorted_zones:
                if zone_key == "global" or zone_key == "inconnu":
                    zone_name = "Toute l'image"
                else:
                    zone_name = zone_names.get(zone_key, f"Zone {zone_key}")
                zone_data[zone_name] = count
            
            # Tracer l'histogramme
            self.zones_histogram.plot_zone_stats(zone_data, "Détections par zone")
    
    def _update_time_stats(self, filtered_data: List[Dict[str, Any]]):
        """
        Met à jour les statistiques par heure
        
        Args:
            filtered_data: Liste des détections filtrées
        """
        # Regrouper par heure
        detections_by_hour = {}
        detection_hours = []
        
        for detection in filtered_data:
            try:
                detection_time = datetime.fromisoformat(detection['time'])
                hour = detection_time.hour
                detection_hours.append(hour)
                
                if hour not in detections_by_hour:
                    detections_by_hour[hour] = 0
                
                detections_by_hour[hour] += 1
            except (ValueError, KeyError):
                continue
        
        # Mise à jour du tableau
        self.time_table.setRowCount(24)  # 24 heures
        
        total_detections = len(filtered_data)
        
        for hour in range(24):
            count = detections_by_hour.get(hour, 0)
            percent = (count / total_detections * 100) if total_detections > 0 else 0
            
            # Formater l'heure
            hour_display = f"{hour:02d}h - {(hour+1)%24:02d}h"
            
            # Remplir le tableau
            self.time_table.setItem(hour, 0, QTableWidgetItem(hour_display))
            self.time_table.setItem(hour, 1, QTableWidgetItem(str(count)))
            self.time_table.setItem(hour, 2, QTableWidgetItem(f"{percent:.1f}%"))
        
        # Ajuster les colonnes
        self.time_table.resizeColumnsToContents()
        
        # Histogramme des heures
        if HAS_MATPLOTLIB and detection_hours:
            # Tracer l'histogramme
            self.time_histogram.plot_time_distribution(detection_hours, "Distribution par heure")
    
    def _update_class_stats(self, filtered_data: List[Dict[str, Any]]):
        """
        Met à jour les statistiques par classe d'objet
        
        Args:
            filtered_data: Liste des détections filtrées
        """
        # Regrouper par classe
        detections_by_class = {}
        for detection in filtered_data:
            try:
                class_name = detection.get('class_name', "inconnu")
                
                if class_name not in detections_by_class:
                    detections_by_class[class_name] = 0
                
                detections_by_class[class_name] += 1
            except (ValueError, KeyError):
                continue
        
        # Mise à jour du tableau
        self.class_table.setRowCount(len(detections_by_class))
        
        total_detections = len(filtered_data)
        row = 0
        
        # Trier par nombre de détections (décroissant)
        sorted_classes = sorted(detections_by_class.items(), key=lambda x: x[1], reverse=True)
        
        for class_name, count in sorted_classes:
            # Calculer le pourcentage
            percent = (count / total_detections * 100) if total_detections > 0 else 0
            
            # Remplir le tableau
            self.class_table.setItem(row, 0, QTableWidgetItem(class_name))
            self.class_table.setItem(row, 1, QTableWidgetItem(str(count)))
            self.class_table.setItem(row, 2, QTableWidgetItem(f"{percent:.1f}%"))
            
            row += 1
        
        # Ajuster les colonnes
        self.class_table.resizeColumnsToContents()
        
        # Histogramme des classes
        if HAS_MATPLOTLIB and detections_by_class:
            # Tracer le camembert
            self.class_histogram.plot_class_stats(detections_by_class, "Répartition par classe d'objet")
    
    def _update_raw_data(self, filtered_data: List[Dict[str, Any]]):
        """
        Met à jour le tableau des données brutes
        
        Args:
            filtered_data: Liste des détections filtrées
        """
        # Limiter à 1000 entrées pour les performances
        max_rows = min(1000, len(filtered_data))
        self.data_table.setRowCount(max_rows)
        
        # Récupérer les noms des zones
        zone_names = {}
        for i, _ in enumerate(self.config.get('zones', [])):
            zone_names[str(i)] = self.config.get('zone_names', {}).get(str(i), f"Zone {i+1}")
        
        # Trier par date (récent d'abord)
        sorted_data = sorted(filtered_data, 
                            key=lambda x: datetime.fromisoformat(x.get('time', '1970-01-01T00:00:00')),
                            reverse=True)
        
        # Limiter les données
        display_data = sorted_data[:max_rows]
        
        for row, detection in enumerate(display_data):
            try:
                # Extraire les données
                detection_time = datetime.fromisoformat(detection.get('time', ''))
                date_str = detection_time.strftime("%Y-%m-%d")
                time_str = detection_time.strftime("%H:%M:%S")
                
                # Zone
                zone_key = str(detection.get('zone', "inconnu"))
                if zone_key == "global" or zone_key == "inconnu":
                    zone_display = "Toute l'image"
                else:
                    zone_display = zone_names.get(zone_key, f"Zone {zone_key}")
                
                # Classe
                class_name = detection.get('class_name', "inconnu")
                
                # Confiance
                confidence = detection.get('confidence', 0.0)
                confidence_str = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else str(confidence)
                
                # Remplir le tableau
                self.data_table.setItem(row, 0, QTableWidgetItem(date_str))
                self.data_table.setItem(row, 1, QTableWidgetItem(time_str))
                self.data_table.setItem(row, 2, QTableWidgetItem(zone_display))
                self.data_table.setItem(row, 3, QTableWidgetItem(class_name))
                self.data_table.setItem(row, 4, QTableWidgetItem(confidence_str))
                
            except (ValueError, KeyError):
                # Ignorer les entrées invalides
                continue
        
        # Ajuster les colonnes
        self.data_table.resizeColumnsToContents()
    
    def export_csv(self):
        """Exporte les données filtrées au format CSV"""
        # Filtrer les données
        filtered_data = self._filter_detection_history()
        
        if not filtered_data:
            QMessageBox.warning(self, "Export CSV", "Aucune donnée à exporter.")
            return
        
        # Demander le chemin du fichier
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Exporter en CSV", "", "Fichiers CSV (*.csv)"
        )
        
        if not file_path:
            return
        
        # Ajouter l'extension .csv si nécessaire
        if not file_path.endswith('.csv'):
            file_path += '.csv'
        
        # Récupérer les noms des zones
        zone_names = {}
        for i, _ in enumerate(self.config.get('zones', [])):
            zone_names[str(i)] = self.config.get('zone_names', {}).get(str(i), f"Zone {i+1}")
        
        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # En-tête
                writer.writerow(['Date', 'Heure', 'Zone', 'Classe', 'Confiance', 'Coordonnées'])
                
                # Données
                for detection in filtered_data:
                    try:
                        # Extraire les données
                        detection_time = datetime.fromisoformat(detection.get('time', ''))
                        date_str = detection_time.strftime("%Y-%m-%d")
                        time_str = detection_time.strftime("%H:%M:%S")
                        
                        # Zone
                        zone_key = str(detection.get('zone', "inconnu"))
                        if zone_key == "global" or zone_key == "inconnu":
                            zone_display = "Toute l'image"
                        else:
                            zone_display = zone_names.get(zone_key, f"Zone {zone_key}")
                        
                        # Classe
                        class_name = detection.get('class_name', "inconnu")
                        
                        # Confiance
                        confidence = detection.get('confidence', 0.0)
                        confidence_str = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else str(confidence)
                        
                        # Coordonnées
                        center = detection.get('center', (0, 0))
                        coords = f"{center[0]},{center[1]}"
                        
                        # Écrire la ligne
                        writer.writerow([date_str, time_str, zone_display, class_name, confidence_str, coords])
                        
                    except (ValueError, KeyError):
                        # Ignorer les entrées invalides
                        continue
            
            QMessageBox.information(self, "Export CSV", f"Données exportées avec succès dans {file_path}")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'export CSV: {str(e)}")
            QMessageBox.critical(self, "Erreur d'export", f"Impossible d'exporter les données: {str(e)}")
    
    def export_json(self):
        """Exporte les données filtrées au format JSON"""
        # Filtrer les données
        filtered_data = self._filter_detection_history()
        
        if not filtered_data:
            QMessageBox.warning(self, "Export JSON", "Aucune donnée à exporter.")
            return
        
        # Demander le chemin du fichier
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Exporter en JSON", "", "Fichiers JSON (*.json)"
        )
        
        if not file_path:
            return
        
        # Ajouter l'extension .json si nécessaire
        if not file_path.endswith('.json'):
            file_path += '.json'
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, indent=2)
            
            QMessageBox.information(self, "Export JSON", f"Données exportées avec succès dans {file_path}")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'export JSON: {str(e)}")
            QMessageBox.critical(self, "Erreur d'export", f"Impossible d'exporter les données: {str(e)}")
    
    def export_current_chart(self):
        """Exporte le graphique actuel en image"""
        if not HAS_MATPLOTLIB:
            QMessageBox.warning(self, "Export impossible", 
                              "Matplotlib n'est pas disponible pour exporter les graphiques.")
            return
        
        # Déterminer le graphique à exporter
        current_tab = self.tabs.currentIndex()
        
        if current_tab == 0:  # Résumé
            figure = self.summary_histogram.figure
            default_name = "resume_detections.png"
        elif current_tab == 1:  # Zones
            figure = self.zones_histogram.figure
            default_name = "zones_detections.png"
        elif current_tab == 2:  # Temps
            figure = self.time_histogram.figure
            default_name = "temps_detections.png"
        elif current_tab == 3:  # Classes
            figure = self.class_histogram.figure
            default_name = "classes_detections.png"
        else:
            QMessageBox.warning(self, "Export impossible", 
                              "Aucun graphique disponible dans cet onglet.")
            return
        
        # Demander le chemin du fichier
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Exporter le graphique", default_name, "Images (*.png *.jpg *.pdf)"
        )
        
        if not file_path:
            return
        
        try:
            # Sauvegarder la figure
            figure.savefig(file_path, dpi=300, bbox_inches='tight')
            
            QMessageBox.information(self, "Export réussi", 
                                  f"Graphique exporté avec succès dans {file_path}")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'export du graphique: {str(e)}")
            QMessageBox.critical(self, "Erreur d'export", 
                               f"Impossible d'exporter le graphique: {str(e)}")
