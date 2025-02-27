#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module d'analyses statistiques pour DETECTCAM
Traite les données de détection pour en extraire des informations utiles.
"""

import os
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import Counter, defaultdict

from utils.logger import get_module_logger

class DetectionAnalytics:
    """
    Analyses statistiques des détections
    """
    
    def __init__(self, detection_history: List[Dict[str, Any]]):
        """
        Initialise l'analyseur avec l'historique des détections
        
        Args:
            detection_history: Liste des détections historiques
        """
        self.logger = get_module_logger('Analytics')
        self.detection_history = detection_history
        
        # Statistiques calculées
        self.stats = self._compute_basic_stats()
        self.logger.info(f"Analyseur initialisé avec {len(detection_history)} détections")
    
    def _compute_basic_stats(self) -> Dict[str, Any]:
        """
        Calcule les statistiques de base sur les détections
        
        Returns:
            Dictionnaire des statistiques
        """
        stats = {
            'total_detections': len(self.detection_history),
            'classes': Counter(),
            'zones': Counter(),
            'hourly_distribution': [0] * 24,
            'daily_distribution': [0] * 7,
            'detections_by_day': defaultdict(int),
            'first_detection': None,
            'last_detection': None,
            'avg_confidence': 0.0,
            'high_confidence_count': 0  # Détections avec confiance > 0.8
        }
        
        # Si aucune détection, retourner les stats vides
        if not self.detection_history:
            return stats
        
        # Analyser chaque détection
        confidences = []
        
        for detection in self.detection_history:
            # Obtenir l'horodatage
            try:
                detection_time = datetime.fromisoformat(detection.get('time', ''))
            except (ValueError, TypeError):
                # Ignorer les détections sans horodatage valide
                continue
            
            # Mettre à jour les premières/dernières détections
            if stats['first_detection'] is None or detection_time < stats['first_detection']:
                stats['first_detection'] = detection_time
                
            if stats['last_detection'] is None or detection_time > stats['last_detection']:
                stats['last_detection'] = detection_time
            
            # Distribution horaire
            hour = detection_time.hour
            stats['hourly_distribution'][hour] += 1
            
            # Distribution par jour de la semaine (0 = lundi, 6 = dimanche)
            weekday = detection_time.weekday()
            stats['daily_distribution'][weekday] += 1
            
            # Détections par jour
            day_key = detection_time.strftime('%Y-%m-%d')
            stats['detections_by_day'][day_key] += 1
            
            # Classes d'objets
            class_name = detection.get('class_name', 'unknown')
            stats['classes'][class_name] += 1
            
            # Zones
            zone = detection.get('zone', 'unknown')
            stats['zones'][str(zone)] += 1
            
            # Confiance
            confidence = detection.get('confidence', 0.0)
            if isinstance(confidence, (int, float)):
                confidences.append(confidence)
                if confidence > 0.8:
                    stats['high_confidence_count'] += 1
        
        # Calculer la confiance moyenne
        if confidences:
            stats['avg_confidence'] = sum(confidences) / len(confidences)
        
        return stats
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtient un résumé des statistiques
        
        Returns:
            Dictionnaire de résumé
        """
        return {
            'total_detections': self.stats['total_detections'],
            'unique_classes': len(self.stats['classes']),
            'unique_zones': len(self.stats['zones']),
            'most_common_class': self.stats['classes'].most_common(1)[0][0] if self.stats['classes'] else None,
            'most_active_hour': np.argmax(self.stats['hourly_distribution']),
            'most_active_day': np.argmax(self.stats['daily_distribution']),
            'first_detection': self.stats['first_detection'],
            'last_detection': self.stats['last_detection'],
            'avg_confidence': self.stats['avg_confidence'],
            'high_confidence_ratio': self.stats['high_confidence_count'] / max(1, self.stats['total_detections'])
        }
    
    def get_detection_trend(self, days: int = 30) -> Dict[str, List[int]]:
        """
        Calcule la tendance des détections sur une période
        
        Args:
            days: Nombre de jours à analyser
            
        Returns:
            Dictionnaire avec les jours et le nombre de détections
        """
        if not self.detection_history:
            return {'dates': [], 'counts': []}
        
        # Déterminer la plage de dates
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days-1)
        
        # Créer un dictionnaire pour chaque jour
        date_range = {}
        current_date = start_date
        while current_date <= end_date:
            date_range[current_date.strftime('%Y-%m-%d')] = 0
            current_date += timedelta(days=1)
        
        # Remplir avec les données
        for day, count in self.stats['detections_by_day'].items():
            if day in date_range:
                date_range[day] = count
        
        # Convertir en listes pour le graphique
        dates = list(date_range.keys())
        counts = list(date_range.values())
        
        return {'dates': dates, 'counts': counts}
    
    def get_hourly_distribution(self) -> Dict[str, List]:
        """
        Obtient la distribution horaire des détections
        
        Returns:
            Dictionnaire avec les heures et le nombre de détections
        """
        hours = list(range(24))
        counts = self.stats['hourly_distribution']
        
        return {'hours': hours, 'counts': counts}
    
    def get_daily_distribution(self) -> Dict[str, List]:
        """
        Obtient la distribution par jour de la semaine des détections
        
        Returns:
            Dictionnaire avec les jours et le nombre de détections
        """
        # Noms des jours
        day_names = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
        counts = self.stats['daily_distribution']
        
        return {'days': day_names, 'counts': counts}
    
    def get_class_distribution(self, top_n: int = 10) -> Dict[str, List]:
        """
        Obtient la distribution des classes d'objets
        
        Args:
            top_n: Nombre de classes à retourner
            
        Returns:
            Dictionnaire avec les classes et le nombre de détections
        """
        # Obtenir les classes les plus fréquentes
        top_classes = self.stats['classes'].most_common(top_n)
        
        # Séparer en deux listes
        classes = [c[0] for c in top_classes]
        counts = [c[1] for c in top_classes]
        
        return {'classes': classes, 'counts': counts}
    
    def get_zone_distribution(self) -> Dict[str, List]:
        """
        Obtient la distribution des détections par zone
        
        Returns:
            Dictionnaire avec les zones et le nombre de détections
        """
        # Convertir en listes
        zones = list(self.stats['zones'].keys())
        counts = list(self.stats['zones'].values())
        
        return {'zones': zones, 'counts': counts}
    
    def get_confidence_distribution(self, bins: int = 10) -> Dict[str, List]:
        """
        Obtient la distribution des scores de confiance
        
        Args:
            bins: Nombre de groupes pour l'histogramme
            
        Returns:
            Dictionnaire avec les intervalles et le nombre de détections
        """
        # Extraire les confidences
        confidences = []
        
        for detection in self.detection_history:
            confidence = detection.get('confidence', None)
            if isinstance(confidence, (int, float)):
                confidences.append(confidence)
        
        # Si aucune confiance valide
        if not confidences:
            return {'bins': [], 'counts': []}
        
        # Créer l'histogramme
        hist, bin_edges = np.histogram(confidences, bins=bins, range=(0, 1))
        
        # Formater les intervalles
        bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges)-1)]
        
        return {'bins': bin_labels, 'counts': hist.tolist()}
    
    def get_peak_detection_times(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Obtient les moments avec le plus de détections
        
        Args:
            top_n: Nombre de pics à retourner
            
        Returns:
            Liste des pics de détection
        """
        # Regrouper par heure
        detections_by_hour = defaultdict(int)
        
        for detection in self.detection_history:
            try:
                detection_time = datetime.fromisoformat(detection.get('time', ''))
                hour_key = detection_time.strftime('%Y-%m-%d %H:00:00')
                detections_by_hour[hour_key] += 1
            except (ValueError, TypeError):
                continue
        
        # Obtenir les heures avec le plus de détections
        peak_hours = sorted(detections_by_hour.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Formater les résultats
        peaks = []
        for hour_key, count in peak_hours:
            try:
                hour_time = datetime.strptime(hour_key, '%Y-%m-%d %H:00:00')
                peaks.append({
                    'time': hour_time,
                    'count': count
                })
            except ValueError:
                continue
        
        return peaks
    
    def get_time_gap_analysis(self) -> Dict[str, Any]:
        """
        Analyse les intervalles entre les détections
        
        Returns:
            Statistiques sur les intervalles
        """
        # Extraire les horodatages
        timestamps = []
        
        for detection in self.detection_history:
            try:
                detection_time = datetime.fromisoformat(detection.get('time', ''))
                timestamps.append(detection_time)
            except (ValueError, TypeError):
                continue
        
        # Si moins de 2 détections, impossible de calculer les intervalles
        if len(timestamps) < 2:
            return {
                'min_gap': None,
                'max_gap': None,
                'avg_gap': None,
                'median_gap': None
            }
        
        # Trier les horodatages
        timestamps.sort()
        
        # Calculer les intervalles en secondes
        gaps = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                for i in range(len(timestamps)-1)]
        
        # Supprimer les valeurs extrêmes (plus de 24h)
        filtered_gaps = [gap for gap in gaps if gap <= 86400]  # 86400 secondes = 24h
        
        if not filtered_gaps:
            filtered_gaps = gaps
        
        # Convertir en tableau numpy pour les calculs
        gap_array = np.array(filtered_gaps)
        
        return {
            'min_gap': np.min(gap_array),
            'max_gap': np.max(gap_array),
            'avg_gap': np.mean(gap_array),
            'median_gap': np.median(gap_array),
            'std_gap': np.std(gap_array)
        }
    
    def get_detection_correlation(self) -> Dict[str, float]:
        """
        Analyse les corrélations entre différentes dimensions des détections
        
        Returns:
            Dictionnaire des corrélations
        """
        correlations = {}
        
        # Corrélation entre l'heure et le nombre de détections
        hour_counts = self.stats['hourly_distribution']
        hour_indices = np.arange(24)
        
        if sum(hour_counts) > 0:
            hour_corr = np.corrcoef(hour_indices, hour_counts)[0, 1]
            correlations['hour_detections'] = hour_corr
        
        # Corrélation entre le jour de la semaine et le nombre de détections
        day_counts = self.stats['daily_distribution']
        day_indices = np.arange(7)
        
        if sum(day_counts) > 0:
            day_corr = np.corrcoef(day_indices, day_counts)[0, 1]
            correlations['day_detections'] = day_corr
        
        return correlations
    
    def analyze_by_period(self, period: str = 'day') -> Dict[str, Dict[str, int]]:
        """
        Analyse les détections par période
        
        Args:
            period: 'hour', 'day', 'month' ou 'year'
            
        Returns:
            Dictionnaire des statistiques par période
        """
        # Format pour chaque période
        period_formats = {
            'hour': '%Y-%m-%d %H',
            'day': '%Y-%m-%d',
            'month': '%Y-%m',
            'year': '%Y'
        }
        
        if period not in period_formats:
            self.logger.error(f"Période non valide: {period}")
            return {}
        
        period_format = period_formats[period]
        
        # Initialiser les compteurs
        stats_by_period = defaultdict(lambda: {
            'total': 0,
            'classes': Counter(),
            'zones': Counter()
        })
        
        # Analyser chaque détection
        for detection in self.detection_history:
            try:
                detection_time = datetime.fromisoformat(detection.get('time', ''))
                period_key = detection_time.strftime(period_format)
                
                # Incrémenter le compteur total
                stats_by_period[period_key]['total'] += 1
                
                # Classes d'objets
                class_name = detection.get('class_name', 'unknown')
                stats_by_period[period_key]['classes'][class_name] += 1
                
                # Zones
                zone = detection.get('zone', 'unknown')
                stats_by_period[period_key]['zones'][str(zone)] += 1
                
            except (ValueError, TypeError):
                continue
        
        # Convertir les defaultdict en dictionnaires normaux pour la sérialisation
        result = {}
        for period_key, stats in stats_by_period.items():
            result[period_key] = {
                'total': stats['total'],
                'classes': dict(stats['classes']),
                'zones': dict(stats['zones'])
            }
        
        return result
    
    def get_anomaly_detection(self, sensitivity: float = 2.0) -> List[Dict[str, Any]]:
        """
        Détecte les anomalies dans les détections
        
        Args:
            sensitivity: Multiplicateur d'écart-type pour considérer une anomalie
            
        Returns:
            Liste des anomalies détectées
        """
        anomalies = []
        
        # Analyser par jour
        daily_stats = self.analyze_by_period('day')
        
        if not daily_stats:
            return anomalies
        
        # Calculer la moyenne et l'écart-type des détections quotidiennes
        daily_counts = [stats['total'] for stats in daily_stats.values()]
        
        if not daily_counts:
            return anomalies
        
        mean_daily = np.mean(daily_counts)
        std_daily = np.std(daily_counts)
        
        # Seuil d'anomalie (moyenne + sensitivity * écart-type)
        threshold = mean_daily + sensitivity * std_daily
        
        # Trouver les jours avec un nombre anormal de détections
        for day, stats in daily_stats.items():
            if stats['total'] > threshold:
                # Calculer le z-score
                z_score = (stats['total'] - mean_daily) / std_daily if std_daily > 0 else 0
                
                # Déterminer la classe principale
                main_class = max(stats['classes'].items(), key=lambda x: x[1])[0] if stats['classes'] else None
                
                # Ajouter l'anomalie
                anomalies.append({
                    'period': day,
                    'count': stats['total'],
                    'expected': mean_daily,
                    'z_score': z_score,
                    'main_class': main_class
                })
        
        # Trier par z-score décroissant
        anomalies.sort(key=lambda x: x['z_score'], reverse=True)
        
        return anomalies
    
    def get_activity_heatmap(self, days: int = 30) -> Dict[str, List]:
        """
        Crée une heatmap d'activité (jours x heures)
        
        Args:
            days: Nombre de jours à analyser
            
        Returns:
            Données pour une heatmap
        """
        # Créer une matrice vide (jours x heures)
        activity_matrix = np.zeros((days, 24), dtype=int)
        
        # Date de fin (aujourd'hui)
        end_date = datetime.now().date()
        
        # Date de début
        start_date = end_date - timedelta(days=days-1)
        
        # Analyser chaque détection
        for detection in self.detection_history:
            try:
                detection_time = datetime.fromisoformat(detection.get('time', ''))
                detection_date = detection_time.date()
                
                # Vérifier si la date est dans la plage
                if start_date <= detection_date <= end_date:
                    # Calculer l'indice du jour (0 = le plus ancien)
                    day_index = (detection_date - start_date).days
                    
                    # Heure
                    hour_index = detection_time.hour
                    
                    # Incrémenter le compteur
                    activity_matrix[day_index, hour_index] += 1
                    
            except (ValueError, TypeError):
                continue
        
        # Générer les dates pour les étiquettes
        date_labels = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') 
                      for i in range(days)]
        
        # Générer les heures pour les étiquettes
        hour_labels = [f"{hour:02d}h" for hour in range(24)]
        
        return {
            'matrix': activity_matrix.tolist(),
            'days': date_labels,
            'hours': hour_labels
        }
    
    def filter_detections(self, start_date: Optional[datetime] = None, 
                        end_date: Optional[datetime] = None,
                        classes: Optional[List[str]] = None,
                        zones: Optional[List[str]] = None,
                        min_confidence: float = 0.0) -> 'DetectionAnalytics':
        """
        Filtre les détections selon différents critères
        
        Args:
            start_date: Date de début
            end_date: Date de fin
            classes: Liste des classes à inclure
            zones: Liste des zones à inclure
            min_confidence: Confiance minimale
            
        Returns:
            Nouvel objet DetectionAnalytics avec les détections filtrées
        """
        filtered_history = []
        
        for detection in self.detection_history:
            # Vérifier la date
            if start_date or end_date:
                try:
                    detection_time = datetime.fromisoformat(detection.get('time', ''))
                    
                    if start_date and detection_time < start_date:
                        continue
                    
                    if end_date and detection_time > end_date:
                        continue
                        
                except (ValueError, TypeError):
                    continue
            
            # Vérifier la classe
            if classes:
                class_name = detection.get('class_name', 'unknown')
                if class_name not in classes:
                    continue
            
            # Vérifier la zone
            if zones:
                zone = str(detection.get('zone', 'unknown'))
                if zone not in zones:
                    continue
            
            # Vérifier la confiance
            if min_confidence > 0:
                confidence = detection.get('confidence', 0.0)
                if not isinstance(confidence, (int, float)) or confidence < min_confidence:
                    continue
            
            # Ajouter à la liste filtrée
            filtered_history.append(detection)
        
        # Créer un nouvel objet avec les détections filtrées
        return DetectionAnalytics(filtered_history)
    
    def compare_periods(self, period1_start: datetime, period1_end: datetime,
                       period2_start: datetime, period2_end: datetime) -> Dict[str, Any]:
        """
        Compare deux périodes de détection
        
        Args:
            period1_start: Date de début de la première période
            period1_end: Date de fin de la première période
            period2_start: Date de début de la deuxième période
            period2_end: Date de fin de la deuxième période
            
        Returns:
            Dictionnaire de comparaison
        """
        # Filtrer les deux périodes
        period1 = self.filter_detections(period1_start, period1_end)
        period2 = self.filter_detections(period2_start, period2_end)
        
        # Obtenir les statistiques
        stats1 = period1.get_summary()
        stats2 = period2.get_summary()
        
        # Calculer les différences
        diff_total = stats2['total_detections'] - stats1['total_detections']
        if stats1['total_detections'] > 0:
            diff_percent = (diff_total / stats1['total_detections']) * 100
        else:
            diff_percent = float('inf') if diff_total > 0 else 0
        
        # Durée des périodes en jours
        period1_days = (period1_end - period1_start).days or 1  # Éviter division par zéro
        period2_days = (period2_end - period2_start).days or 1
        
        # Détections par jour
        detections_per_day1 = stats1['total_detections'] / period1_days
        detections_per_day2 = stats2['total_detections'] / period2_days
        
        # Différence de détections par jour
        diff_per_day = detections_per_day2 - detections_per_day1
        if detections_per_day1 > 0:
            diff_per_day_percent = (diff_per_day / detections_per_day1) * 100
        else:
            diff_per_day_percent = float('inf') if diff_per_day > 0 else 0
        
        return {
            'period1': {
                'start': period1_start,
                'end': period1_end,
                'days': period1_days,
                'total': stats1['total_detections'],
                'per_day': detections_per_day1
            },
            'period2': {
                'start': period2_start,
                'end': period2_end,
                'days': period2_days,
                'total': stats2['total_detections'],
                'per_day': detections_per_day2
            },
            'diff': {
                'total': diff_total,
                'percent': diff_percent,
                'per_day': diff_per_day,
                'per_day_percent': diff_per_day_percent
            }
        }
