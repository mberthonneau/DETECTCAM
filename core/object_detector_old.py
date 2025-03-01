#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module de détection d'objets pour DETECTCAM
Wrapper autour de YOLO avec optimisations.
"""

import os
import cv2
import torch
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

from ultralytics import YOLO
from utils.logger import get_module_logger

class ObjectDetector:
    """
    Classe de détection d'objets utilisant YOLO avec optimisations
    pour les performances et l'efficacité.
    """
    
    def __init__(self, 
                 model_path: str = 'yolo11m.pt',
                 conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45,
                 use_cuda: bool = True,
                 half_precision: bool = True):
        """
        Initialise le détecteur d'objets
        
        Args:
            model_path: Chemin vers le modèle YOLO
            conf_threshold: Seuil de confiance (0.0 - 1.0)
            iou_threshold: Seuil IoU pour NMS (0.0 - 1.0)
            use_cuda: Utiliser CUDA si disponible
            half_precision: Utiliser la demi-précision (FP16) pour accélérer
        """
        self.logger = get_module_logger('ObjectDetector')
        self.logger.info(f"Initialisation du détecteur avec le modèle {model_path}")
        
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.use_cuda = use_cuda
        self.half_precision = half_precision
        
        # Détection de device
        self.device = self._select_device()
        
        # Chargement du modèle
        self._load_model()
        
        # Mappage français des classes (pour les 80 classes COCO)
        self.class_mapping = {
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
            "frisbee": "frisbee",
            "skis": "skis",
            "snowboard": "snowboard",
            "sports ball": "ballon de sport",
            "kite": "cerf-volant",
            "baseball bat": "batte de baseball",
            "baseball glove": "gant de baseball",
            "skateboard": "skateboard",
            "surfboard": "planche de surf",
            "tennis racket": "raquette de tennis",
            "bottle": "bouteille",
            "wine glass": "verre à vin",
            "cup": "tasse",
            "fork": "fourchette",
            "knife": "couteau",
            "spoon": "cuillère",
            "bowl": "bol",
            "banana": "banane",
            "apple": "pomme",
            "sandwich": "sandwich",
            "orange": "orange",
            "broccoli": "brocoli",
            "carrot": "carotte",
            "hot dog": "hot-dog",
            "pizza": "pizza",
            "donut": "donut",
            "cake": "gâteau",
            "chair": "chaise",
            "couch": "canapé",
            "potted plant": "plante en pot",
            "bed": "lit",
            "dining table": "table à manger",
            "toilet": "toilettes",
            "tv": "téléviseur",
            "laptop": "ordinateur portable",
            "mouse": "souris",
            "remote": "télécommande",
            "keyboard": "clavier",
            "cell phone": "téléphone",
            "microwave": "micro-ondes",
            "oven": "four",
            "toaster": "grille-pain",
            "sink": "évier",
            "refrigerator": "réfrigérateur",
            "book": "livre",
            "clock": "horloge",
            "vase": "vase",
            "scissors": "ciseaux",
            "teddy bear": "ours en peluche",
            "hair drier": "sèche-cheveux",
            "toothbrush": "brosse à dents"
        }
    
    def _select_device(self) -> str:
        """
        Sélectionne le périphérique optimal pour l'inférence
        
        Returns:
            Périphérique à utiliser ('cuda', 'mps', ou 'cpu')
        """
        if self.use_cuda and torch.cuda.is_available():
            device = 'cuda'
            # Afficher les informations sur le GPU
            gpu_name = torch.cuda.get_device_name(0)
            self.logger.info(f"Utilisation du GPU: {gpu_name}")
        elif hasattr(torch, 'mps') and torch.mps.is_available():
            # Support pour Apple Silicon (M1/M2)
            device = 'mps'
            self.logger.info("Utilisation d'Apple Silicon (MPS)")
        else:
            device = 'cpu'
            self.logger.info("Utilisation du CPU")
        
        return device
    
    def _load_model(self) -> bool:
        """
        Charge le modèle YOLO
        
        Returns:
            True si le chargement a réussi, False sinon
        """
        try:
            # Vérifier si le fichier existe
            if not os.path.exists(self.model_path) and not self.model_path.startswith('yolo'):
                self.logger.error(f"Le modèle {self.model_path} n'existe pas.")
                raise FileNotFoundError(f"Le modèle {self.model_path} n'existe pas.")
            
            # Charger le modèle
            self.model = YOLO(self.model_path)
            
            # Déplacer le modèle sur le périphérique
            self.model.to(self.device)
            
            # Appliquer la demi-précision si demandé
            if self.half_precision and self.device == 'cuda':
                self.logger.info("Activation de la demi-précision (FP16)")
                self.model.model.half()
            
            self.logger.info(f"Modèle {self.model_path} chargé avec succès sur {self.device}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
            # Fallback vers un modèle par défaut en cas d'erreur
            try:
                self.model_path = 'yolo11m.pt'  # Modèle le plus léger
                self.model = YOLO(self.model_path)
                self.model.to(self.device)
                self.logger.warning(f"Fallback vers le modèle {self.model_path}")
                return True
            except:
                self.logger.critical("Impossible de charger un modèle de fallback.")
                return False
    
    def detect(self, 
               frame: np.ndarray, 
               conf: Optional[float] = None,
               iou: Optional[float] = None,
               multi_scale: bool = False) -> List:
        """
        Détecte les objets dans une frame
        
        Args:
            frame: Frame à analyser
            conf: Seuil de confiance (utilise la valeur par défaut si None)
            iou: Seuil IoU (utilise la valeur par défaut si None)
            multi_scale: Utiliser la détection multi-échelle (plus précis mais plus lent)
            
        Returns:
            Résultats de la détection (format YOLO)
        """
        if frame is None:
            return []
        
        # Utiliser les seuils par défaut si non spécifiés
        conf = conf if conf is not None else self.conf_threshold
        iou = iou if iou is not None else self.iou_threshold
        
        try:
            # Convertir en RGB si nécessaire (YOLO attend des images RGB)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                # Check if the frame is BGR (OpenCV default)
                if cv2.__version__.startswith('4'):
                    # OpenCV 4.x
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    # OpenCV 3.x and earlier
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convertir BGR -> RGB si c'est une image OpenCV
                #frame_rgb = cv2.cvtColor(frame, color_space)
                
            else:
                frame_rgb = frame
            
            # Détection avec YOLO
            results = self.model(
                frame_rgb,
                conf=conf,
                iou=iou,
                half=self.half_precision,
                device=self.device,
                augment=multi_scale  # Multi-scale inference si demandé
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la détection: {str(e)}")
            return []  # Retourner une liste vide en cas d'erreur
    
    def set_conf_threshold(self, threshold: float):
        """
        Définit le seuil de confiance
        
        Args:
            threshold: Nouveau seuil (0.0 - 1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self.conf_threshold = threshold
    
    def set_iou_threshold(self, threshold: float):
        """
        Définit le seuil IoU
        
        Args:
            threshold: Nouveau seuil (0.0 - 1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self.iou_threshold = threshold
    
    def configure(self, **kwargs):
        """
        Configure le détecteur avec de multiples paramètres
        
        Args:
            **kwargs: Paramètres à configurer
        """
        if 'conf_threshold' in kwargs:
            self.set_conf_threshold(kwargs['conf_threshold'])
        
        if 'iou_threshold' in kwargs:
            self.set_iou_threshold(kwargs['iou_threshold'])
        
        if 'model_path' in kwargs and kwargs['model_path'] != self.model_path:
            self.model_path = kwargs['model_path']
            self._load_model()
        
        if 'use_cuda' in kwargs:
            if kwargs['use_cuda'] != self.use_cuda:
                self.use_cuda = kwargs['use_cuda']
                self.device = self._select_device()
                self.model.to(self.device)
        
        if 'half_precision' in kwargs:
            self.half_precision = kwargs['half_precision']
            if self.device == 'cuda':
                if self.half_precision:
                    self.model.model.half()
                else:
                    self.model.model.float()
    
    def get_class_name(self, class_id: int) -> str:
        """
        Retourne le nom de la classe en français
        
        Args:
            class_id: Identifiant de la classe
            
        Returns:
            Nom de la classe en français
        """
        try:
            original_name = self.model.names[class_id]
            return self.class_mapping.get(original_name, original_name)
        except:
            return f"classe_{class_id}"
    
    def preprocess_frame(self, frame: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
        """
        Prétraite une frame pour optimiser la détection
        
        Args:
            frame: Frame à prétraiter
            target_size: Taille cible (width, height) ou None pour garder la taille originale
            
        Returns:
            Frame prétraitée
        """
        if frame is None:
            return None
        
        # Redimensionner si nécessaire
        if target_size is not None:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalisation des couleurs (améliore la robustesse)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return frame
    
    def enable_tracking(self, enabled: bool = True):
        """
        Active ou désactive le tracking d'objets
        
        Args:
            enabled: True pour activer, False pour désactiver
        """
        self.model.tracker = enabled
        self.logger.info(f"Tracking {'activé' if enabled else 'désactivé'}")
