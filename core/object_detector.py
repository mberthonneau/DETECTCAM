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
        self.model = None
        
        # Détection de device
        self.device = self._select_device()
        
        # Chargement du modèle
        success = self._load_model()
        if not success:
            self.logger.critical("Initialisation du détecteur échouée")
        
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
        try:
            if self.use_cuda and torch.cuda.is_available():
                device = 'cuda'
                # Afficher les informations sur le GPU
                gpu_name = torch.cuda.get_device_name(0)
                self.logger.info(f"Utilisation du GPU: {gpu_name}")
            elif hasattr(torch, 'mps') and hasattr(torch.mps, 'is_available') and torch.mps.is_available():
                # Support pour Apple Silicon (M1/M2)
                device = 'mps'
                self.logger.info("Utilisation d'Apple Silicon (MPS)")
            else:
                device = 'cpu'
                self.logger.info("Utilisation du CPU")
            
            return device
        except Exception as e:
            self.logger.error(f"Erreur lors de la sélection du périphérique: {str(e)}")
            return 'cpu'  # Fallback sur CPU en cas d'erreur
    
    def _load_model(self) -> bool:
        """
        Charge le modèle YOLO
        
        Returns:
            True si le chargement a réussi, False sinon
        """
        try:
            # Si un modèle est déjà chargé, libérer les ressources
            if self.model is not None:
                del self.model
                torch.cuda.empty_cache()  # Libérer la mémoire GPU si applicable
                self.model = None
            
            # Vérifier si le fichier existe pour les modèles locaux
            if not os.path.exists(self.model_path) and not self.model_path.startswith('yolo'):
                self.logger.error(f"Le modèle {self.model_path} n'existe pas.")
                
                # Essayer un modèle par défaut
                default_model = 'yolo11n.pt'
                if os.path.exists(default_model):
                    self.logger.warning(f"Utilisation du modèle par défaut {default_model}")
                    self.model_path = default_model
                else:
                    # Si pas de modèle local, essayer de télécharger depuis Ultralytics
                    self.logger.warning(f"Tentative de téléchargement du modèle {self.model_path}")
            
            # Charger le modèle avec gestion des erreurs
            try:
                self.model = YOLO(self.model_path)
            except Exception as model_error:
                self.logger.error(f"Erreur lors du chargement du modèle {self.model_path}: {str(model_error)}")
                
                # Fallback vers un modèle plus léger
                fallback_models = ['yolo11m.pt', 'yolo11s.pt']
                for fallback in fallback_models:
                    try:
                        self.logger.warning(f"Tentative de fallback vers {fallback}")
                        self.model = YOLO(fallback)
                        self.model_path = fallback
                        break
                    except:
                        continue
                
                if self.model is None:
                    raise RuntimeError("Impossible de charger un modèle, même en fallback")
            
            # Déplacer le modèle sur le périphérique
            if self.model is not None:
                self.model.to(self.device)
                
                # Appliquer la demi-précision si demandé et sur CUDA
                if self.half_precision and self.device == 'cuda':
                    self.logger.info("Activation de la demi-précision (FP16)")
                    try:
                        # Différentes versions de YOLO peuvent avoir des structures différentes
                        if hasattr(self.model, 'half'):
                            self.model.half()
                        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'half'):
                            self.model.model.half()
                        else:
                            self.logger.warning("Impossible d'activer la demi-précision : méthode half() non trouvée")
                    except Exception as half_error:
                        self.logger.warning(f"Erreur lors de l'activation de la demi-précision: {str(half_error)}")
                
                self.logger.info(f"Modèle {self.model_path} chargé avec succès sur {self.device}")
                return True
            else:
                return False
            
        except Exception as e:
            self.logger.error(f"Erreur critique lors du chargement du modèle: {str(e)}")
            
            # Dernier recours - essai de chargement d'un modèle minimal
            try:
                self.model_path = 'yolo11n.pt'  # Modèle le plus léger
                self.model = YOLO(self.model_path)
                self.model.to('cpu')  # Utiliser le CPU pour plus de fiabilité
                self.device = 'cpu'
                self.logger.warning(f"Fallback d'urgence vers le modèle {self.model_path} sur CPU")
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
        # Vérifier que le modèle est chargé
        if self.model is None:
            if not self._load_model():
                self.logger.error("Modèle non chargé, détection impossible")
                return []
        
        # Vérifier que la frame est valide
        if frame is None or frame.size == 0:
            self.logger.warning("Frame vide ou invalide")
            return []
        
        # Utiliser les seuils par défaut si non spécifiés
        conf = conf if conf is not None else self.conf_threshold
        iou = iou if iou is not None else self.iou_threshold
        
        try:
            # Vérifier que la frame a le bon format
            if not isinstance(frame, np.ndarray):
                self.logger.error(f"Type de frame invalide: {type(frame)}")
                return []
            
            # Vérifier les dimensions et le type de la frame
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                self.logger.error(f"Dimensions de frame invalides: {frame.shape}")
                return []
            
            # Convertir en RGB si nécessaire (YOLO attend des images RGB)
            try:
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                
                # Convertir BGR -> RGB (OpenCV utilise BGR par défaut)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception as convert_error:
                self.logger.error(f"Erreur lors de la conversion de la frame: {str(convert_error)}")
                # Essayer d'utiliser la frame telle quelle
                frame_rgb = frame
            
            # Détection avec YOLO et gestion des exceptions
            try:
                results = self.model(
                    frame_rgb,
                    conf=conf,
                    iou=iou,
                    half=self.half_precision and self.device == 'cuda',
                    device=self.device,
                    augment=multi_scale  # Multi-scale inference si demandé
                )
                return results
            except RuntimeError as runtime_error:
                # Erreur CUDA out of memory
                if "CUDA out of memory" in str(runtime_error):
                    self.logger.error("CUDA out of memory - passage en mode CPU")
                    
                    # Fallback sur CPU
                    old_device = self.device
                    self.device = 'cpu'
                    
                    # Recharger le modèle sur CPU
                    if self.model is not None:
                        self.model.to('cpu')
                    
                    # Réessayer la détection
                    try:
                        results = self.model(
                            frame_rgb,
                            conf=conf,
                            iou=iou,
                            half=False,  # Pas de half precision sur CPU
                            device='cpu',
                            augment=False  # Désactiver multi-scale pour économiser la mémoire
                        )
                        return results
                    except Exception as cpu_error:
                        self.logger.error(f"Échec de la détection sur CPU: {str(cpu_error)}")
                        return []
                    finally:
                        # Essayer de revenir au device original
                        try:
                            if old_device != 'cpu':
                                self.device = old_device
                                if self.model is not None:
                                    self.model.to(self.device)
                        except:
                            pass
                else:
                    raise  # Relancer l'erreur si ce n'est pas CUDA OOM
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la détection: {str(e)}")
            return []  # Retourner une liste vide en cas d'erreur
    
    def set_conf_threshold(self, threshold: float):
        """
        Définit le seuil de confiance
        
        Args:
            threshold: Nouveau seuil (0.0 - 1.0)
        """
        try:
            if 0.0 <= threshold <= 1.0:
                self.conf_threshold = threshold
            else:
                self.logger.warning(f"Seuil de confiance invalide: {threshold}, doit être entre 0.0 et 1.0")
        except Exception as e:
            self.logger.error(f"Erreur lors de la définition du seuil de confiance: {str(e)}")
    
    def set_iou_threshold(self, threshold: float):
        """
        Définit le seuil IoU
        
        Args:
            threshold: Nouveau seuil (0.0 - 1.0)
        """
        try:
            if 0.0 <= threshold <= 1.0:
                self.iou_threshold = threshold
            else:
                self.logger.warning(f"Seuil IoU invalide: {threshold}, doit être entre 0.0 et 1.0")
        except Exception as e:
            self.logger.error(f"Erreur lors de la définition du seuil IoU: {str(e)}")
    
    def configure(self, **kwargs):
        """
        Configure le détecteur avec de multiples paramètres
        
        Args:
            **kwargs: Paramètres à configurer
        """
        try:
            needs_reload = False
            
            if 'conf_threshold' in kwargs:
                self.set_conf_threshold(kwargs['conf_threshold'])
            
            if 'iou_threshold' in kwargs:
                self.set_iou_threshold(kwargs['iou_threshold'])
            
            # Paramètres qui nécessitent un rechargement du modèle
            if 'model_path' in kwargs and kwargs['model_path'] != self.model_path:
                self.model_path = kwargs['model_path']
                needs_reload = True
            
            if 'use_cuda' in kwargs and kwargs['use_cuda'] != self.use_cuda:
                self.use_cuda = kwargs['use_cuda']
                self.device = self._select_device()
                needs_reload = True
            
            if 'half_precision' in kwargs and kwargs['half_precision'] != self.half_precision:
                self.half_precision = kwargs['half_precision']
                needs_reload = True
            
            # Recharger le modèle si nécessaire
            if needs_reload:
                success = self._load_model()
                if not success:
                    self.logger.error("Échec du rechargement du modèle après configuration")
                    
        except Exception as e:
            self.logger.error(f"Erreur lors de la configuration du détecteur: {str(e)}")
    
    def get_class_name(self, class_id: int) -> str:
        """
        Retourne le nom de la classe en français
        
        Args:
            class_id: Identifiant de la classe
            
        Returns:
            Nom de la classe en français
        """
        try:
            if self.model is None or not hasattr(self.model, 'names'):
                return f"classe_{class_id}"
                
            # Vérifier que l'ID de classe est valide
            if class_id < 0 or class_id >= len(self.model.names):
                return f"classe_{class_id}"
                
            original_name = self.model.names[class_id]
            return self.class_mapping.get(original_name, original_name)
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération du nom de classe: {str(e)}")
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
        if frame is None or frame.size == 0:
            return None
        
        try:
            # Redimensionner si nécessaire
            if target_size is not None:
                if target_size[0] > 0 and target_size[1] > 0:
                    try:
                        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
                    except Exception as resize_error:
                        self.logger.error(f"Erreur lors du redimensionnement: {str(resize_error)}")
                        # Continuer avec la frame originale
            
            # Conversion des couleurs optimisée
            try:
                # Convertir BGR -> RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception as color_error:
                self.logger.error(f"Erreur lors de la conversion des couleurs: {str(color_error)}")
                # Continuer avec la frame originale
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Erreur lors du prétraitement de la frame: {str(e)}")
            return frame  # Retourner la frame originale en cas d'erreur
    
    def enable_tracking(self, enabled: bool = True, method: str = 'bytetrack'):
        """
        Active ou désactive le tracking d'objets
        
        Args:
            enabled: True pour activer, False pour désactiver
            method: Méthode de tracking ('bytetrack', 'deepsort', etc.)
        """
        try:
            if self.model is not None:
                # Les versions récentes de YOLO peuvent avoir une API différente pour le tracking
                try:
                    if enabled:
                        self.model.tracker = method
                        self.logger.info(f"Tracking activé avec la méthode {method}")
                    else:
                        self.model.tracker = None
                        self.logger.info("Tracking désactivé")
                except AttributeError:
                    # Fallback pour d'autres versions de YOLO
                    try:
                        if enabled:
                            if hasattr(self.model, 'track'):
                                # Certaines versions utilisent model.track()
                                self.model.tracking = True
                                self.logger.info(f"Tracking activé avec la méthode par défaut")
                            else:
                                self.logger.warning("Cette version de YOLO ne semble pas supporter le tracking")
                        else:
                            if hasattr(self.model, 'track'):
                                self.model.tracking = False
                                self.logger.info("Tracking désactivé")
                    except:
                        self.logger.warning("Impossible de configurer le tracking")
        except Exception as e:
            self.logger.error(f"Erreur lors de la configuration du tracking: {str(e)}")
