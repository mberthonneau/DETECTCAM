#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module de gestion des alertes pour DETECTCAM
Gère l'envoi de notifications par email, webhook et système.
"""
import sys  # Ajout pour les références à sys.platform
import os
import smtplib
import requests
import json
import threading
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# Importation conditionnelle des bibliothèques de notification système
try:
    # Linux
    import notify2
    HAS_NOTIFY2 = True
except ImportError:
    HAS_NOTIFY2 = False

try:
    # Windows
    from win10toast import ToastNotifier
    HAS_WIN10TOAST = True
except ImportError:
    HAS_WIN10TOAST = False

try:
    # macOS
    import pync
    HAS_PYNC = True
except ImportError:
    HAS_PYNC = False

from utils.logger import get_module_logger

class AlertManager:
    """
    Gestionnaire d'alertes pour les détections
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le gestionnaire d'alertes
        
        Args:
            config: Configuration de l'application
        """
        self.logger = get_module_logger('AlertManager')
        
        # Configuration
        self.config = config
        self.alert_config = config.get('alerts', {})
        
        # État interne
        self.detection_counter = 0
        self.alert_threshold = self.alert_config.get('alert_threshold', 5)
        
        # Initialiser les notifications système si disponibles
        self._init_system_notifications()
        
        self.logger.info("Gestionnaire d'alertes initialisé")
    
    def _init_system_notifications(self):
        """Initialise les notifications système selon la plateforme"""
        try:
            # Linux (notify2)
            if HAS_NOTIFY2:
                notify2.init("DETECTCAM")
                self.logger.info("Notifications système initialisées (notify2)")
            
            # Windows (win10toast)
            elif HAS_WIN10TOAST:
                self.toaster = ToastNotifier()
                self.logger.info("Notifications système initialisées (win10toast)")
            
            # macOS (pync)
            elif HAS_PYNC:
                self.logger.info("Notifications système initialisées (pync)")
            
            else:
                self.logger.warning("Aucune bibliothèque de notification système disponible")
                
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation des notifications système: {str(e)}")
    
    def process_detection(self, detection_info: Dict[str, Any], 
                        image_path: Optional[str] = None) -> bool:
        """
        Traite une détection et déclenche des alertes si nécessaire
        
        Args:
            detection_info: Informations sur la détection
            image_path: Chemin de l'image de détection
            
        Returns:
            True si une alerte a été déclenchée, False sinon
        """
        # Incrémenter le compteur
        self.detection_counter += 1
        
        # Vérifier si le seuil est atteint
        if self.detection_counter >= self.alert_threshold:
            # Déclencher les alertes
            self.trigger_alerts(detection_info, image_path)
            
            # Réinitialiser le compteur
            self.detection_counter = 0
            return True
        
        return False
    
    def reset_counter(self):
        """Réinitialise le compteur de détection"""
        self.detection_counter = 0
    
    def trigger_alerts(self, detection_info: Dict[str, Any], 
                      image_path: Optional[str] = None,
                      force: bool = False):
        """
        Déclenche toutes les alertes configurées
        
        Args:
            detection_info: Informations sur la détection
            image_path: Chemin de l'image de détection
            force: Forcer le déclenchement même si désactivé
        """
        self.logger.info("Déclenchement des alertes")
        
        # Créer un thread pour chaque type d'alerte
        threads = []
        
        # Email
        if self.alert_config.get('email_enabled', False) or force:
            email_thread = threading.Thread(
                target=self._send_email_alert,
                args=(detection_info, image_path)
            )
            email_thread.daemon = True
            threads.append(email_thread)
        
        # Notification système
        if self.alert_config.get('notification_enabled', True) or force:
            notification_thread = threading.Thread(
                target=self._send_system_notification,
                args=(detection_info,)
            )
            notification_thread.daemon = True
            threads.append(notification_thread)
        
        # Webhook
        if self.alert_config.get('webhook_enabled', False) or force:
            webhook_thread = threading.Thread(
                target=self._send_webhook,
                args=(detection_info, image_path)
            )
            webhook_thread.daemon = True
            threads.append(webhook_thread)
        
        # Son d'alerte
        if self.alert_config.get('sound_alert', False) or force:
            sound_thread = threading.Thread(
                target=self._play_alert_sound
            )
            sound_thread.daemon = True
            threads.append(sound_thread)
        
        # Démarrer tous les threads
        for thread in threads:
            thread.start()
    
    def _send_email_alert(self, detection_info: Dict[str, Any], 
                        image_path: Optional[str] = None):
        """
        Envoie une alerte par email
        
        Args:
            detection_info: Informations sur la détection
            image_path: Chemin de l'image de détection
        """
        try:
            # Récupérer les paramètres
            recipient = self.alert_config.get('email_address', '')
            smtp_server = self.alert_config.get('smtp_server', 'smtp.gmail.com')
            smtp_port = self.alert_config.get('smtp_port', 587)
            username = self.alert_config.get('email_user', '')
            password = self.alert_config.get('email_password', '')
            
            # Vérifier que les paramètres sont valides
            if not recipient or not smtp_server or not username or not password:
                self.logger.error("Configuration email incomplète")
                return
            
            # Créer le message
            msg = MIMEMultipart()
            msg['Subject'] = f"DETECTCAM - Alerte de détection"
            msg['From'] = username
            msg['To'] = recipient
            
            # Créer le contenu HTML
            html = f"""
            <html>
            <body>
            <h2>Alerte de détection</h2>
            <p>Une détection a été enregistrée le {datetime.now().strftime('%Y-%m-%d à %H:%M:%S')}</p>
            <p><strong>Détails:</strong></p>
            <ul>
            """
            
            # Ajouter les détails de la détection
            if 'class_name' in detection_info:
                html += f"<li>Objet: {detection_info['class_name']}</li>"
            if 'zone' in detection_info:
                zone_name = f"Zone {detection_info['zone']}"
                html += f"<li>Zone: {zone_name}</li>"
            if 'confidence' in detection_info:
                confidence = detection_info['confidence']
                if isinstance(confidence, (int, float)):
                    html += f"<li>Confiance: {confidence:.2f}</li>"
                else:
                    html += f"<li>Confiance: {confidence}</li>"
            
            html += """
            </ul>
            <p>Cette alerte a été envoyée automatiquement par DETECTCAM.</p>
            </body>
            </html>
            """
            
            # Ajouter le contenu HTML
            msg.attach(MIMEText(html, 'html'))
            
            # Ajouter l'image si disponible
            if image_path and os.path.exists(image_path):
                with open(image_path, 'rb') as img_file:
                    img = MIMEImage(img_file.read())
                    img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
                    msg.attach(img)
            
            # Connexion au serveur SMTP et envoi
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)
            
            self.logger.info(f"Alerte email envoyée à {recipient}")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'envoi de l'alerte email: {str(e)}")
    
    def _send_system_notification(self, detection_info: Dict[str, Any]):
        """
        Envoie une notification système
        
        Args:
            detection_info: Informations sur la détection
        """
        try:
            # Créer le titre et le message
            title = "DETECTCAM - Alerte de détection"
            
            # Créer le message
            message_parts = []
            
            if 'class_name' in detection_info:
                message_parts.append(f"Objet: {detection_info['class_name']}")
            if 'zone' in detection_info:
                zone_name = f"Zone {detection_info['zone']}"
                message_parts.append(f"Zone: {zone_name}")
            
            message = "\n".join(message_parts) if message_parts else "Détection de mouvement"
            
            # Envoyer la notification selon la plateforme
            if HAS_NOTIFY2:
                # Linux
                n = notify2.Notification(title, message)
                n.show()
                self.logger.info("Notification système envoyée (notify2)")
                
            elif HAS_WIN10TOAST:
                # Windows
                self.toaster.show_toast(
                    title,
                    message,
                    duration=5,
                    threaded=True
                )
                self.logger.info("Notification système envoyée (win10toast)")
                
            elif HAS_PYNC:
                # macOS
                pync.notify(
                    message,
                    title=title,
                    sound=True
                )
                self.logger.info("Notification système envoyée (pync)")
                
            else:
                self.logger.warning("Aucune bibliothèque de notification système disponible")
                
        except Exception as e:
            self.logger.error(f"Erreur lors de l'envoi de la notification système: {str(e)}")
    
    def _send_webhook(self, detection_info: Dict[str, Any], 
                    image_path: Optional[str] = None):
        """
        Envoie une alerte via webhook
        
        Args:
            detection_info: Informations sur la détection
            image_path: Chemin de l'image de détection
        """
        try:
            webhook_url = self.alert_config.get('webhook_url', '')
            
            if not webhook_url:
                self.logger.error("URL de webhook non définie")
                return
            
            # Créer les données à envoyer
            webhook_data = {
                'event_type': 'detection',
                'timestamp': datetime.now().isoformat(),
                'detection': detection_info
            }
            
            # Pour une intégration Discord
            if 'discord.com' in webhook_url:
                self._send_discord_webhook(webhook_url, detection_info, image_path)
                return
            
            # Pour une intégration Slack
            if 'hooks.slack.com' in webhook_url:
                self._send_slack_webhook(webhook_url, detection_info, image_path)
                return
            
            # Webhook générique (JSON)
            response = requests.post(
                webhook_url,
                json=webhook_data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code < 200 or response.status_code >= 300:
                self.logger.error(f"Erreur de webhook: {response.status_code} {response.text}")
            else:
                self.logger.info(f"Webhook envoyé: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Erreur lors de l'envoi du webhook: {str(e)}")
    
    def _send_discord_webhook(self, webhook_url: str, 
                            detection_info: Dict[str, Any], 
                            image_path: Optional[str] = None):
        """
        Envoie une alerte via un webhook Discord
        
        Args:
            webhook_url: URL du webhook Discord
            detection_info: Informations sur la détection
            image_path: Chemin de l'image de détection
        """
        try:
            # Créer l'embed Discord
            timestamp = datetime.now().isoformat()
            
            # Créer la description
            description = "Une détection a été enregistrée.\n\n"
            
            if 'class_name' in detection_info:
                description += f"**Objet**: {detection_info['class_name']}\n"
            if 'zone' in detection_info:
                zone_name = f"Zone {detection_info['zone']}"
                description += f"**Zone**: {zone_name}\n"
            if 'confidence' in detection_info:
                confidence = detection_info['confidence']
                if isinstance(confidence, (int, float)):
                    description += f"**Confiance**: {confidence:.2f}\n"
                else:
                    description += f"**Confiance**: {confidence}\n"
            
            # Créer l'embed
            embed = {
                'title': 'DETECTCAM - Alerte de détection',
                'description': description,
                'color': 15158332,  # Rouge
                'timestamp': timestamp
            }
            
            # Ajouter l'image
            if image_path and os.path.exists(image_path):
                # Discord ne permet pas d'envoyer directement des fichiers via webhook
                # On peut seulement fournir une URL d'image
                # Dans ce cas, on pourrait implémenter un serveur web temporaire ou utiliser un service de partage
                embed['footer'] = {
                    'text': 'Une image de la détection est disponible dans l\'application'
                }
            
            # Construire le payload
            payload = {
                'embeds': [embed]
            }
            
            # Envoyer le webhook
            response = requests.post(
                webhook_url,
                json=payload
            )
            
            if response.status_code < 200 or response.status_code >= 300:
                self.logger.error(f"Erreur de webhook Discord: {response.status_code} {response.text}")
            else:
                self.logger.info("Webhook Discord envoyé")
                
        except Exception as e:
            self.logger.error(f"Erreur lors de l'envoi du webhook Discord: {str(e)}")
    
    def _send_slack_webhook(self, webhook_url: str, 
                          detection_info: Dict[str, Any], 
                          image_path: Optional[str] = None):
        """
        Envoie une alerte via un webhook Slack
        
        Args:
            webhook_url: URL du webhook Slack
            detection_info: Informations sur la détection
            image_path: Chemin de l'image de détection
        """
        try:
            # Créer le texte
            text = "DETECTCAM - Alerte de détection"
            
            # Créer les blocs
            blocks = [
                {
                    'type': 'header',
                    'text': {
                        'type': 'plain_text',
                        'text': 'DETECTCAM - Alerte de détection'
                    }
                },
                {
                    'type': 'section',
                    'text': {
                        'type': 'mrkdwn',
                        'text': f"Une détection a été enregistrée le {datetime.now().strftime('%Y-%m-%d à %H:%M:%S')}"
                    }
                }
            ]
            
            # Ajouter les détails
            fields = []
            
            if 'class_name' in detection_info:
                fields.append({
                    'type': 'mrkdwn',
                    'text': f"*Objet:*\n{detection_info['class_name']}"
                })
            
            if 'zone' in detection_info:
                zone_name = f"Zone {detection_info['zone']}"
                fields.append({
                    'type': 'mrkdwn',
                    'text': f"*Zone:*\n{zone_name}"
                })
            
            if 'confidence' in detection_info:
                confidence = detection_info['confidence']
                if isinstance(confidence, (int, float)):
                    confidence_text = f"{confidence:.2f}"
                else:
                    confidence_text = f"{confidence}"
                
                fields.append({
                    'type': 'mrkdwn',
                    'text': f"*Confiance:*\n{confidence_text}"
                })
            
            if fields:
                blocks.append({
                    'type': 'section',
                    'fields': fields
                })
            
            # Ajouter l'image
            if image_path and os.path.exists(image_path):
                # Slack ne permet pas d'envoyer directement des fichiers via webhook
                blocks.append({
                    'type': 'context',
                    'elements': [
                        {
                            'type': 'mrkdwn',
                            'text': 'Une image de la détection est disponible dans l\'application'
                        }
                    ]
                })
            
            # Construire le payload
            payload = {
                'text': text,
                'blocks': blocks
            }
            
            # Envoyer le webhook
            response = requests.post(
                webhook_url,
                json=payload
            )
            
            if response.status_code < 200 or response.status_code >= 300:
                self.logger.error(f"Erreur de webhook Slack: {response.status_code} {response.text}")
            else:
                self.logger.info("Webhook Slack envoyé")
                
        except Exception as e:
            self.logger.error(f"Erreur lors de l'envoi du webhook Slack: {str(e)}")
    
    def _play_alert_sound(self):
        """Joue le son d'alerte configuré"""
        try:
            sound_file = self.alert_config.get('sound_file', '')
            
            if not sound_file or not os.path.exists(sound_file):
                self.logger.error("Fichier son d'alerte non trouvé")
                return
            
            # Déterminer la plateforme
            if os.name == 'posix':
                # Linux ou macOS
                os.system(f"afplay {sound_file}" if sys.platform == 'darwin' else f"aplay {sound_file}")
            elif os.name == 'nt':
                # Windows
                import winsound
                winsound.PlaySound(sound_file, winsound.SND_FILENAME)
            
            self.logger.info(f"Son d'alerte joué: {sound_file}")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la lecture du son d'alerte: {str(e)}")
    
    def test_all_alerts(self, image_path: Optional[str] = None) -> Dict[str, bool]:
        """
        Teste toutes les alertes configurées
        
        Args:
            image_path: Chemin d'une image de test
            
        Returns:
            Dictionnaire des résultats de test
        """
        # Créer des informations de détection factices
        test_detection = {
            'class_name': 'Test',
            'zone': 'Test',
            'confidence': 0.99,
            'time': datetime.now().isoformat()
        }
        
        # Résultats des tests
        results = {
            'email': False,
            'notification': False,
            'webhook': False,
            'sound': False
        }
        
        # Tester l'email
        if self.alert_config.get('email_enabled', False):
            try:
                self._send_email_alert(test_detection, image_path)
                results['email'] = True
            except Exception as e:
                self.logger.error(f"Test email échoué: {str(e)}")
        
        # Tester la notification système
        if self.alert_config.get('notification_enabled', True):
            try:
                self._send_system_notification(test_detection)
                results['notification'] = True
            except Exception as e:
                self.logger.error(f"Test notification système échoué: {str(e)}")
        
        # Tester le webhook
        if self.alert_config.get('webhook_enabled', False):
            try:
                self._send_webhook(test_detection, image_path)
                results['webhook'] = True
            except Exception as e:
                self.logger.error(f"Test webhook échoué: {str(e)}")
        
        # Tester le son
        if self.alert_config.get('sound_alert', False):
            try:
                self._play_alert_sound()
                results['sound'] = True
            except Exception as e:
                self.logger.error(f"Test son échoué: {str(e)}")
        
        return results

