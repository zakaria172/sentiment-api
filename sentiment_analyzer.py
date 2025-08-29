from transformers import pipeline
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Analyseur de sentiment utilisant DistilBERT"""
    
    def __init__(self):
        """Initialise l'analyseur avec le modèle imposé"""
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        
        try:
            logger.info(f"🔄 Chargement du modèle {self.model_name}...")
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=-1,  # Force CPU pour compatibilité déploiement
                return_all_scores=True
            )
            logger.info(f"✅ Modèle {self.model_name} chargé avec succès")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du modèle: {e}")
            raise RuntimeError(f"Impossible de charger le modèle: {e}")
    
    def analyze(self, text: str) -> Dict[str, any]:
        """
        Analyse le sentiment d'un texte
        
        Args:
            text (str): Texte à analyser
            
        Returns:
            Dict: {"sentiment": str, "confidence": float}
        """
        try:
            # Analyse du sentiment
            results = self.pipeline(text)
            
            # Le modèle retourne tous les scores, on prend celui avec le plus haut score
            best_result = max(results[0], key=lambda x: x['score'])
            
            sentiment = best_result['label']
            confidence = round(best_result['score'], 4)
            
            logger.debug(f"Résultat brut: {results}")
            logger.info(f"Sentiment analysé: {sentiment} (confiance: {confidence})")
            
            return {
                "sentiment": sentiment,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'analyse: {e}")
            raise RuntimeError(f"Erreur lors de l'analyse du sentiment: {e}")
    
    def get_model_info(self) -> Dict[str, str]:
        """Retourne les informations sur le modèle"""
        return {
            "model_name": self.model_name,
            "task": "sentiment-analysis",
            "framework": "transformers/pytorch"
        }