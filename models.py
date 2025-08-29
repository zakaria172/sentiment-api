from pydantic import BaseModel, Field, field_validator
from typing import Literal

class TextRequest(BaseModel):
    """Modèle pour la requête d'analyse de sentiment"""
    text: str = Field(
        ...,
        description="Texte à analyser pour le sentiment",
        max_length=5000,
        json_schema_extra={"example": "I love this product! It's amazing."}
    )

    @field_validator('text')
    def validate_text(cls, v: str) -> str:
        """Valider que le texte n'est pas vide après suppression des espaces"""
        if not v or not v.strip():
            raise ValueError('Le texte ne peut pas être vide')
        return v.strip()


class SentimentResponse(BaseModel):
    """Modèle pour la réponse d'analyse de sentiment"""
    sentiment: Literal["POSITIVE", "NEGATIVE"] = Field(
        ...,
        description="Sentiment détecté dans le texte"
    )
    confidence: float = Field(
        ...,
        description="Score de confiance entre 0 et 1",
        ge=0.0,
        le=1.0
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "sentiment": "POSITIVE",
                "confidence": 0.9947
            }
        }
    }


class ErrorResponse(BaseModel):
    """Modèle pour les réponses d'erreur"""
    detail: str = Field(
        ...,
        description="Description de l'erreur",
        json_schema_extra={"example": "Le texte ne peut pas être vide"}
    )
