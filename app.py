from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
import logging
import os
from sentiment_analyzer import SentimentAnalyzer
from models import TextRequest, SentimentResponse, ErrorResponse

from fastapi.responses import JSONResponse
from starlette.status import HTTP_400_BAD_REQUEST

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Création de l'application FastAPI
app = FastAPI(
    title="Sentiment Analysis API",
    description="API d'analyse de sentiment utilisant DistilBERT",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montage des fichiers statiques
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialisation de l'analyseur de sentiment
try:
    analyzer = SentimentAnalyzer()
    logger.info("✅ Modèle d'analyse de sentiment chargé avec succès")
except Exception as e:
    logger.error(f"❌ Erreur lors du chargement du modèle: {e}")
    analyzer = None

@app.get("/")
async def serve_homepage():
    """Servir la page d'accueil ou un message JSON"""
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return JSONResponse(content={
        "message": "API d'analyse de sentiment opérationnelle",
        "docs": "/docs"
    })

@app.get("/health")
async def health_check():
    """Endpoint de santé pour vérifier le statut de l'API"""
    if analyzer is None:
        raise HTTPException(
            status_code=503,
            detail="Service indisponible - Modèle non chargé"
        )
    
    logger.info("✅ Health check - Service opérationnel")
    return {
        "status": "healthy",
        "model": "distilbert-base-uncased-finetuned-sst-2-english",
        "message": "Service d'analyse de sentiment opérationnel"
    }

@app.post("/predict", response_model=SentimentResponse, responses={400: {"model": ErrorResponse}})
async def predict_sentiment(request: TextRequest):
    """
    Analyser le sentiment d'un texte
    
    - **text**: Le texte à analyser (obligatoire, non vide)
    
    Retourne:
    - **sentiment**: POSITIVE ou NEGATIVE
    - **confidence**: Score de confiance entre 0 et 1
    """
    try:
        # Validation du modèle
        if analyzer is None:
            logger.error("❌ Tentative d'analyse avec modèle non chargé")
            raise HTTPException(
                status_code=503,
                detail="Service indisponible - Modèle non chargé"
            )
        
        # Validation du texte
        if not request.text or not request.text.strip():
            logger.warning("⚠️ Tentative d'analyse avec texte vide")
            raise HTTPException(
                status_code=400,
                detail="Le texte ne peut pas être vide"
            )
        
        # Analyse du sentiment
        logger.info(f"🔍 Analyse du texte: '{request.text[:50]}...'")
        result = analyzer.analyze(request.text.strip())
        
        response = SentimentResponse(
            sentiment=result["sentiment"],
            confidence=result["confidence"]
        )
        
        logger.info(f"✅ Analyse terminée - Sentiment: {response.sentiment}, Confiance: {response.confidence}")
        return response
        
    except HTTPException:
        # Re-lancer les erreurs HTTP
        raise
    except Exception as e:
        logger.error(f"❌ Erreur inattendue lors de l'analyse: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Erreur interne du serveur"
        )

@app.get("/models")
async def get_model_info():
    """Informations sur le modèle utilisé"""
    return {
        "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
        "model_type": "DistilBERT",
        "task": "sentiment-analysis",
        "languages": ["en"],
        "labels": ["NEGATIVE", "POSITIVE"]
    }

# Gestion des erreurs globales
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint non trouvé",
            "available_endpoints": ["/docs", "/predict", "/health"]
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Gestion personnalisée des erreurs de validation (422 → 400)"""
    return JSONResponse(
        status_code=HTTP_400_BAD_REQUEST,
        content={"detail": "Le texte ne peut pas être vide"}
    )



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)