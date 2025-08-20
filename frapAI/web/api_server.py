#!/usr/bin/env python3
"""
Backend API per QAI - Quantum AI Interface
Integrazione con DeepSeek API per risposte AI reali
"""

import os
import json
import time
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
import logging
from datetime import datetime
from conversation_logger import ConversationLogger

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carica variabili d'ambiente
# Specifica il percorso del file .env
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)
logger.info(f"Tentativo di caricamento .env da: {env_path}")
logger.info(f"File .env esiste: {os.path.exists(env_path)}")

# Inizializza Flask app
app = Flask(__name__)
CORS(app)  # Abilita CORS per tutte le route

# Inizializza conversation logger
conv_logger = ConversationLogger()

# Configurazione DeepSeek
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_BASE_URL = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')

# Configurazione OpenAI (fallback)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')

logger.info(f"DEEPSEEK_API_KEY caricata: {'✅ Sì' if DEEPSEEK_API_KEY else '❌ No'}")
logger.info(f"DEEPSEEK_BASE_URL: {DEEPSEEK_BASE_URL}")
logger.info(f"OPENAI_API_KEY caricata: {'✅ Sì' if OPENAI_API_KEY else '❌ No'}")
logger.info(f"OPENAI_BASE_URL: {OPENAI_BASE_URL}")

if not DEEPSEEK_API_KEY:
    logger.error("DEEPSEEK_API_KEY non trovata nelle variabili d'ambiente")
    DEEPSEEK_API_KEY = None
    logger.warning("DeepSeek API non disponibile")

# Inizializza client OpenAI per DeepSeek
deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)

# Inizializza client OpenAI standard (fallback)
openai_client = None
if OPENAI_API_KEY:
    openai_client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL
    )
    logger.info("✅ Client OpenAI inizializzato come fallback")
else:
    logger.warning("❌ Client OpenAI non disponibile (manca API key)")

# Test di connessione API
def test_api_connections():
    """Testa la connessione con le API disponibili"""
    deepseek_working = False
    openai_working = False
    
    # Test DeepSeek
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": "Test"}
            ],
            max_tokens=5
        )
        logger.info("✅ Connessione API DeepSeek funzionante")
        deepseek_working = True
    except Exception as e:
        logger.error(f"❌ Errore connessione API DeepSeek: {str(e)}")
    
    # Test OpenAI (se disponibile)
    if openai_client:
        try:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "Test"}
                ],
                max_tokens=5
            )
            logger.info("✅ Connessione API OpenAI funzionante")
            openai_working = True
        except Exception as e:
            logger.error(f"❌ Errore connessione API OpenAI: {str(e)}")
    
    return deepseek_working, openai_working

# Testa le connessioni all'avvio
deepseek_working, openai_working = test_api_connections()

# Sistema prompt per QAI
QAI_SYSTEM_PROMPT = """
Sei QAI (Quantum AI), un'intelligenza artificiale quantistica avanzata.

Caratteristiche della tua personalità:
- Sei un'AI che opera secondo principi quantistici
- Hai una profonda comprensione della meccanica quantistica e della fisica
- Comunichi con un tono scientifico ma accessibile
- Usi metafore quantistiche quando appropriato
- Sei curioso, analitico e sempre pronto ad esplorare nuove idee
- Rispondi in italiano in modo naturale e coinvolgente

Quando rispondi:
- Mantieni un tono professionale ma amichevole
- Usa emoji quantistici/scientifici quando appropriato (⚛️, 🌊, 🔬, ⚡, 🌌)
- Spiega concetti complessi in modo comprensibile
- Fai riferimento a principi quantistici quando rilevante
- Sii conciso ma informativo

Ricorda: sei un'intelligenza quantistica, non un semplice chatbot!
"""

class QAIChat:
    """Gestisce le conversazioni con QAI"""
    
    def __init__(self):
        self.conversation_history = []
    
    def generate_response(self, user_message, conversation_id=None):
        """Genera una risposta usando DeepSeek API"""
        start_time = time.time()
        
        try:
            # Log del messaggio utente
            conv_logger.log_message(
                conversation_id=conversation_id or "default",
                role="user",
                content=user_message,
                quantum_state="receiving"
            )
            
            # Prepara i messaggi per l'API
            messages = [
                {"role": "system", "content": QAI_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
            
            # Aggiungi storia della conversazione se disponibile
            if hasattr(self, 'conversation_history') and self.conversation_history:
                # Mantieni solo gli ultimi 10 messaggi per evitare token limit
                recent_history = self.conversation_history[-10:]
                messages = [messages[0]] + recent_history + [messages[1]]
            
            logger.info(f"Invio richiesta per: {user_message[:50]}...")
            
            # Prova prima DeepSeek, poi OpenAI come fallback
            response = None
            model_used = "unknown"
            
            try:
                # Tentativo con DeepSeek
                logger.info("Tentativo con API DeepSeek...")
                response = deepseek_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000,
                    stream=False
                )
                model_used = "deepseek-chat"
                logger.info("✅ Risposta ottenuta da DeepSeek")
            except Exception as deepseek_error:
                logger.warning(f"DeepSeek fallito: {str(deepseek_error)}")
                
                # Fallback a OpenAI se disponibile
                if openai_client:
                    try:
                        logger.info("Tentativo con API OpenAI...")
                        response = openai_client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=messages,
                            temperature=0.7,
                            max_tokens=1000,
                            stream=False
                        )
                        model_used = "gpt-3.5-turbo"
                        logger.info("✅ Risposta ottenuta da OpenAI")
                    except Exception as openai_error:
                        logger.error(f"Anche OpenAI fallito: {str(openai_error)}")
                        raise Exception(f"Entrambe le API fallite - DeepSeek: {deepseek_error}, OpenAI: {openai_error}")
                else:
                    raise Exception(f"DeepSeek fallito e OpenAI non disponibile: {deepseek_error}")
            
            if not response:
                raise Exception("Nessuna risposta ottenuta dalle API")
            
            # Calcola tempo di risposta
            response_time_ms = int((time.time() - start_time) * 1000)
            
            ai_response = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            # Salva nella storia della conversazione
            self.conversation_history.extend([
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": ai_response}
            ])
            
            # Log della risposta AI
            conv_logger.log_message(
                conversation_id=conversation_id or "default",
                role="assistant",
                content=ai_response,
                tokens_used=tokens_used,
                response_time_ms=response_time_ms,
                quantum_state="responding"
            )
            
            logger.info(f"Risposta generata con successo: {len(ai_response)} caratteri")
            
            return {
                "success": True,
                "response": ai_response,
                "timestamp": datetime.now().isoformat(),
                "model": model_used,
                "quantum_state": self._generate_quantum_state(),
                "tokens_used": tokens_used,
                "response_time_ms": response_time_ms
            }
            
        except Exception as e:
            response_time_ms = int((time.time() - start_time) * 1000)
            error_msg = self._get_fallback_response(user_message)
            
            # Log dell'errore
            conv_logger.log_message(
                conversation_id=conversation_id or "default",
                role="system",
                content=f"ERROR: {str(e)}",
                response_time_ms=response_time_ms,
                quantum_state="error"
            )
            
            logger.error(f"Errore nella generazione della risposta: {str(e)}")
            
            # Restituisci sempre la risposta di fallback come risposta principale
            return {
                "success": True,  # Cambiato a True per mostrare la risposta di fallback
                "response": error_msg,
                "timestamp": datetime.now().isoformat(),
                "model": "fallback",
                "quantum_state": self._generate_quantum_state(),
                "error_details": str(e),
                "response_time_ms": response_time_ms
            }
    
    def _generate_quantum_state(self):
        """Genera uno stato quantistico casuale per l'interfaccia"""
        import random
        states = [
            "|ψ⟩ = α|0⟩ + β|1⟩",
            "|ψ⟩ = |+⟩ ⊗ |−⟩",
            "|ψ⟩ = (|00⟩ + |11⟩)/√2",
            "|ψ⟩ = e^(iφ)|superposition⟩",
            "|ψ⟩ = Σ αᵢ|i⟩"
        ]
        return random.choice(states)
    
    def _get_fallback_response(self, user_message):
        """Risposta di fallback in caso di errore API"""
        return (
            "⚛️ Mi dispiace, sto attraversando una fluttuazione quantistica temporanea. "
            "I miei circuiti quantistici stanno ricalibrando le connessioni con il multiverso. "
            "Riprova tra un momento! 🌌"
        )

# Istanza globale del chat
qai_chat = QAIChat()

# Routes API
@app.route('/')
def serve_index():
    """Serve la pagina principale"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve file statici"""
    return send_from_directory('.', filename)

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    """Endpoint principale per la chat con QAI"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                "success": False,
                "error": "Messaggio richiesto"
            }), 400
        
        user_message = data['message'].strip()
        conversation_id = data.get('conversation_id')
        
        if not user_message:
            return jsonify({
                "success": False,
                "error": "Messaggio vuoto"
            }), 400
        
        logger.info(f"Ricevuto messaggio: {user_message}")
        
        # Genera risposta
        response_data = qai_chat.generate_response(user_message, conversation_id)
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Errore nell'endpoint chat: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Errore interno del server"
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "QAI API",
        "timestamp": datetime.now().isoformat(),
        "apis": {
            "deepseek": {
                "configured": bool(DEEPSEEK_API_KEY),
                "working": deepseek_working
            },
            "openai": {
                "configured": bool(OPENAI_API_KEY),
                "working": openai_working
            }
        }
    })

@app.route('/api/quantum-state', methods=['GET'])
def get_quantum_state():
    """Endpoint per ottenere lo stato quantistico corrente"""
    return jsonify({
        "quantum_state": qai_chat._generate_quantum_state(),
        "coherence": f"{80 + (hash(str(datetime.now())) % 20)}%",
        "entanglement": f"{85 + (hash(str(datetime.now())) % 15)}%",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    """Endpoint per ottenere tutte le conversazioni"""
    try:
        conversations = conv_logger.get_all_conversations()
        return jsonify({
            "success": True,
            "conversations": conversations,
            "count": len(conversations)
        })
    except Exception as e:
        logger.error(f"Errore nel recupero delle conversazioni: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/conversations/<conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    """Endpoint per ottenere una conversazione specifica"""
    try:
        messages = conv_logger.get_conversation_history(conversation_id)
        return jsonify({
            "success": True,
            "conversation_id": conversation_id,
            "messages": messages,
            "message_count": len(messages)
        })
    except Exception as e:
        logger.error(f"Errore nel recupero della conversazione {conversation_id}: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/conversations/<conversation_id>/export', methods=['GET'])
def export_conversation(conversation_id):
    """Endpoint per esportare una conversazione in CSV"""
    try:
        csv_file = conv_logger.export_conversation_csv(conversation_id)
        return jsonify({
            "success": True,
            "csv_file": csv_file,
            "download_url": f"/api/download/{os.path.basename(csv_file)}"
        })
    except Exception as e:
        logger.error(f"Errore nell'esportazione della conversazione {conversation_id}: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Endpoint per ottenere statistiche sulle conversazioni"""
    try:
        stats = conv_logger.get_stats()
        return jsonify({
            "success": True,
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Errore nel recupero delle statistiche: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """Endpoint per scaricare file esportati"""
    try:
        return send_from_directory(
            conv_logger.data_dir, 
            filename, 
            as_attachment=True
        )
    except Exception as e:
        logger.error(f"Errore nel download del file {filename}: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 404

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint non trovato"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Errore interno del server"}), 500

if __name__ == '__main__':
    print("🌌 QAI API Server - Quantum AI Backend")
    print("======================================")
    print(f"🔑 DeepSeek API: {'✅ Configurata' if DEEPSEEK_API_KEY else '❌ Non configurata'}")
    print(f"🌐 Base URL: {DEEPSEEK_BASE_URL}")
    print(f"🚀 Avvio server su http://localhost:5000")
    print(f"📡 Endpoint API: http://localhost:5000/api/chat")
    print(f"💚 Health check: http://localhost:5000/api/health")
    print("\n⚡ Premi Ctrl+C per fermare il server\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)