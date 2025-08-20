import csv
import json
import os
from datetime import datetime
from typing import Dict, List, Any
import uuid

class ConversationLogger:
    """Gestisce il salvataggio delle conversazioni in formato CSV e JSON"""
    
    def __init__(self, data_dir: str = "data/conversations"):
        self.data_dir = data_dir
        self.csv_file = os.path.join(data_dir, "conversations.csv")
        self.json_file = os.path.join(data_dir, "conversations.json")
        
        # Crea directory se non esiste
        os.makedirs(data_dir, exist_ok=True)
        
        # Inizializza file CSV con header se non esiste
        self._init_csv_file()
        
        # Inizializza file JSON se non esiste
        self._init_json_file()
    
    def _init_csv_file(self):
        """Inizializza il file CSV con gli header se non esiste"""
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'conversation_id', 'message_id', 
                    'role', 'content', 'tokens_used', 'response_time_ms',
                    'quantum_state', 'session_id'
                ])
    
    def _init_json_file(self):
        """Inizializza il file JSON se non esiste"""
        if not os.path.exists(self.json_file):
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump({"conversations": []}, f, indent=2)
    
    def log_message(self, 
                   conversation_id: str,
                   role: str,
                   content: str,
                   tokens_used: int = 0,
                   response_time_ms: int = 0,
                   quantum_state: str = "stable",
                   session_id: str = None) -> str:
        """Salva un messaggio sia in CSV che in JSON"""
        
        message_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Salva in CSV
        self._save_to_csv({
            'timestamp': timestamp,
            'conversation_id': conversation_id,
            'message_id': message_id,
            'role': role,
            'content': content,
            'tokens_used': tokens_used,
            'response_time_ms': response_time_ms,
            'quantum_state': quantum_state,
            'session_id': session_id or conversation_id
        })
        
        # Salva in JSON
        self._save_to_json({
            'timestamp': timestamp,
            'conversation_id': conversation_id,
            'message_id': message_id,
            'role': role,
            'content': content,
            'metadata': {
                'tokens_used': tokens_used,
                'response_time_ms': response_time_ms,
                'quantum_state': quantum_state,
                'session_id': session_id or conversation_id
            }
        })
        
        return message_id
    
    def _save_to_csv(self, message_data: Dict[str, Any]):
        """Salva un messaggio nel file CSV"""
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                message_data['timestamp'],
                message_data['conversation_id'],
                message_data['message_id'],
                message_data['role'],
                message_data['content'],
                message_data['tokens_used'],
                message_data['response_time_ms'],
                message_data['quantum_state'],
                message_data['session_id']
            ])
    
    def _save_to_json(self, message_data: Dict[str, Any]):
        """Salva un messaggio nel file JSON"""
        # Leggi il file JSON esistente
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {"conversations": []}
        
        # Trova o crea la conversazione
        conversation_id = message_data['conversation_id']
        conversation = None
        
        for conv in data['conversations']:
            if conv['conversation_id'] == conversation_id:
                conversation = conv
                break
        
        if conversation is None:
            conversation = {
                'conversation_id': conversation_id,
                'created_at': message_data['timestamp'],
                'messages': []
            }
            data['conversations'].append(conversation)
        
        # Aggiungi il messaggio
        conversation['messages'].append({
            'message_id': message_data['message_id'],
            'timestamp': message_data['timestamp'],
            'role': message_data['role'],
            'content': message_data['content'],
            'metadata': message_data['metadata']
        })
        
        # Salva il file JSON aggiornato
        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Recupera la cronologia di una conversazione dal JSON"""
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for conv in data['conversations']:
                if conv['conversation_id'] == conversation_id:
                    return conv['messages']
            
            return []
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def get_all_conversations(self) -> List[Dict[str, Any]]:
        """Recupera tutte le conversazioni dal JSON"""
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('conversations', [])
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def export_conversation_csv(self, conversation_id: str, output_file: str = None) -> str:
        """Esporta una singola conversazione in CSV"""
        if output_file is None:
            output_file = os.path.join(self.data_dir, f"conversation_{conversation_id}.csv")
        
        messages = self.get_conversation_history(conversation_id)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'role', 'content', 'tokens_used', 'response_time_ms'])
            
            for msg in messages:
                writer.writerow([
                    msg['timestamp'],
                    msg['role'],
                    msg['content'],
                    msg['metadata'].get('tokens_used', 0),
                    msg['metadata'].get('response_time_ms', 0)
                ])
        
        return output_file
    
    def get_stats(self) -> Dict[str, Any]:
        """Restituisce statistiche sulle conversazioni"""
        conversations = self.get_all_conversations()
        
        total_conversations = len(conversations)
        total_messages = sum(len(conv['messages']) for conv in conversations)
        total_tokens = 0
        
        for conv in conversations:
            for msg in conv['messages']:
                total_tokens += msg['metadata'].get('tokens_used', 0)
        
        return {
            'total_conversations': total_conversations,
            'total_messages': total_messages,
            'total_tokens_used': total_tokens,
            'average_messages_per_conversation': total_messages / max(total_conversations, 1)
        }