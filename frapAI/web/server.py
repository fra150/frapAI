#!/usr/bin/env python3
"""
Semplice server web per l'interfaccia QAI
"""

import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

class QAIHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler per servire i file dell'interfaccia QAI"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent), **kwargs)
    
    def end_headers(self):
        # Aggiungi headers per CORS e cache
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()
    
    def do_GET(self):
        # Serve index.html per la root
        if self.path == '/':
            self.path = '/index.html'
        return super().do_GET()
    
    def log_message(self, format, *args):
        # Log personalizzato
        print(f"🌐 {self.address_string()} - {format % args}")

def start_server(port=8000):
    """Avvia il server web per QAI"""
    
    # Cambia directory al folder web
    web_dir = Path(__file__).parent
    os.chdir(web_dir)
    
    print(f"🚀 Avvio server QAI...")
    print(f"📁 Directory: {web_dir}")
    print(f"🌐 Porta: {port}")
    
    try:
        with socketserver.TCPServer(("", port), QAIHTTPRequestHandler) as httpd:
            server_url = f"http://localhost:{port}"
            
            print(f"\n✨ QAI Interface attiva su: {server_url}")
            print(f"🔗 Apri il browser e vai a: {server_url}")
            print(f"⚡ Premi Ctrl+C per fermare il server\n")
            
            # Apri automaticamente il browser
            try:
                webbrowser.open(server_url)
                print(f"🌍 Browser aperto automaticamente")
            except Exception as e:
                print(f"⚠️  Impossibile aprire il browser automaticamente: {e}")
                print(f"   Apri manualmente: {server_url}")
            
            # Avvia il server
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print(f"\n🛑 Server fermato dall'utente")
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Porta {port} già in uso. Prova con un'altra porta.")
            print(f"   Esempio: python server.py --port 8001")
        else:
            print(f"❌ Errore nell'avvio del server: {e}")
    except Exception as e:
        print(f"❌ Errore imprevisto: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Server web per QAI Interface")
    parser.add_argument('--port', '-p', type=int, default=8000, 
                       help='Porta del server (default: 8000)')
    
    args = parser.parse_args()
    
    print("🌌 QAI - Quantum AI Interface Server")
    print("=====================================\n")
    
    start_server(args.port)