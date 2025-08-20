// QAI - Quantum AI Interface JavaScript

class QuantumAI {
    constructor() {
        this.chatMessages = document.getElementById('chatMessages');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.quantumCanvas = document.getElementById('quantumCanvas');
        this.quantumState = document.getElementById('quantumState');
        
        this.isTyping = false;
        this.messageHistory = [];
        this.conversationId = this.generateConversationId();
        
        this.initializeEventListeners();
        this.initializeQuantumVisualizer();
        this.startQuantumAnimations();
        
        // Aggiungi pannello di controllo quantistico
        this.addQuantumControlPanel();
        
        // Aggiungi pannello statistiche conversazioni
        this.addConversationStatsPanel();
        
        // Avvia aggiornamenti periodici dello stato quantistico
        this.startQuantumUpdates();
        
        // Carica statistiche iniziali
        this.loadConversationStats();
    }

    generateConversationId() {
        return 'qai_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    initializeEventListeners() {
        // Send message on button click
        this.sendButton.addEventListener('click', () => this.sendMessage());
        
        // Send message on Enter key
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Quick action buttons
        document.querySelectorAll('.quick-action').forEach(button => {
            button.addEventListener('click', () => {
                const message = button.getAttribute('data-message');
                this.messageInput.value = message;
                this.sendMessage();
            });
        });
        
        // Auto-resize input
        this.messageInput.addEventListener('input', () => {
            this.adjustInputHeight();
        });
    }

    adjustInputHeight() {
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.isTyping) return;
        
        // Add user message
        this.addMessage(message, 'user');
        this.messageInput.value = '';
        this.adjustInputHeight();
        
        // Show typing indicator
        this.showTypingIndicator();
        
        // Simulate AI response delay
        await this.delay(1000 + Math.random() * 2000);
        
        // Generate AI response
        const response = await this.generateAIResponse(message);
        
        // Hide typing indicator and show response
        this.hideTypingIndicator();
        this.addMessage(response, 'ai');
        
        // Update quantum state
        this.updateQuantumState();
    }

    addMessage(content, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const timestamp = new Date().toLocaleTimeString('it-IT', {
            hour: '2-digit',
            minute: '2-digit'
        });
        
        const avatar = sender === 'ai' ? '<i class="fas fa-atom"></i>' : '<i class="fas fa-user"></i>';
        const senderName = sender === 'ai' ? 'QAI' : 'Tu';
        
        messageDiv.innerHTML = `
            <div class="message-avatar">
                ${avatar}
            </div>
            <div class="message-content">
                <div class="message-header">
                    <span class="sender-name">${senderName}</span>
                    <span class="timestamp">${timestamp}</span>
                </div>
                <div class="message-text">
                    ${this.formatMessage(content)}
                </div>
            </div>
        `;
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
        
        // Store in history
        this.messageHistory.push({ content, sender, timestamp });
    }

    formatMessage(content) {
        // Convert markdown-like formatting
        content = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        content = content.replace(/\*(.*?)\*/g, '<em>$1</em>');
        content = content.replace(/`(.*?)`/g, '<code>$1</code>');
        
        // Convert line breaks
        content = content.replace(/\n/g, '<br>');
        
        return content;
    }

    showTypingIndicator() {
        this.isTyping = true;
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message ai-message typing-message';
        typingDiv.id = 'typingIndicator';
        
        typingDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-atom"></i>
            </div>
            <div class="message-content">
                <div class="message-header">
                    <span class="sender-name">QAI</span>
                    <span class="timestamp">Ora</span>
                </div>
                <div class="message-text">
                    <div class="typing-indicator">
                        <span>Elaborando risposta quantistica</span>
                        <div class="typing-dots">
                            <div class="typing-dot"></div>
                            <div class="typing-dot"></div>
                            <div class="typing-dot"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        this.chatMessages.appendChild(typingDiv);
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
        this.isTyping = false;
    }

    async generateAIResponse(userMessage) {
        try {
            // Chiamata al backend API
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: userMessage,
                    conversation_id: this.conversationId || null
                })
            });

            const data = await response.json();

            if (data.success) {
                // Aggiorna stato quantistico se fornito
                if (data.quantum_state) {
                    this.updateQuantumStateFromAPI(data.quantum_state);
                }
                
                return data.response;
            } else {
                // Fallback in caso di errore
                console.error('Errore API:', data.error);
                return data.fallback_response || this.getFallbackResponse();
            }
        } catch (error) {
            console.error('Errore di connessione:', error);
            return this.getFallbackResponse();
        }
    }

    getFallbackResponse() {
        return `⚛️ Mi dispiace, sto attraversando una fluttuazione quantistica temporanea. 
        I miei circuiti quantistici stanno ricalibrando le connessioni con il multiverso. 
        Riprova tra un momento! 🌌`;
    }

    updateQuantumStateFromAPI(quantumState) {
        const stateValue = this.quantumState.querySelector('.state-value');
        if (stateValue) {
            stateValue.style.opacity = '0';
            setTimeout(() => {
                stateValue.textContent = quantumState;
                stateValue.style.opacity = '1';
            }, 300);
        }
    }

    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // Quantum Visualizer
    initializeQuantumVisualizer() {
        this.ctx = this.quantumCanvas.getContext('2d');
        this.waveData = [];
        this.time = 0;
        
        // Initialize wave data
        for (let i = 0; i < 100; i++) {
            this.waveData.push({
                amplitude: Math.random() * 0.5 + 0.5,
                frequency: Math.random() * 0.1 + 0.05,
                phase: Math.random() * Math.PI * 2
            });
        }
        
        this.animateWaveFunction();
    }

    animateWaveFunction() {
        const canvas = this.quantumCanvas;
        const ctx = this.ctx;
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw quantum wave function
        ctx.strokeStyle = '#00d4ff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        for (let x = 0; x < canvas.width; x++) {
            let y = canvas.height / 2;
            
            // Superposition of multiple waves
            for (let i = 0; i < this.waveData.length; i++) {
                const wave = this.waveData[i];
                y += wave.amplitude * Math.sin(
                    wave.frequency * x + wave.phase + this.time * 0.02
                ) * 30;
            }
            
            if (x === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        
        ctx.stroke();
        
        // Draw probability density
        ctx.fillStyle = 'rgba(0, 212, 255, 0.1)';
        ctx.beginPath();
        ctx.moveTo(0, canvas.height / 2);
        
        for (let x = 0; x < canvas.width; x++) {
            let y = canvas.height / 2;
            
            for (let i = 0; i < this.waveData.length; i++) {
                const wave = this.waveData[i];
                y += wave.amplitude * Math.sin(
                    wave.frequency * x + wave.phase + this.time * 0.02
                ) * 30;
            }
            
            ctx.lineTo(x, y);
        }
        
        ctx.lineTo(canvas.width, canvas.height / 2);
        ctx.closePath();
        ctx.fill();
        
        this.time++;
        requestAnimationFrame(() => this.animateWaveFunction());
    }

    updateQuantumState() {
        const states = [
            '|ψ⟩ = α|0⟩ + β|1⟩',
            '|ψ⟩ = |+⟩ ⊗ |−⟩',
            '|ψ⟩ = (|00⟩ + |11⟩)/√2',
            '|ψ⟩ = e^(iφ)|superposition⟩',
            '|ψ⟩ = Σ αᵢ|i⟩'
        ];
        
        const randomState = states[Math.floor(Math.random() * states.length)];
        const stateValue = this.quantumState.querySelector('.state-value');
        
        // Animate state change
        stateValue.style.opacity = '0';
        setTimeout(() => {
            stateValue.textContent = randomState;
            stateValue.style.opacity = '1';
        }, 300);
    }

    startQuantumAnimations() {
        // Update quantum stats periodically
        setInterval(() => {
            this.updateQuantumStats();
        }, 3000);
        
        // Update quantum state periodically
        setInterval(() => {
            this.updateQuantumState();
        }, 5000);
    }

    startQuantumUpdates() {
        // Aggiorna stato quantistico ogni 30 secondi
        setInterval(() => {
            this.updateQuantumStateFromAPI();
        }, 30000);
        
        // Aggiorna statistiche ogni 2 minuti
        setInterval(() => {
            this.loadConversationStats();
        }, 120000);
    }

    addQuantumControlPanel() {
        // Implementazione del pannello di controllo quantistico
        console.log('Quantum control panel initialized');
    }

    addConversationStatsPanel() {
        const sidebar = document.querySelector('.sidebar');
        
        const statsPanel = document.createElement('div');
        statsPanel.className = 'stats-panel';
        statsPanel.innerHTML = `
            <h3>📊 Statistiche Conversazioni</h3>
            <div class="stats-grid">
                <div class="stat-item">
                    <span class="stat-label">Conversazioni:</span>
                    <span class="stat-value" id="total-conversations">0</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Messaggi:</span>
                    <span class="stat-value" id="total-messages">0</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Token usati:</span>
                    <span class="stat-value" id="total-tokens">0</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Media msg/conv:</span>
                    <span class="stat-value" id="avg-messages">0</span>
                </div>
            </div>
            <div class="conversation-actions">
                <button onclick="quantumAI.exportCurrentConversation()" class="export-btn">
                    💾 Esporta Conversazione
                </button>
                <button onclick="quantumAI.showAllConversations()" class="view-btn">
                    📋 Vedi Tutte
                </button>
            </div>
        `;
        
        sidebar.appendChild(statsPanel);
    }

    async loadConversationStats() {
        try {
            const response = await fetch('/api/stats');
            const data = await response.json();
            
            if (data.success) {
                const stats = data.stats;
                document.getElementById('total-conversations').textContent = stats.total_conversations;
                document.getElementById('total-messages').textContent = stats.total_messages;
                document.getElementById('total-tokens').textContent = stats.total_tokens_used;
                document.getElementById('avg-messages').textContent = stats.average_messages_per_conversation.toFixed(1);
            }
        } catch (error) {
            console.error('Errore nel caricamento delle statistiche:', error);
        }
    }

    async exportCurrentConversation() {
        try {
            const response = await fetch(`http://localhost:5000/api/conversations/${this.conversationId}/export`);
            const data = await response.json();
            
            if (data.success) {
                // Scarica il file CSV
                window.open(`http://localhost:5000${data.download_url}`, '_blank');
                this.showNotification('✅ Conversazione esportata con successo!');
            } else {
                this.showNotification('❌ Errore nell\'esportazione');
            }
        } catch (error) {
            console.error('Errore nell\'esportazione:', error);
            this.showNotification('❌ Errore nell\'esportazione');
        }
    }

    async showAllConversations() {
        try {
            const response = await fetch('/api/conversations');
            const data = await response.json();
            
            if (data.success) {
                this.displayConversationsList(data.conversations);
            }
        } catch (error) {
            console.error('Errore nel caricamento delle conversazioni:', error);
        }
    }

    displayConversationsList(conversations) {
        const modal = document.createElement('div');
        modal.className = 'conversations-modal';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h2>📋 Tutte le Conversazioni</h2>
                    <button onclick="this.parentElement.parentElement.parentElement.remove()" class="close-btn">✕</button>
                </div>
                <div class="conversations-list">
                    ${conversations.map(conv => `
                        <div class="conversation-item">
                            <div class="conv-info">
                                <strong>ID:</strong> ${conv.conversation_id.substring(0, 8)}...<br>
                                <strong>Creata:</strong> ${new Date(conv.created_at).toLocaleString()}<br>
                                <strong>Messaggi:</strong> ${conv.messages.length}
                            </div>
                            <div class="conv-actions">
                                <button onclick="quantumAI.exportConversation('${conv.conversation_id}')" class="mini-btn">
                                    💾 Esporta
                                </button>
                                <button onclick="quantumAI.loadConversation('${conv.conversation_id}')" class="mini-btn">
                                    📖 Carica
                                </button>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
    }

    async exportConversation(conversationId) {
        try {
            const response = await fetch(`/api/conversations/${conversationId}/export`);
            const data = await response.json();
            
            if (data.success) {
                window.open(`http://localhost:5000${data.download_url}`, '_blank');
                this.showNotification('✅ Conversazione esportata!');
            }
        } catch (error) {
            console.error('Errore nell\'esportazione:', error);
        }
    }

    async loadConversation(conversationId) {
        try {
            const response = await fetch(`http://localhost:5000/api/conversations/${conversationId}`);
            const data = await response.json();
            
            if (data.success) {
                // Pulisci chat corrente
                this.chatMessages.innerHTML = '';
                
                // Carica messaggi della conversazione
                data.messages.forEach(msg => {
                    if (msg.role === 'user' || msg.role === 'assistant') {
                        this.addMessage(msg.content, msg.role === 'user' ? 'user' : 'ai');
                    }
                });
                
                // Aggiorna conversation ID
                this.conversationId = conversationId;
                
                this.showNotification('✅ Conversazione caricata!');
                
                // Chiudi modal
                document.querySelector('.conversations-modal')?.remove();
            }
        } catch (error) {
            console.error('Errore nel caricamento della conversazione:', error);
        }
    }

    showNotification(message) {
        const notification = document.createElement('div');
        notification.className = 'notification';
        notification.textContent = message;
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }

    updateQuantumStats() {
        const coherenceBar = document.querySelector('.stat-fill');
        const entanglementBar = document.querySelectorAll('.stat-fill')[1];
        const coherenceValue = document.querySelector('.stat-value');
        const entanglementValue = document.querySelectorAll('.stat-value')[1];
        
        // Generate realistic quantum values
        const coherence = Math.floor(Math.random() * 20 + 80); // 80-100%
        const entanglement = Math.floor(Math.random() * 15 + 85); // 85-100%
        
        // Animate bars
        coherenceBar.style.width = coherence + '%';
        entanglementBar.style.width = entanglement + '%';
        
        // Update values
        coherenceValue.textContent = coherence + '%';
        entanglementValue.textContent = entanglement + '%';
    }
}

// Initialize QAI when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new QuantumAI();
    
    // Add some startup effects
    setTimeout(() => {
        document.body.classList.add('loaded');
    }, 500);
});

// Add some interactive effects
document.addEventListener('mousemove', (e) => {
    const particles = document.querySelector('.particles');
    const x = e.clientX / window.innerWidth;
    const y = e.clientY / window.innerHeight;
    
    particles.style.transform = `translate(${x * 10}px, ${y * 10}px)`;
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + K to focus input
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        document.getElementById('messageInput').focus();
    }
    
    // Escape to clear input
    if (e.key === 'Escape') {
        document.getElementById('messageInput').value = '';
    }
});

// Add quantum glitch effect on logo click
document.querySelector('.logo').addEventListener('click', () => {
    const logo = document.querySelector('.logo h1');
    logo.style.animation = 'none';
    logo.style.filter = 'hue-rotate(180deg) saturate(2)';
    
    setTimeout(() => {
        logo.style.animation = '';
        logo.style.filter = '';
    }, 1000);
});