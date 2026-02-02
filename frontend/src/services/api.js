const API_BASE_URL = process.env.REACT_APP_API_URL;

export const chatService = {
  async streamChat(question, sessionId, conversationHistory, onMessage) {
    const response = await fetch(`${API_BASE_URL}/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question,
        session_id: sessionId,
        conversation_history: conversationHistory
      })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n').filter(line => line.trim());

      for (const line of lines) {
        try {
          const data = JSON.parse(line);
          onMessage(data);

          // 🔥 Stop stream immediately on error or completion
          if (data.type === 'error' || data.type === 'complete') {
            await reader.cancel();
            return;
          }
        } catch (e) {
          console.error('Parse error:', e);
        }
      }
    }
  },

  async getSheets() {
    const response = await fetch(`${API_BASE_URL}/sheets`);
    return response.json();
  },

  async reindex() {
    const response = await fetch(`${API_BASE_URL}/reindex`, {
      method: 'POST'
    });
    return response.json();
  }
};
