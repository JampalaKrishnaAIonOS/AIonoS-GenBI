import React, { useState, useRef, useEffect } from 'react';
import { Send, RotateCw, Loader, Paperclip, ChevronLeft, LogOut, ArrowDown } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import MessageBubble from './MessageBubble';
import { chatService } from '../services/api';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingWords, setStreamingWords] = useState([]);
  const [currentMessage, setCurrentMessage] = useState({});
  const [sessionId] = useState(() => `session_${Date.now()}`);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const scrollRef = useRef(null);
  const navigate = useNavigate();

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const streamedAnswerRef = useRef('');
  const streamedMetaRef = useRef({ source: [], table: null, chart: null, code: null });

  const scrollToBottom = (behavior = 'smooth') => {
    messagesEndRef.current?.scrollIntoView({ behavior });
  };

  const handleScroll = () => {
    if (!scrollRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 100;
    setShowScrollButton(!isAtBottom);
  };

  useEffect(() => {
    const welcome = {
      answer: "Hi — I'm your **GenBI assistant**. I can help you analyze, visualize, and reason about your Excel data. How can I assist you today?",
      isUser: false
    };
    setMessages([welcome]);
  }, []);

  useEffect(() => {
    if (!showScrollButton) {
      scrollToBottom('auto');
    }
  }, [messages, streamingWords]);

  const handleStreamMessage = (data) => {
    switch (data.type) {
      case 'answer_start':
        streamedAnswerRef.current = '';
        setStreamingWords([]);
        streamedMetaRef.current = { source: [], table: null, chart: null, code: null };
        setCurrentMessage(prev => ({ ...prev, answer: '', source: [], table: null, chart: null, code: null }));
        break;

      case 'status':
        setCurrentMessage(prev => ({ ...prev, status: data.content }));
        break;

      case 'word':
        streamedAnswerRef.current += data.content;
        setStreamingWords(prev => [...prev, data.content]);
        break;

      case 'source':
        streamedMetaRef.current.source.push(data.content);
        setCurrentMessage(prev => ({
          ...prev,
          source: [...streamedMetaRef.current.source]
        }));
        break;

      case 'table':
        streamedMetaRef.current.table = data.content;
        setCurrentMessage(prev => ({ ...prev, table: data.content }));
        break;

      case 'chart':
        streamedMetaRef.current.chart = data.content;
        setCurrentMessage(prev => ({ ...prev, chart: data.content }));
        break;

      case 'code':
        streamedMetaRef.current.code = data.content;
        setCurrentMessage(prev => ({ ...prev, code: data.content }));
        break;

      case 'complete':
        const finalMsg = {
          answer: streamedAnswerRef.current,
          source: streamedMetaRef.current.source,
          table: streamedMetaRef.current.table,
          chart: streamedMetaRef.current.chart,
          // code: streamedMetaRef.current.code, // Removed: Don't store code in messages
          isUser: false
        };

        setMessages(prev => [...prev, finalMsg]);
        streamedAnswerRef.current = '';
        streamedMetaRef.current = { source: [], table: null, chart: null, code: null };
        setCurrentMessage({});
        setStreamingWords([]);
        setIsStreaming(false);
        break;

      case 'error':
        setMessages(prev => [...prev, { answer: `❌ ${data.content}`, isError: true }]);
        setIsStreaming(false);
        setStreamingWords([]);
        break;

      default:
        break;
    }
  };

  const handleSend = async () => {
    if (!input.trim() || isStreaming) return;

    const userMessage = { content: input, isUser: true };
    setMessages(prev => [...prev, userMessage]);

    const userQuery = input;
    setInput('');
    setIsStreaming(true);
    setCurrentMessage({ status: 'Connecting to server...' });

    const maxRetries = 3;
    let attempt = 0;
    let connected = false;

    while (attempt < maxRetries && !connected) {
      try {
        const conversationHistory = messages.map(msg => ({
          role: msg.isUser ? 'user' : 'assistant',
          content: msg.isUser ? msg.content : msg.answer || ''
        }));

        await chatService.streamChat(
          userQuery,
          sessionId,
          conversationHistory,
          handleStreamMessage
        );
        connected = true;
      } catch (error) {
        attempt++;
        console.error(`Attempt ${attempt} failed:`, error);

        if (attempt < maxRetries) {
          setCurrentMessage({ status: `Server is warming up, retrying... (Attempt ${attempt}/${maxRetries})` });
          // Wait 3 seconds before next retry
          await new Promise(resolve => setTimeout(resolve, 3000));
        } else {
          setMessages(prev => [...prev, {
            answer: `❌ Connection Error: The server appears to be offline or still starting up. Please try again in a minute.`,
            isError: true
          }]);
          setIsStreaming(false);
          setCurrentMessage(null);
        }
      }
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('genbi_auth');
    navigate('/login');
  };

  const handleRefresh = () => {
    const welcome = {
      answer: "Hi — I'm your **GenBI assistant**. I can help you analyze, visualize, and reason about your Excel data. How can I assist you today?",
      isUser: false
    };
    setMessages([welcome]);
    setInput('');
    chatService.reindex().catch(console.error);
  };

  return (
    <div className="chat-container">
      <header className="chat-header">
        <div className="header-left">
          <button className="back-btn">
            <ChevronLeft size={20} /> Back
          </button>
          <div className="bot-info">
            <div className="bot-avatar-header">G</div>
            <div className="bot-name-container">
              <h2>GenBI <span style={{ fontSize: '12px', fontWeight: 400, opacity: 0.7, marginLeft: '4px' }}>by AIonOS</span></h2>
            </div>
          </div>
        </div>

        <div className="header-right">
          <div className="status-badge live">
            <div className="status-dot"></div>
            Live
          </div>
          <button className="refresh-btn" onClick={handleRefresh}>
            <RotateCw size={14} /> Refresh
          </button>
          <button className="refresh-btn logout" onClick={handleLogout}>
            <LogOut size={14} /> Logout
          </button>
        </div>
      </header>

      <main
        className="messages-container"
        ref={scrollRef}
        onScroll={handleScroll}
      >
        {messages.map((msg, idx) => (
          <MessageBubble key={idx} message={msg} isUser={msg.isUser} />
        ))}

        {isStreaming && (
          <MessageBubble
            message={{ ...currentMessage, answer: streamedAnswerRef.current }}
            isUser={false}
          />
        )}

        <div ref={messagesEndRef} />

        {showScrollButton && (
          <button
            className="scroll-bottom-btn"
            onClick={() => scrollToBottom('smooth')}
          >
            <ArrowDown size={20} />
          </button>
        )}
      </main>

      <footer className="input-area">
        <div className="input-container-wrapper">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message..."
            disabled={isStreaming}
            rows={1}
            onInput={(e) => {
              e.target.style.height = 'auto';
              e.target.style.height = e.target.scrollHeight + 'px';
            }}
          />
          <button
            onClick={handleSend}
            disabled={isStreaming || !input.trim()}
            className="send-btn"
          >
            {isStreaming ? <Loader className="animate-spin" size={18} /> : <Send size={18} />}
          </button>
        </div>
      </footer>
    </div>
  );
};

export default ChatInterface;