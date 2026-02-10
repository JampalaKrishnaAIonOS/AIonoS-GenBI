import React, { useState, useRef, useEffect } from 'react';
import { Send, RotateCw, Loader, ChevronLeft, LogOut, ArrowDown } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import MessageBubble from './MessageBubble';
import StreamDebugger from './StreamDebugger';
import TableDisplay from './TableDisplay';
import ChartDisplay from './ChartDisplay';
import { chatService } from '../services/api';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [rawEvents, setRawEvents] = useState([]);
  const [showRawDebug, setShowRawDebug] = useState(false);
  const [markdownText, setMarkdownText] = useState("");
  const [tableData, setTableData] = useState(null);
  const [chartData, setChartData] = useState(null);
  const [currentStatus, setCurrentStatus] = useState("");
  const [currentMessage, setCurrentMessage] = useState({}); // Keep for metadata if needed
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
  }, [messages, markdownText, tableData, chartData, showScrollButton]);

  const handleStreamMessage = (data) => {
    // 🔍 LOG: Track ALL incoming events
    console.log('🌊 SSE EVENT RECEIVED:', data.type, data);

    // store raw events for debugging when enabled
    setRawEvents(prev => [data, ...prev].slice(0, 200));

    switch (data.type) {
      case 'answer_start':
        streamedAnswerRef.current = '';
        setMarkdownText("");
        setTableData(null);
        setChartData(null);
        streamedMetaRef.current = { source: [], table: null, chart: null, code: null };
        break;

      case 'status':
        setCurrentStatus(data.content);
        break;

      case 'word':
        streamedAnswerRef.current += data.content;
        setMarkdownText(prev => prev + data.content);
        break;

      case 'source':
        streamedMetaRef.current.source.push(data.content);
        setCurrentMessage(prev => ({
          ...prev,
          source: [...streamedMetaRef.current.source]
        }));
        break;


      case 'table':
        const tbl = data.content;
        console.log('📊 RAW TABLE EVENT:', JSON.stringify(tbl, null, 2));
        if (tbl && tbl.columns && Array.isArray(tbl.rows)) {
          console.log('✅ TABLE RECEIVED:', tbl.rows.length);
          streamedMetaRef.current.table = tbl;
          setTableData(tbl);
        } else {
          console.error('❌ INVALID TABLE STRUCTURE:', tbl);
        }
        break;

      case 'chart':
        streamedMetaRef.current.chart = data.content;
        setChartData(data.content);
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
          isUser: false
        };

        console.log('✅ COMPLETE - Final Message:', {
          hasAnswer: !!finalMsg.answer,
          hasTable: !!finalMsg.table,
          tableRowCount: finalMsg.table?.rows?.length,
          hasChart: !!finalMsg.chart
        });

        setMessages(prev => [...prev, finalMsg]);

        // Reset streaming state
        streamedAnswerRef.current = '';
        streamedMetaRef.current = { source: [], table: null, chart: null, code: null };
        setMarkdownText("");
        setTableData(null);
        setChartData(null);
        setCurrentStatus("");
        setCurrentMessage({});
        setIsStreaming(false);
        break;

      case 'error':
        setMessages(prev => [...prev, { answer: `❌ ${data.content}`, isError: true }]);
        setIsStreaming(false);
        setMarkdownText("");
        setTableData(null);
        setChartData(null);
        break;

      default:
        break;
    }
  };

  // Request a visualization from the backend for a given table and columns
  const requestPlotFromTable = async (table, xColumn, yColumn) => {
    if (!table || !xColumn || !yColumn) return;
    setIsStreaming(true);
    setCurrentMessage({ status: 'Requesting visualization...' });

    try {
      // Send the full table payload to the backend plot streaming endpoint
      await chatService.plotFromTable(table, xColumn, yColumn, sessionId, handleStreamMessage);
    } catch (e) {
      console.error('Visualization request failed', e);
      setMessages(prev => [...prev, { answer: '❌ Visualization request failed.', isError: true }]);
      setIsStreaming(false);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || isStreaming) return;
    const userQuery = input.trim();

    // 🎯 Fix 1: Enhanced plot detection - catch "plot top 10..." style queries
    const plotTriggers = ['plot', 'chart', 'graph', 'visualize', 'show chart'];
    const userQueryLower = userQuery.toLowerCase();

    if (plotTriggers.some(trigger => userQueryLower.includes(trigger))) {
      // Look in LAST 3 assistant messages for ANY cached table data
      const recentMessages = messages.slice(-6).filter(m => !m.isUser && m.table);
      const lastTable = recentMessages.length > 0 ? recentMessages[recentMessages.length - 1].table : null;

      if (lastTable && lastTable.columns && Array.isArray(lastTable.rows) && lastTable.rows.length > 0) {
        console.log("🎯 Using cached table for plot:", lastTable.rows.length, "rows");

        // Smart column detection
        const isNumeric = (v) => v === null ? false : !Number.isNaN(Number(v));
        const firstRow = lastTable.rows[0] || {};
        const xCol = lastTable.columns[0];
        let yCol = lastTable.columns.find(c => isNumeric(firstRow[c]));
        if (!yCol) yCol = lastTable.columns[1] || lastTable.columns[0];

        // Use cached table for plot - skip SQLAgent completely!
        await requestPlotFromTable(lastTable, xCol, yCol);
        setInput('');
        return;
      }
    }

    // Intercept short plot-only requests locally and use the last assistant table
    const plotOnlyTriggers = ['plot', 'plot it', 'show chart', 'visualize', 'visualize it', 'chart it', 'plot the data'];
    if (plotOnlyTriggers.includes(userQuery.toLowerCase())) {
      // find last assistant message with table, or use the in-progress streamed table as fallback
      const lastAssistantWithTable = [...messages].reverse().find(m => m && m.table && m.table.columns && Array.isArray(m.table.rows) && m.table.rows.length > 0);
      const inProgressTable = streamedMetaRef.current && streamedMetaRef.current.table;
      const table = lastAssistantWithTable ? lastAssistantWithTable.table : (inProgressTable ? inProgressTable : null);
      if (table) {
        // pick sensible defaults: x=first column, y=first numeric column
        const isNumeric = (v) => v === null ? false : !Number.isNaN(Number(v));
        const firstRow = table.rows && table.rows[0] ? table.rows[0] : {};
        const xCol = table.columns[0];
        let yCol = table.columns.find(c => isNumeric(firstRow[c]));
        if (!yCol) yCol = table.columns[1] || table.columns[0];
        // call plot request and return without sending to LLM
        await requestPlotFromTable(table, xCol, yCol);
        setInput('');
        return;
      } else {
        setMessages(prev => [...prev, { answer: `❌ I don't have any data in memory to plot yet. Please ask a data-related question first (e.g., 'What is the total coal supply?').`, isError: true }]);
        setInput('');
        return;
      }
    }

    const userMessage = { content: input, isUser: true };
    setMessages(prev => [...prev, userMessage]);
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
          <button className="debug-toggle" onClick={() => setShowRawDebug(s => !s)} style={{ marginLeft: 8 }}>
            {showRawDebug ? 'Hide Raw' : 'Show Raw'}
          </button>
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
          <MessageBubble
            key={idx}
            message={msg}
            isUser={msg.isUser}
            onRequestPlot={requestPlotFromTable}
            sessionId={sessionId}
            conversation={messages}
          />
        ))}

        {showRawDebug && <StreamDebugger events={rawEvents} />}

        {isStreaming && (
          <div className="streaming-message-wrapper">
            {currentStatus && <div className="streaming-status">{currentStatus}</div>}

            {markdownText && (
              <MessageBubble
                message={{ answer: markdownText }}
                isUser={false}
              />
            )}

            {tableData && (
              <div style={{ padding: '0 24px', margin: '16px 0' }}>
                <TableDisplay data={tableData} onRequestPlot={requestPlotFromTable} />
              </div>
            )}

            {chartData && (
              <div style={{ padding: '0 24px', margin: '16px 0' }}>
                <ChartDisplay chart={chartData} />
              </div>
            )}
          </div>
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