import React from 'react';
import { User, Bot } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import TableDisplay from './TableDisplay';
import ChartDisplay from './ChartDisplay';
import SourceBadge from './SourceBadge';

const cleanResponseText = (text) => {
    if (!text) return text;
    // Strip SQL code blocks
    return text.replace(/```sql[\s\S]*?```/gi, '').trim();
};

const MessageBubble = ({ message, isUser, onRequestPlot }) => {
    // ✅ Fix 9 — HIDE TECHNICAL TOKENS IN FRONTEND
    if (message.type === "code" || message.type === "status") return null;

    // Process text extraction and cleaning
    let displayTable = message.table;
    let cleanText = message.answer || message.content;

    if (!isUser) {
        // Final polish: remove SQL
        cleanText = cleanResponseText(cleanText);
    }

    // If backend returned a plain sentence list (e.g. "The top 10 companies by revenue are: X with $Y, ..."),
    // try to parse it into a structured table so the frontend can display and plot it.
    const parseCompaniesWithRevenue = (text) => {
        if (!text) return null;
        // regex: capture "NAME with $1,234,456" or "NAME with 123456" patterns
        const itemRe = /([A-Z0-9&.,'()\- ]{3,100}?)\s+with\s+\$?([0-9,\.]+)/gi;
        const rows = [];
        let m;
        while ((m = itemRe.exec(text)) !== null) {
            const name = m[1].trim().replace(/\s{2,}/g, ' ');
            const revStr = m[2].replace(/,/g, '');
            const rev = Number(revStr) || null;
            rows.push({ company: name, revenue: rev });
        }
        if (rows.length > 0) {
            return {
                columns: ['company', 'revenue'],
                rows: rows
            };
        }
        return null;
    };

    const parseNumberedList = (text) => {
        if (!text) return null;
        const lines = text.split('\n').filter(l => /^\d+\./.test(l.trim()));
        if (lines.length < 2) return null;
        const rows = lines.map(line => {
            const match = line.match(/^(\d+)\.\s*(.+?)(?:\s+with\s+\$?([0-9,\.]+))?$/i);
            if (!match) return null;
            const rank = match[1];
            const content = match[2].trim();
            const value = match[3] ? Number(match[3].replace(/,/g, '')) : null;

            if (value !== null) {
                return { rank, item: content, value };
            }
            return { rank, item: content };
        }).filter(Boolean);

        if (rows.length > 0) {
            const cols = Object.keys(rows[0]);
            return { columns: cols, rows };
        }
        return null;
    };


    // 🔍 LOG: Track table rendering decisions
    console.log('📋 MessageBubble Render:', {
        isUser,
        hasMessage: !!message,
        hasAnswer: !!message.answer,
        hasTable: !!message.table,
        displayTableValid: !!(displayTable && displayTable.columns && Array.isArray(displayTable.rows) && displayTable.rows.length > 0),
        displayTable: displayTable
    });

    return (
        <div className={`message-container ${isUser ? 'user' : 'bot'}`}>
            <div className={`message-avatar ${isUser ? 'user' : 'bot'}`}>
                {isUser ? <User size={18} /> : <Bot size={18} />}
            </div>

            <div className={`message-bubble ${isUser ? 'user-bubble' : 'bot-bubble'} ${!isUser && !message.answer && !message.content ? 'thinking' : ''}`}>
                <div className="message-content">
                    {/* Answer text with Markdown support */}
                    {(cleanText) ? (
                        <div className={`markdown-content ${isUser ? 'user-text' : 'bot-text'}`}>
                            <ReactMarkdown
                                remarkPlugins={[remarkGfm]}
                            >
                                {cleanText}
                            </ReactMarkdown>
                        </div>
                    ) : (
                        !isUser && (
                            <div className="thinking-container">
                                <div className="status-text">{message.status || 'Thinking...'}</div>
                                <div className="typing-loader">
                                    <span></span>
                                    <span></span>
                                    <span></span>
                                </div>
                            </div>
                        )
                    )}

                    {/* Source badge(s) */}
                    {!isUser && message.source && (
                        <div className="sources-container">
                            {Array.isArray(message.source) ? (
                                message.source.map((s, idx) => (
                                    <SourceBadge key={idx} source={s} />
                                ))
                            ) : (
                                <SourceBadge source={message.source} />
                            )}
                        </div>
                    )}

                    {/* Explicit Table data from backend - ENHANCED */}
                    {!isUser && displayTable && displayTable.columns && Array.isArray(displayTable.rows) && displayTable.rows.length > 0 && (
                        <div className="data-section table-section" style={{ marginTop: '16px' }}>
                            <TableDisplay data={displayTable} onRequestPlot={onRequestPlot} />
                        </div>
                    )}

                    {!isUser && message.chart && (
                        <div className="data-section chart-section">
                            <ChartDisplay chart={message.chart} />
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default MessageBubble;