import React from 'react';
import { User, Bot } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import TableDisplay from './TableDisplay';
import ChartDisplay from './ChartDisplay';

import SourceBadge from './SourceBadge';

const MessageBubble = ({ message, isUser }) => {
    return (
        <div className={`message-container ${isUser ? 'user' : 'bot'}`}>
            <div className={`message-avatar ${isUser ? 'user' : 'bot'}`}>
                {isUser ? <User size={18} /> : <Bot size={18} />}
            </div>

            <div className={`message-bubble ${isUser ? 'user-bubble' : 'bot-bubble'} ${!isUser && !message.answer && !message.content ? 'thinking' : ''}`}>
                <div className="message-content">
                    {/* Answer text with Markdown support */}
                    {(message.answer || message.content) ? (
                        <div className={`markdown-content ${isUser ? 'user-text' : 'bot-text'}`}>
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                {isUser ? message.content : message.answer}
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

                    {/* Table data */}
                    {!isUser && message.table && (
                        <TableDisplay data={message.table} />
                    )}

                    {!isUser && message.chart && (
                        <ChartDisplay chart={message.chart} />
                    )}


                </div>
            </div>
        </div>
    );
};

export default MessageBubble;