import React from 'react';
import { User, Bot } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import TableDisplay from './TableDisplay';
import ChartDisplay from './ChartDisplay';
import SourceBadge from './SourceBadge';

const extractTableAndCleanText = (text) => {
    if (!text) return { table: null, cleanText: text };

    try {
        // --- 1. Python List of Tuples Patterns ---
        const listMatch = text.match(/\[\s*\(.*?\)\s*(?:,\s*\(.*?\)\s*)*\]/s);
        if (listMatch) {
            const listStr = listMatch[0];
            const tupleRegex = /\(\s*(['"])(.*?)\1\s*,\s*(.*?)\s*\)/g;
            const rows = [];
            let match;
            while ((match = tupleRegex.exec(listStr)) !== null) {
                const category = match[2];
                let value = match[3].trim();
                rows.push({ "Category": category, "Value": value });
            }
            if (rows.length >= 2) {
                return {
                    table: { columns: ["Category", "Value"], rows },
                    cleanText: text.replace(listStr, '').trim()
                };
            }
        }

        // --- 2. Line-by-Line List Parsing ---
        const lines = text.split('\n');
        const listRows = [];
        let firstListIdx = -1;
        let lastListIdx = -1;
        const itemRegex = /^(\d+[\.\)]|[\*\-])\s+/;

        for (let i = 0; i < lines.length; i++) {
            const trimmed = lines[i].trim();
            if (itemRegex.test(trimmed)) {
                if (firstListIdx === -1) firstListIdx = i;
                lastListIdx = i;
                const content = trimmed.replace(itemRegex, '').trim();
                const parts = content.split(/[:\-–—|]+/);
                listRows.push({
                    "ID": listRows.length + 1,
                    "Entity": parts[0].trim(),
                    "Info": parts.slice(1).join(' ').trim() || "-"
                });
            }
        }

        if (listRows.length >= 4) {
            const before = lines.slice(0, firstListIdx).join('\n');
            const after = lines.slice(lastListIdx + 1).join('\n');
            return {
                table: { columns: ["ID", "Entity", "Info"], rows: listRows },
                cleanText: (before + '\n' + after).trim()
            };
        }
    } catch (e) {
        console.error("Extraction failed:", e);
    }
    return { table: null, cleanText: text };
};
const cleanResponseText = (text, hasExplicitData) => {
    if (!text) return text;
    let clean = text;

    // 1. Always strip SQL code blocks (we use them for backend extraction, not frontend display)
    clean = clean.replace(/```sql[\s\S]*?```/gi, '').trim();

    // 2. If we have explicit table/chart data, strip common raw data patterns
    if (hasExplicitData) {
        // Strip Python list of tuples/dicts
        clean = clean.replace(/\[\s*\(.*?\)\s*(?:,\s*\(.*?\)\s*)*\]/gs, '');
        clean = clean.replace(/\[\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*\]/gs, '');

        // Strip numbered/bulleted lists that look like data summaries (4+ items)
        const lines = clean.split('\n');
        let firstListIdx = -1;
        let lastListIdx = -1;
        let count = 0;

        const itemRegex = /^(\d+[\.\)]|[\*\-])\s+/;
        for (let i = 0; i < lines.length; i++) {
            if (itemRegex.test(lines[i].trim())) {
                if (firstListIdx === -1) firstListIdx = i;
                lastListIdx = i;
                count++;
            } else if (count > 0 && lines[i].trim() === '') {
                // Allow empty lines between list items
                continue;
            } else if (count > 0) {
                // Non-list item found, if we had a significant list, stop here
                if (count >= 4) break;
                // Otherwise reset and keep looking
                firstListIdx = -1;
                lastListIdx = -1;
                count = 0;
            }
        }

        if (count >= 4) {
            const before = lines.slice(0, firstListIdx).join('\n');
            const after = lines.slice(lastListIdx + 1).join('\n');
            clean = (before + '\n' + after).trim();
        }
    }

    return clean;
};

const MessageBubble = ({ message, isUser }) => {
    const hasData = !!(message.table || message.chart);

    // Process text extraction and cleaning
    let displayTable = message.table;
    let cleanText = message.answer || message.content;

    if (!isUser) {
        if (!displayTable && message.answer) {
            // Try to extract a table if backend didn't provide one
            const extracted = extractTableAndCleanText(message.answer);
            displayTable = extracted.table;
            cleanText = extracted.cleanText;
        }

        // Final polish: remove SQL and redundant lists
        cleanText = cleanResponseText(cleanText, !!(displayTable || message.chart));
    }

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

                    {/* Explicit Table data from backend */}
                    {!isUser && displayTable && displayTable.rows?.length > 0 && (
                        <div className="data-section table-section">
                            <TableDisplay data={displayTable} />
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