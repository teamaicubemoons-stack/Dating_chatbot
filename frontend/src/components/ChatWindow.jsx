/**
 * components/ChatWindow.jsx
 * --------------------------
 * The full chat interface panel — message list, typing indicator, and input bar.
 * Manages auto-scroll to the latest message and input submission.
 *
 * Props:
 *   profile  — full persona profile object { id, name, short_bio, personality_traits, ... }
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import MessageBubble from './MessageBubble';
import TypingIndicator from './TypingIndicator';
import { useChatContext } from '../context/ChatContext';

export default function ChatWindow({ profile }) {
  const { getMessages, sendChatMessage, isTyping, error, clearError } = useChatContext();
  const messages = getMessages(profile.id);

  const [input, setInput] = useState('');
  const [emptyAvatarError, setEmptyAvatarError] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Auto-scroll to bottom whenever messages or typing state change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, [profile.id]);

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || isTyping) return;
    setInput('');
    clearError();
    await sendChatMessage(profile.id, text);
  }, [input, isTyping, sendChatMessage, profile.id, clearError]);

  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }, [handleSend]);

  const avatarChar = profile.name?.[0]?.toUpperCase() || '✦';

  return (
    <div className="chat-window">
      {/* ---- Messages area ---- */}
      <div className="chat-window__messages" role="log" aria-live="polite" aria-label="Chat messages">
        {messages.length === 0 && (
          <div className="chat-window__empty">
            <div className="chat-window__empty-avatar">
              {profile.avatar_url && !emptyAvatarError ? (
                <img
                  src={profile.avatar_url}
                  alt=""
                  onError={() => setEmptyAvatarError(true)}
                  style={{ width: '100%', height: '100%', borderRadius: '50%', objectFit: 'cover' }}
                />
              ) : (
                avatarChar
              )}
            </div>
            <p className="chat-window__empty-title">Chat with {profile.name}</p>
            <p className="chat-window__empty-subtitle">{profile.short_bio}</p>
            <div
              className="chat-window__starters"
              style={{
                display: 'flex',
                flexWrap: 'wrap',
                gap: '8px',
                justifyContent: 'center',
                marginTop: '14px',
                maxWidth: '380px',
              }}
            >
              {['Hey 🙂', "How's your day going?", 'What are you up to?'].map((starter) => (
                <button
                  key={starter}
                  type="button"
                  onClick={() => sendChatMessage(profile.id, starter)}
                  style={{
                    background: '#f1f5f9',
                    border: '1px solid #cbd5e1',
                    borderRadius: '999px',
                    padding: '7px 15px',
                    fontSize: '0.85rem',
                    fontWeight: '500',
                    color: '#334155',
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                  }}
                  onMouseOver={(e) => {
                    e.currentTarget.style.background = '#e2e8f0';
                    e.currentTarget.style.borderColor = '#94a3b8';
                  }}
                  onMouseOut={(e) => {
                    e.currentTarget.style.background = '#f1f5f9';
                    e.currentTarget.style.borderColor = '#cbd5e1';
                  }}
                >
                  {starter}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <MessageBubble
            key={msg.id}
            message={msg}
            profileName={profile.name}
            avatarChar={avatarChar}
          />
        ))}

        {isTyping && <TypingIndicator profileName={profile.name} />}

        {error && (
          <div className="chat-window__error" role="alert">
            <span>⚠ {error}</span>
            <button onClick={clearError} className="chat-window__error-dismiss" aria-label="Dismiss error">×</button>
          </div>
        )}

        <div ref={messagesEndRef} aria-hidden="true" />
      </div>

      {/* ---- Input bar ---- */}
      <div className="chat-window__input-bar">
        <textarea
          ref={inputRef}
          id={`chat-input-${profile.id}`}
          className="chat-window__input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={`Message ${profile.name}…`}
          rows={1}
          maxLength={4000}
          disabled={isTyping}
          aria-label={`Type a message to ${profile.name}`}
        />
        <button
          id={`send-btn-${profile.id}`}
          className="chat-window__send-btn"
          onClick={handleSend}
          disabled={!input.trim() || isTyping}
          aria-label="Send message"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <line x1="22" y1="2" x2="11" y2="13" />
            <polygon points="22 2 15 22 11 13 2 9 22 2" />
          </svg>
        </button>
      </div>
    </div>
  );
}
