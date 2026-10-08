/**
 * components/MessageBubble.jsx
 * -----------------------------
 * Renders a single chat message bubble — either user or AI (assistant).
 * Handles timestamp formatting and animation on mount.
 */

import React from 'react';

function formatTime(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

export default function MessageBubble({ message, profileName, avatarChar }) {
  const isUser = message.role === 'user';

  return (
    <div className={`message-bubble ${isUser ? 'message-bubble--user' : 'message-bubble--ai'}`}>
      {/* AI avatar dot */}
      {!isUser && (
        <div className="message-bubble__avatar" aria-hidden="true">
          {avatarChar || '✦'}
        </div>
      )}

      <div className="message-bubble__body">
        {!isUser && (
          <span className="message-bubble__name">{profileName}</span>
        )}
        <div className="message-bubble__text">
          {message.content}
        </div>
        <span className="message-bubble__time" aria-label={`Sent at ${formatTime(message.timestamp)}`}>
          {formatTime(message.timestamp)}
        </span>
      </div>
    </div>
  );
}
