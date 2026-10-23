/**
 * components/MessageBubble.jsx
 * -----------------------------
 * Renders a single chat message bubble — either user or AI (assistant).
 * Handles timestamp formatting and animation on mount.
 */

import React from 'react';

function formatTime(iso) {
  if (!iso) return '';
  let s = String(iso);
  // If naive ISO string from backend lacks timezone offset, append 'Z' so client parses as UTC
  if (!s.endsWith('Z') && !/[+-]\d{2}:\d{2}$/.test(s)) {
    s = s + 'Z';
  }
  const d = new Date(s);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

export default function MessageBubble({ message, profileName, avatarUrl, avatarChar }) {
  const isUser = message.role === 'user';
  const [avatarErr, setAvatarErr] = React.useState(false);

  return (
    <div className={`message-bubble ${isUser ? 'message-bubble--user' : 'message-bubble--ai'}`}>
      {/* AI avatar photo */}
      {!isUser && (
        <div className="message-bubble__avatar" aria-hidden="true" style={{ overflow: 'hidden' }}>
          {avatarUrl && !avatarErr ? (
            <img
              src={avatarUrl}
              alt=""
              onError={() => setAvatarErr(true)}
              style={{ width: '100%', height: '100%', borderRadius: '50%', objectFit: 'cover' }}
            />
          ) : (
            avatarChar || '✦'
          )}
        </div>
      )}

      <div className="message-bubble__body">
        {!isUser && (
          <span className="message-bubble__name">{profileName}</span>
        )}
        <div className="message-bubble__text">
          {(message.content || '').replace(/\|\|\|/g, '').trim()}
        </div>
        <span className="message-bubble__time" aria-label={`Sent at ${formatTime(message.timestamp)}`}>
          {formatTime(message.timestamp)}
        </span>
      </div>
    </div>
  );
}
