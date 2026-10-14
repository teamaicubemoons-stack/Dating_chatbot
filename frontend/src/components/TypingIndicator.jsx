/**
 * components/TypingIndicator.jsx
 * --------------------------------
 * Animated "…" typing indicator shown while the AI is generating a reply.
 * Three bouncing dots give a natural, human-paced feel.
 */

import React from 'react';

export default function TypingIndicator({ profileName, avatarUrl, avatarChar }) {
  const [avatarErr, setAvatarErr] = React.useState(false);

  return (
    <div className="typing-indicator" role="status" aria-label={`${profileName || 'Companion'} is typing`}>
      <div className="typing-indicator__avatar" aria-hidden="true">
        {avatarUrl && !avatarErr ? (
          <img
            src={avatarUrl}
            alt=""
            onError={() => setAvatarErr(true)}
            style={{ width: '100%', height: '100%', borderRadius: '50%', objectFit: 'cover' }}
          />
        ) : (
          avatarChar || profileName?.[0]?.toUpperCase() || '✦'
        )}
      </div>
      <div className="typing-indicator__bubble">
        <span className="typing-indicator__dot" style={{ animationDelay: '0ms' }} />
        <span className="typing-indicator__dot" style={{ animationDelay: '160ms' }} />
        <span className="typing-indicator__dot" style={{ animationDelay: '320ms' }} />
      </div>
      <span
        className="typing-indicator__label"
        style={{
          fontSize: '0.8rem',
          color: '#94a3b8',
          fontStyle: 'italic',
          marginLeft: '6px',
          alignSelf: 'center',
        }}
      >
        {profileName || 'Diya'} typing...
      </span>
    </div>
  );
}
