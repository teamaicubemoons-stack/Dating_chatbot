/**
 * components/TypingIndicator.jsx
 * --------------------------------
 * Animated "…" typing indicator shown while the AI is generating a reply.
 * Three bouncing dots give a natural, human-paced feel.
 */

import React from 'react';

export default function TypingIndicator({ profileName }) {
  return (
    <div className="typing-indicator" role="status" aria-label={`${profileName} is typing`}>
      <div className="typing-indicator__avatar" aria-hidden="true">✦</div>
      <div className="typing-indicator__bubble">
        <span className="typing-indicator__dot" style={{ animationDelay: '0ms' }} />
        <span className="typing-indicator__dot" style={{ animationDelay: '160ms' }} />
        <span className="typing-indicator__dot" style={{ animationDelay: '320ms' }} />
      </div>
    </div>
  );
}
