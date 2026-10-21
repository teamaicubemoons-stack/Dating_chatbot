/**
 * components/ClearChatModal.jsx
 * ------------------------------
 * Confirmation modal before clearing conversation history with a persona.
 * Explicitly reassures the user that the persona profile is safe, only the
 * messages are being cleared.
 */

import React, { useState } from 'react';

export default function ClearChatModal({ isOpen, profile, onClose, onConfirmed }) {
  const [clearing, setClearing] = useState(false);
  const [error, setError] = useState(null);

  if (!isOpen || !profile) return null;

  const handleConfirm = async () => {
    setClearing(true);
    setError(null);
    try {
      await onConfirmed(profile.id);
      onClose();
    } catch (err) {
      const message = err?.message || '';
      const isReplyTimeoutMessage = message.includes('Reply aane mein thoda time lag gaya');
      setError(
        isReplyTimeoutMessage
          ? 'Chat history clear nahi ho payi. Connection check karke dobara try karo.'
          : message || 'Chat history clear nahi ho payi. Please try again.'
      );
    } finally {
      setClearing(false);
    }
  };

  const avatarChar = profile.name?.[0]?.toUpperCase() || '✦';

  return (
    <div className="modal-overlay" onClick={onClose} role="dialog" aria-modal="true">
      <div
        className="modal-card modal-card--confirm"
        onClick={(e) => e.stopPropagation()}
        style={{ maxWidth: '440px' }}
      >
        {/* Header */}
        <div className="modal-header" style={{ borderBottom: 'none', paddingBottom: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span
              style={{
                width: '38px',
                height: '38px',
                borderRadius: '50%',
                background: '#fee2e2',
                color: '#ef4444',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '1.2rem',
                flexShrink: 0,
              }}
            >
              🗑️
            </span>
            <div>
              <h2 className="modal-title" style={{ fontSize: '1.25rem' }}>
                Clear Conversation
              </h2>
            </div>
          </div>
          <button className="modal-close-btn" onClick={onClose} aria-label="Close modal">
            ✕
          </button>
        </div>

        {/* Companion Preview */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            background: 'var(--surface, #f8fafc)',
            borderRadius: '12px',
            padding: '12px 14px',
            margin: '16px 0 14px',
            border: '1px solid var(--border, #e2e8f0)',
          }}
        >
          <div
            style={{
              width: '42px',
              height: '42px',
              borderRadius: '50%',
              background: 'linear-gradient(135deg, var(--accent, #e11d48), #f43f5e)',
              color: '#fff',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontWeight: 600,
              fontSize: '1.1rem',
              overflow: 'hidden',
              flexShrink: 0,
            }}
          >
            {profile.avatar_url ? (
              <img
                src={profile.avatar_url}
                alt={profile.name}
                style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                onError={(e) => {
                  e.currentTarget.style.display = 'none';
                }}
              />
            ) : (
              avatarChar
            )}
          </div>
          <div style={{ minWidth: 0 }}>
            <div style={{ fontWeight: 600, color: 'var(--text-primary, #0f172a)' }}>
              {profile.name}
            </div>
            <div
              style={{
                fontSize: '0.8rem',
                color: 'var(--text-secondary, #64748b)',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
            >
              {profile.short_bio || 'AI Companion'}
            </div>
          </div>
        </div>

        {/* Explanation text */}
        <p
          style={{
            fontSize: '0.9rem',
            color: 'var(--text-secondary, #475569)',
            lineHeight: 1.5,
            marginBottom: '10px',
          }}
        >
          Are you sure you want to clear your chat with <strong>{profile.name}</strong>?
        </p>

        <div
          style={{
            background: '#eff6ff',
            border: '1px solid #bfdbfe',
            borderRadius: '8px',
            padding: '10px 12px',
            fontSize: '0.82rem',
            color: '#1e40af',
            marginBottom: '16px',
            lineHeight: 1.4,
          }}
        >
          ✓ <strong>{profile.name}'s profile will NOT be deleted.</strong> Only previous chat messages in this conversation will be cleared so you can start fresh.
        </div>

        {error && (
          <div
            style={{
              background: '#fee2e2',
              color: '#b91c1c',
              padding: '8px 12px',
              borderRadius: '8px',
              fontSize: '0.85rem',
              marginBottom: '12px',
            }}
          >
            {error}
          </div>
        )}

        {/* Action buttons */}
        <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end', marginTop: '12px' }}>
          <button
            type="button"
            className="btn btn--secondary"
            onClick={onClose}
            disabled={clearing}
            style={{ padding: '8px 18px', fontSize: '0.9rem' }}
          >
            Cancel
          </button>
          <button
            type="button"
            className="btn btn--danger"
            onClick={handleConfirm}
            disabled={clearing}
            style={{
              padding: '8px 18px',
              fontSize: '0.9rem',
              background: '#ef4444',
              color: '#fff',
              border: 'none',
              borderRadius: '8px',
              cursor: clearing ? 'not-allowed' : 'pointer',
              fontWeight: 500,
            }}
          >
            {clearing ? 'Clearing...' : 'Clear Chat History'}
          </button>
        </div>
      </div>
    </div>
  );
}
