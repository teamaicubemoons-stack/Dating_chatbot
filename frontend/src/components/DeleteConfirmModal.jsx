/**
 * components/DeleteConfirmModal.jsx
 * ----------------------------------
 * Confirmation modal before permanently deleting a persona profile.
 * Prevents accidental deletions and cleanly informs the user that
 * associated messages and memory will be erased.
 */

import React, { useState } from 'react';
import { deleteProfile } from '../services/api';

export default function DeleteConfirmModal({ isOpen, profile, onClose, onDeleted }) {
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState(null);

  if (!isOpen || !profile) return null;

  const handleDelete = async () => {
    setDeleting(true);
    setError(null);
    try {
      await deleteProfile(profile.id);
      onDeleted(profile.id);
      onClose();
    } catch (err) {
      setError(err.message || 'Failed to delete profile. Please try again.');
      setDeleting(false);
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
              🗑
            </span>
            <div>
              <h2 className="modal-title" style={{ fontSize: '1.25rem' }}>
                Delete Profile
              </h2>
            </div>
          </div>
          <button className="modal-close-btn" onClick={onClose} aria-label="Close modal">
            ✕
          </button>
        </div>

        {/* Companion Brief */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            background: '#f8fafc',
            border: '1px solid #e2e8f0',
            padding: '12px 14px',
            borderRadius: '12px',
            marginTop: '16px',
            marginBottom: '16px',
          }}
        >
          <div
            style={{
              width: '46px',
              height: '46px',
              borderRadius: '50%',
              overflow: 'hidden',
              flexShrink: 0,
              background: 'linear-gradient(135deg, #f0e5fc 0%, #e0c8f8 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontWeight: 700,
              color: '#9b35d4',
            }}
          >
            {profile.avatar_url ? (
              <img
                src={profile.avatar_url}
                alt={profile.name}
                style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                onError={(e) => { e.currentTarget.style.display = 'none'; }}
              />
            ) : (
              avatarChar
            )}
          </div>
          <div>
            <div style={{ fontWeight: 700, color: '#1e293b', fontSize: '0.95rem' }}>
              {profile.name}{profile.age ? `, ${profile.age}` : ''}
            </div>
            {profile.country && (
              <div style={{ fontSize: '0.76rem', color: '#64748b' }}>
                📍 {profile.city ? `${profile.city}, ` : ''}{profile.country}
              </div>
            )}
          </div>
        </div>

        <p style={{ color: '#475569', fontSize: '0.9rem', lineHeight: 1.5, margin: '0 0 16px' }}>
          Are you sure you want to delete <strong>{profile.name}</strong>? This will permanently remove this companion persona, their memory, and all chat history.
        </p>

        {error && (
          <div className="modal-error-banner" style={{ marginBottom: '16px' }} role="alert">
            <span>⚠ {error}</span>
          </div>
        )}

        {/* Actions */}
        <div className="modal-actions" style={{ borderTop: '1px solid #e2e8f0', paddingTop: '16px', marginTop: 0 }}>
          <button
            type="button"
            className="btn btn--secondary"
            onClick={onClose}
            disabled={deleting}
          >
            Cancel
          </button>
          <button
            type="button"
            className="btn"
            style={{
              background: '#ef4444',
              color: '#ffffff',
              border: 'none',
              padding: '10px 18px',
              borderRadius: '999px',
              fontWeight: 600,
              cursor: deleting ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
            }}
            onClick={handleDelete}
            disabled={deleting}
          >
            {deleting ? 'Deleting Companion…' : 'Yes, Delete Profile'}
          </button>
        </div>
      </div>
    </div>
  );
}
