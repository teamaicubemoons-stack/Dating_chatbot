/**
 * components/ProfileCard.jsx
 * ---------------------------
 * A single persona card shown in the homepage grid.
 * Clicking navigates to that persona's chat page.
 *
 * Props:
 *   profile — { id, name, short_bio, avatar_url, personality_traits }
 */

import React from 'react';
import { useNavigate } from 'react-router-dom';

export default function ProfileCard({ profile, onDelete, onCall }) {
  const navigate = useNavigate();
  const [imgError, setImgError] = React.useState(false);
  const avatarChar = profile.name?.[0]?.toUpperCase() || '?';

  const handleClick = () => navigate(`/chat/${profile.id}`);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleClick();
    }
  };

  const handleCall = (e) => {
    e.stopPropagation();
    if (onCall) {
      onCall(profile);
    } else {
      alert(`📞 Voice call with ${profile.name} will be integrated soon!`);
    }
  };

  const hasValidAvatar = Boolean(profile.avatar_url) && !imgError;

  return (
    <article
      className="profile-card"
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      tabIndex={0}
      role="button"
      aria-label={`Chat with ${profile.name}`}
      id={`profile-card-${profile.id}`}
    >
      {/* Glow hover effect */}
      <div className="profile-card__glow" aria-hidden="true" />

      {/* Delete button */}
      {onDelete && (
        <button
          type="button"
          className="profile-card__delete-btn"
          title={`Delete ${profile.name}'s profile`}
          aria-label={`Delete ${profile.name}'s profile`}
          onClick={(e) => {
            e.stopPropagation();
            onDelete(profile);
          }}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="3 6 5 6 21 6" />
            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
            <line x1="10" y1="11" x2="10" y2="17" />
            <line x1="14" y1="11" x2="14" y2="17" />
          </svg>
        </button>
      )}

      {/* Avatar */}
      <div className="profile-card__avatar-wrapper">
        {hasValidAvatar ? (
          <img
            className="profile-card__avatar-img"
            src={profile.avatar_url}
            alt={`${profile.name}'s avatar`}
            onError={() => setImgError(true)}
          />
        ) : (
          <div className="profile-card__avatar-placeholder" aria-hidden="true">
            {avatarChar}
          </div>
        )}
        <span className="profile-card__online-dot" aria-label="Online" />
      </div>

      {/* Info */}
      <div className="profile-card__info">
        <h3 className="profile-card__name">
          {profile.name}{profile.age ? `, ${profile.age}` : ''}
        </h3>
        {profile.country && (
          <span
            className="profile-card__location"
            style={{
              fontSize: '0.76rem',
              color: '#64748b',
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
              fontWeight: '500',
              marginTop: '-4px',
              marginBottom: '2px',
            }}
          >
            📍 {profile.city ? `${profile.city}, ` : ''}{profile.country}
          </span>
        )}
        <p className="profile-card__bio">{profile.short_bio}</p>
      </div>

      {/* CTAs: Say hi & Call */}
      <div className="profile-card__actions" onClick={(e) => e.stopPropagation()}>
        <button
          type="button"
          className="profile-card__cta"
          onClick={handleClick}
          title={`Chat with ${profile.name}`}
        >
          Say hi →
        </button>
        <button
          type="button"
          className="profile-card__call-btn"
          onClick={handleCall}
          title={`Call ${profile.name}`}
          aria-label={`Call ${profile.name}`}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2.3" strokeLinecap="round" strokeLinejoin="round">
            <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z" />
          </svg>
          <span>Call</span>
        </button>
      </div>
    </article>
  );
}
