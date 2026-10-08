/**
 * components/ProfileList.jsx
 * ---------------------------
 * Renders the full grid of ProfileCard components.
 * Handles loading and error states from the profiles API.
 *
 * Props:
 *   profiles  — array of profile objects
 *   loading   — boolean
 *   error     — string | null
 */

import React from 'react';
import ProfileCard from './ProfileCard';

export default function ProfileList({ profiles, loading, error, onDeleteProfile, onCall }) {
  if (loading) {
    return (
      <div className="profile-list__loading" aria-live="polite" aria-busy="true">
        {[...Array(3)].map((_, i) => (
          <div key={i} className="profile-card profile-card--skeleton" aria-hidden="true">
            <div className="skeleton skeleton--avatar" />
            <div className="skeleton skeleton--line" style={{ width: '60%' }} />
            <div className="skeleton skeleton--line" style={{ width: '80%' }} />
            <div className="skeleton skeleton--line" style={{ width: '40%' }} />
          </div>
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className="profile-list__error" role="alert">
        <p>⚠ Couldn't load profiles — is the backend running?</p>
        <p className="profile-list__error-detail">{error}</p>
      </div>
    );
  }

  if (!profiles.length) {
    return (
      <div className="profile-list__empty">
        <p>No profiles available. Add persona JSON configs to <code>backend/app/personas/configs/</code></p>
      </div>
    );
  }

  return (
    <div className="profile-list" role="list" aria-label="Available AI companions">
      {profiles.map((profile) => (
        <div key={profile.id} role="listitem">
          <ProfileCard
            profile={profile}
            onDelete={onDeleteProfile}
            onCall={onCall}
          />
        </div>
      ))}
    </div>
  );
}
