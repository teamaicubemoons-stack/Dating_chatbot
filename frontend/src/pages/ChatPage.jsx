/**
 * pages/ChatPage.jsx
 * -------------------
 * The individual chat page for a specific AI persona.
 * Fetches the full profile detail from the backend, loads chat history,
 * and renders the ChatWindow component.
 *
 * URL: /chat/:profileId
 */

import React, { useEffect, useState } from 'react';
import { useParams, useNavigate, Link, useOutletContext } from 'react-router-dom';
import ChatWindow from '../components/ChatWindow';
import DeleteConfirmModal from '../components/DeleteConfirmModal';
import { useChatContext } from '../context/ChatContext';
import { getProfile } from '../services/api';

export default function ChatPage() {
  const { profileId } = useParams();
  const navigate = useNavigate();
  const outletContext = useOutletContext() || {};
  const { onToggleMobileNav } = outletContext;
  const { loadHistory, setActiveProfileId } = useChatContext();

  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [topbarImgError, setTopbarImgError] = useState(false);

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    setError(null);
    setActiveProfileId(profileId);

    Promise.all([
      getProfile(profileId),
      loadHistory(profileId),
    ])
      .then(([profileData]) => {
        if (mounted) setProfile(profileData);
      })
      .catch((err) => {
        if (mounted) setError(err.message);
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });

    return () => { mounted = false; };
  }, [profileId, loadHistory, setActiveProfileId]);

  if (loading) {
    return (
      <div className="chat-page chat-page--loading" aria-busy="true">
        <div className="spinner" role="status" aria-label="Loading chat" />
      </div>
    );
  }

  if (error || !profile) {
    return (
      <div className="chat-page chat-page--error">
        <p>⚠ Couldn't load this profile.</p>
        <p className="chat-page__error-detail">{error}</p>
        <button className="btn btn--ghost" onClick={() => navigate('/')}>
          ← Back to profiles
        </button>
      </div>
    );
  }

  const avatarChar = profile.name?.[0]?.toUpperCase() || '✦';

  return (
    <div className="chat-page">
      {/* ---- Top bar ---- */}
      <header className="chat-topbar" role="banner">
        <button
          className="chat-topbar__back"
          onClick={() => navigate('/')}
          aria-label="Back to matches"
          id="back-to-profiles"
          title="Back to all matches"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="15 18 9 12 15 6" />
          </svg>
          <span className="chat-topbar__back-text">Matches</span>
        </button>

        {/* Profile mini-header */}
        <div className="chat-topbar__profile">
          <div className="chat-topbar__avatar" aria-hidden="true">
            {profile.avatar_url && !topbarImgError
              ? <img src={profile.avatar_url} alt="" onError={() => setTopbarImgError(true)} />
              : avatarChar}
          </div>
          <div className="chat-topbar__meta">
            <span className="chat-topbar__name">{profile.name}</span>
            <span className="chat-topbar__status">
              <span className="chat-topbar__online-dot" aria-hidden="true" />
              Active now
            </span>
          </div>
        </div>

        <div className="chat-topbar__actions">
          <button
            type="button"
            className="chat-topbar__call-btn"
            title={`Call ${profile.name}`}
            aria-label={`Call ${profile.name}`}
            onClick={() => {
              if (outletContext?.showToast) {
                outletContext.showToast(`📞 Voice call with ${profile.name} will be integrated soon!`);
              } else {
                alert(`📞 Voice call with ${profile.name} will be integrated soon!`);
              }
            }}
          >
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z" />
            </svg>
          </button>
          <button
            type="button"
            className="chat-topbar__delete-btn"
            title={`Delete ${profile.name}'s profile`}
            aria-label={`Delete ${profile.name}'s profile`}
            onClick={() => setShowDeleteModal(true)}
          >
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="3 6 5 6 21 6" />
              <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
              <line x1="10" y1="11" x2="10" y2="17" />
              <line x1="14" y1="11" x2="14" y2="17" />
            </svg>
          </button>
          <Link
            to="/settings"
            className="chat-topbar__settings"
            aria-label="Settings"
            id="chat-settings-link"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
            </svg>
          </Link>
        </div>
      </header>

      {/* ---- Chat window ---- */}
      <main className="chat-page__main" id="chat-main">
        <ChatWindow profile={profile} />
      </main>

      {/* Delete Confirmation Modal */}
      <DeleteConfirmModal
        isOpen={showDeleteModal}
        profile={profile}
        onClose={() => setShowDeleteModal(false)}
        onDeleted={() => navigate('/')}
      />
    </div>
  );
}
