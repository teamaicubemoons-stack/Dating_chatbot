/**
 * components/AppSidebar.jsx
 * --------------------------
 * Left-side navigation & conversation drawer for the Tinder/Bumble-style
 * Master-Detail dating web app layout.
 *
 * Features:
 *  - Matches carousel (horizontal avatar stories with online indicators)
 *  - Search bar to filter companions by name, location, or bio
 *  - Real-time conversation threads with last message preview & timestamps
 *  - 1-click companion switching without page reload
 *  - Quick "+ Add Profile" action and delete options
 */

import React, { useState, useMemo } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useChatContext } from '../context/ChatContext';

function formatTimestamp(iso) {
  if (!iso) return '';
  const date = new Date(iso);
  const now = new Date();
  const diffMs = now - date;
  const diffMins = Math.floor(diffMs / 60000);
  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours}h ago`;
  return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

export default function AppSidebar({
  profiles = [],
  activeProfileId = null,
  onOpenAddModal,
  onDeleteProfile,
  isMobileOpen = false,
  onCloseMobile,
}) {
  const navigate = useNavigate();
  const { messagesByProfile } = useChatContext();
  const [searchQuery, setSearchQuery] = useState('');
  const [imgErrors, setImgErrors] = useState({});

  const handleImgError = (id) => {
    setImgErrors((prev) => ({ ...prev, [id]: true }));
  };

  // Filter profiles based on search input
  const filteredProfiles = useMemo(() => {
    const q = searchQuery.toLowerCase().trim();
    if (!q) return profiles;
    return profiles.filter(
      (p) =>
        p.name?.toLowerCase().includes(q) ||
        p.city?.toLowerCase().includes(q) ||
        p.country?.toLowerCase().includes(q) ||
        p.short_bio?.toLowerCase().includes(q)
    );
  }, [profiles, searchQuery]);

  const handleSelect = (profileId) => {
    navigate(`/chat/${profileId}`);
    if (onCloseMobile) onCloseMobile();
  };

  return (
    <>
      {isMobileOpen && (
        <div
          className="app-sidebar__backdrop"
          onClick={onCloseMobile}
          aria-hidden="true"
        />
      )}
      <aside className={`app-sidebar ${isMobileOpen ? 'app-sidebar--mobile-open' : ''}`}>
      {/* 1. Header */}
      <div className="app-sidebar__header">
        <Link to="/" className="app-sidebar__brand" onClick={onCloseMobile}>
          <span className="app-sidebar__logo gradient-text">✦ Spark</span>
          <span className="app-sidebar__badge">Dating</span>
        </Link>
        <div className="app-sidebar__header-actions">
          <button
            type="button"
            className="btn-add-profile app-sidebar__add-btn"
            onClick={onOpenAddModal}
            title="Create new AI persona"
          >
            <span>+</span>
            <span className="btn-add-text">Add</span>
          </button>
          <Link
            to="/settings"
            className="app-sidebar__icon-btn"
            title="About & Settings"
            onClick={onCloseMobile}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
            </svg>
          </Link>
        </div>
      </div>

      {/* 2. Matches Carousel (Tinder/Bumble Horizontal Stories) */}
      <div className="app-sidebar__section app-sidebar__matches-section">
        <div className="app-sidebar__section-title">
          <span>Companions ({profiles.length})</span>
          <Link to="/" className="app-sidebar__explore-link" onClick={onCloseMobile}>
            Explore All ↗
          </Link>
        </div>
        <div className="app-sidebar__matches-row">
          {profiles.map((p) => {
            const isSelected = p.id === activeProfileId;
            const hasValidAvatar = Boolean(p.avatar_url) && !imgErrors[p.id];
            const initial = p.name?.[0]?.toUpperCase() || '✦';

            return (
              <div
                key={p.id}
                className={`match-pill ${isSelected ? 'match-pill--selected' : ''}`}
                onClick={() => handleSelect(p.id)}
                role="button"
                tabIndex={0}
                title={`Chat with ${p.name}`}
              >
                <div className="match-pill__avatar-wrapper">
                  {hasValidAvatar ? (
                    <img
                      src={p.avatar_url}
                      alt={p.name}
                      onError={() => handleImgError(p.id)}
                    />
                  ) : (
                    <div className="match-pill__fallback">{initial}</div>
                  )}
                  <span className="match-pill__online-dot" />
                </div>
                <span className="match-pill__name">{p.name}</span>
              </div>
            );
          })}
        </div>
      </div>

      {/* 3. Search Bar */}
      <div className="app-sidebar__search-box">
        <div className="app-sidebar__search-inner">
          <svg className="app-sidebar__search-icon" width="15" height="15" viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <input
            type="text"
            className="app-sidebar__search-input"
            placeholder="Search matches or cities..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          {searchQuery && (
            <button
              type="button"
              className="app-sidebar__search-clear"
              onClick={() => setSearchQuery('')}
            >
              ✕
            </button>
          )}
        </div>
      </div>

      {/* 4. Active Conversations List */}
      <div className="app-sidebar__section app-sidebar__threads-section">
        <div className="app-sidebar__section-title">
          <span>Messages</span>
          <span className="app-sidebar__threads-count">
            {filteredProfiles.length} active
          </span>
        </div>

        <div className="app-sidebar__threads-list" role="list">
          {filteredProfiles.length === 0 ? (
            <div className="app-sidebar__empty">
              <p>No companions found matching "{searchQuery}"</p>
            </div>
          ) : (
            filteredProfiles.map((p) => {
              const isSelected = p.id === activeProfileId;
              const hasValidAvatar = Boolean(p.avatar_url) && !imgErrors[p.id];
              const initial = p.name?.[0]?.toUpperCase() || '✦';
              const msgs = messagesByProfile?.[p.id] || [];
              const lastMsg = msgs.length > 0 ? msgs[msgs.length - 1] : null;

              return (
                <div
                  key={p.id}
                  className={`thread-item ${isSelected ? 'thread-item--active' : ''}`}
                  onClick={() => handleSelect(p.id)}
                  role="button"
                  tabIndex={0}
                >
                  {/* Left: Avatar */}
                  <div className="thread-item__avatar-box">
                    {hasValidAvatar ? (
                      <img
                        src={p.avatar_url}
                        alt={p.name}
                        onError={() => handleImgError(p.id)}
                      />
                    ) : (
                      <div className="thread-item__fallback">{initial}</div>
                    )}
                    <span className="thread-item__online-dot" />
                  </div>

                  {/* Middle: Details */}
                  <div className="thread-item__info">
                    <div className="thread-item__top-row">
                      <span className="thread-item__name">
                        {p.name}{p.age ? `, ${p.age}` : ''}
                      </span>
                      {lastMsg && (
                        <span className="thread-item__time">
                          {formatTimestamp(lastMsg.timestamp)}
                        </span>
                      )}
                    </div>

                    <div className="thread-item__location">
                      📍 {p.city ? `${p.city}, ` : ''}{p.country || 'Global'}
                    </div>

                    <p className="thread-item__snippet">
                      {lastMsg ? (
                        <span>
                          {lastMsg.role === 'user' ? 'You: ' : ''}
                          {lastMsg.content}
                        </span>
                      ) : (
                        <span className="thread-item__bio-preview">{p.short_bio}</span>
                      )}
                    </p>
                  </div>

                  {/* Right: Quick Delete Button */}
                  {onDeleteProfile && (
                    <button
                      type="button"
                      className="thread-item__delete-btn"
                      title={`Delete ${p.name}`}
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteProfile(p);
                      }}
                    >
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
                        stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="3 6 5 6 21 6" />
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                      </svg>
                    </button>
                  )}
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* 5. Footer */}
      <div className="app-sidebar__footer">
        <span className="app-sidebar__footer-text">
          ✦ AI Companions with Real Memory & Heart
        </span>
      </div>
    </aside>
    </>
  );
}
