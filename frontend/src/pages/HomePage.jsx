/**
 * pages/HomePage.jsx
 * -------------------
 * Discover & Companion Showcase view rendered in the main viewport.
 * Works seamlessly inside AppLayout via useOutletContext.
 */

import React from 'react';
import { Link, useOutletContext } from 'react-router-dom';
import ProfileList from '../components/ProfileList';

export default function HomePage() {
  const context = useOutletContext() || {};
  const {
    profiles = [],
    loading = false,
    error = null,
    onOpenAddModal,
    onDeleteProfile,
    onToggleMobileNav,
    showToast,
  } = context;

  const handleCall = (profile) => {
    if (showToast) {
      showToast(`📞 Voice call with ${profile.name} will be integrated soon!`);
    } else {
      alert(`📞 Voice call with ${profile.name} will be integrated soon!`);
    }
  };

  return (
    <div className="page home-page">
      {/* Background glows */}
      <div className="bg-glow bg-glow--top-left" aria-hidden="true" />
      <div className="bg-glow bg-glow--bottom-right" aria-hidden="true" />

      {/* Top navigation */}
      <nav className="topbar" role="navigation" aria-label="Main navigation">
        <div className="topbar__brand">
          {onToggleMobileNav && (
            <button
              type="button"
              className="topbar__mobile-menu-btn"
              onClick={onToggleMobileNav}
              aria-label="Open matches menu"
            >
              ☰
            </button>
          )}
          <span className="topbar__logo gradient-text">✦ Discover</span>
        </div>
        <div className="topbar__links">
          {onOpenAddModal && (
            <button
              id="btn-add-profile"
              className="btn-add-profile"
              onClick={onOpenAddModal}
            >
              <span className="btn-add-profile__icon">+</span>
              <span>Add Profile</span>
            </button>
          )}
          <Link to="/settings" className="topbar__link" id="nav-settings">
            About
          </Link>
        </div>
      </nav>

      {/* Hero section */}
      <header className="home-hero">
        <div className="home-hero__inner container">
          <div className="home-hero__badge">AI Companions</div>
          <h1 className="home-hero__title">
            Meet someone{' '}
            <span className="gradient-text">different</span>
          </h1>
          <p className="home-hero__subtitle">
            Thoughtful conversations with AI companions who have real depth —
            distinct personalities, memories of what you've shared, and a genuine
            curiosity about you.
          </p>
        </div>
      </header>

      {/* Profiles grid */}
      <main className="home-main container" id="main-content">
        <div className="home-main__header">
          <h2 className="home-main__heading">Choose your companion</h2>
          <p className="home-main__count">
            {!loading && !error && `${profiles.length} available`}
          </p>
        </div>
        <ProfileList
          profiles={profiles}
          loading={loading}
          error={error}
          onDeleteProfile={onDeleteProfile}
          onCall={handleCall}
        />
      </main>

      {/* Footer */}
      <footer className="home-footer" role="contentinfo">
        <p>
          <Link to="/settings" className="home-footer__link">About & Disclosure</Link>
          {' · '}
          <span className="home-footer__note">AI-powered companions</span>
        </p>
      </footer>
    </div>
  );
}
