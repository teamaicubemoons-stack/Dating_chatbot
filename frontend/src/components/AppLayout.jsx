/**
 * components/AppLayout.jsx
 * -------------------------
 * Master-Detail application layout shell for Spark Dating Web App.
 * Houses the persistent AppSidebar on the left and dynamic content
 * (Discover Grid or Active Chat) on the right.
 */

import React, { useEffect, useState } from 'react';
import { Outlet, useParams, useLocation } from 'react-router-dom';
import AppSidebar from './AppSidebar';
import AddProfileModal from './AddProfileModal';
import DeleteConfirmModal from './DeleteConfirmModal';
import { getProfiles } from '../services/api';

export default function AppLayout() {
  const [profiles, setProfiles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [profileToDelete, setProfileToDelete] = useState(null);
  const [toastMessage, setToastMessage] = useState(null);
  const [isMobileNavOpen, setIsMobileNavOpen] = useState(false);

  const location = useLocation();
  const params = useParams();
  const activeProfileId = params.profileId || null;

  useEffect(() => {
    let mounted = true;
    getProfiles()
      .then((data) => {
        if (mounted) setProfiles(data);
      })
      .catch((err) => {
        if (mounted) setError(err.message);
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });
    return () => {
      mounted = false;
    };
  }, []);

  const handleProfileCreated = (newProfile) => {
    setProfiles((prev) => [newProfile, ...prev]);
    setToastMessage(`Created companion: ${newProfile.name} ✦`);
    setTimeout(() => setToastMessage(null), 4000);
  };

  const handleProfileDeleted = (deletedId) => {
    const deleted = profiles.find((p) => p.id === deletedId);
    setProfiles((prev) => prev.filter((p) => p.id !== deletedId));
    if (deleted) {
      setToastMessage(`${deleted.name}'s profile was removed.`);
      setTimeout(() => setToastMessage(null), 4000);
    }
  };

  // Close mobile nav on route change
  useEffect(() => {
    setIsMobileNavOpen(false);
  }, [location.pathname]);

  const isChatRoute = Boolean(activeProfileId);

  return (
    <div className={`app-shell ${isChatRoute ? 'app-shell--chat-active' : 'app-shell--home-active'}`}>
      {/* 1. Persistent Left Sidebar */}
      <AppSidebar
        profiles={profiles}
        activeProfileId={activeProfileId}
        onOpenAddModal={() => setIsModalOpen(true)}
        onDeleteProfile={setProfileToDelete}
        isMobileOpen={isMobileNavOpen}
        onCloseMobile={() => setIsMobileNavOpen(false)}
      />

      {/* 2. Right Main Viewport */}
      <div className="app-main-viewport">
        <Outlet
          context={{
            profiles,
            loading,
            error,
            onOpenAddModal: () => setIsModalOpen(true),
            onDeleteProfile: setProfileToDelete,
            onToggleMobileNav: () => setIsMobileNavOpen((prev) => !prev),
            showToast: (msg) => {
              setToastMessage(msg);
              setTimeout(() => setToastMessage(null), 3500);
            },
          }}
        />
      </div>

      {/* 3. Toast Notifications */}
      {toastMessage && (
        <div className="home-toast" role="status">
          <span>✓ {toastMessage}</span>
        </div>
      )}

      {/* 4. Modals */}
      <AddProfileModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onProfileCreated={handleProfileCreated}
      />

      <DeleteConfirmModal
        isOpen={Boolean(profileToDelete)}
        profile={profileToDelete}
        onClose={() => setProfileToDelete(null)}
        onDeleted={handleProfileDeleted}
      />
    </div>
  );
}
