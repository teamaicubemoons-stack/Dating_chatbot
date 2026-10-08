/**
 * App.jsx
 * --------
 * Root application component. Sets up React Router and wraps the entire
 * app in the ChatProvider context so any page can access shared chat state.
 */

import React from 'react';
import './App.css';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { ChatProvider } from './context/ChatContext';
import HomePage from './pages/HomePage';
import ChatPage from './pages/ChatPage';
import SettingsPage from './pages/SettingsPage';
import AppLayout from './components/AppLayout';

export default function App() {
  return (
    <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <ChatProvider>
        <Routes>
          {/* Main App Layout (Persistent Left Sidebar + Dynamic Right Viewport) */}
          <Route element={<AppLayout />}>
            <Route path="/" element={<HomePage />} />
            <Route path="/chat/:profileId" element={<ChatPage />} />
          </Route>
          <Route path="/settings" element={<SettingsPage />} />
          {/* Catch-all → home */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </ChatProvider>
    </BrowserRouter>
  );
}
