/**
 * pages/SettingsPage.jsx
 * -----------------------
 * About, Settings, and User Memory Management page.
 *
 * Provides:
 *  - Clear AI Companion disclosure.
 *  - Interactive Memory Management: inspect what the AI remembers about you,
 *    edit/delete individual memories, forget by scope, and export/delete data.
 *  - Privacy & GDPR compliance controls.
 */

import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { getMemories, deleteMemory, forgetMemories, exportUserData, deleteAccountData } from '../services/api';

export default function SettingsPage() {
  const navigate = useNavigate();
  const userId = localStorage.getItem('spark_user_id') || 'anonymous_user';

  const [memories, setMemories] = useState([]);
  const [loadingMemories, setLoadingMemories] = useState(false);
  const [actionMessage, setActionMessage] = useState('');

  const fetchMemories = useCallback(async () => {
    setLoadingMemories(true);
    try {
      const data = await getMemories(userId);
      setMemories(data || []);
    } catch {
      setMemories([]);
    } finally {
      setLoadingMemories(false);
    }
  }, [userId]);

  useEffect(() => {
    fetchMemories();
  }, [fetchMemories]);

  const handleDeleteMemory = async (memId) => {
    try {
      await deleteMemory(memId, userId);
      setActionMessage('Memory deleted.');
      setMemories((prev) => prev.filter((m) => m.id !== memId));
      setTimeout(() => setActionMessage(''), 3000);
    } catch (err) {
      alert('Failed to delete memory: ' + err.message);
    }
  };

  const handleForgetAll = async () => {
    if (!window.confirm('Are you sure you want to clear all memories? The companions will forget saved facts about you.')) return;
    try {
      await forgetMemories(userId, 'all');
      setActionMessage('All memories cleared.');
      fetchMemories();
      setTimeout(() => setActionMessage(''), 3000);
    } catch (err) {
      alert('Failed to clear memories: ' + err.message);
    }
  };

  const handleForgetRelationship = async () => {
    if (!window.confirm('Forget persona-specific inside jokes and shared nicknames?')) return;
    try {
      await forgetMemories(userId, 'relationship');
      setActionMessage('Persona-specific memories cleared.');
      fetchMemories();
      setTimeout(() => setActionMessage(''), 3000);
    } catch (err) {
      alert('Failed to clear relationship memories: ' + err.message);
    }
  };

  const handleExportData = async () => {
    try {
      const data = await exportUserData(userId);
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `spark_data_export_${userId.slice(0, 8)}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      alert('Failed to export data: ' + err.message);
    }
  };

  const handleDeleteAllData = async () => {
    if (!window.confirm('⚠️ WARNING: This will permanently delete all your chat messages, memories, and session data. Continue?')) return;
    try {
      await deleteAccountData(userId);
      localStorage.removeItem('spark_user_id');
      alert('All your data has been permanently erased.');
      navigate('/');
    } catch (err) {
      alert('Failed to erase account data: ' + err.message);
    }
  };

  return (
    <div className="page settings-page">
      <div className="bg-glow bg-glow--top-left" aria-hidden="true" />

      {/* Top bar */}
      <header className="topbar" role="banner">
        <div className="topbar__brand">
          <span className="topbar__logo gradient-text">✦ Spark</span>
        </div>
        <button
          className="topbar__back-btn"
          onClick={() => navigate(-1)}
          aria-label="Go back"
          id="settings-back-btn"
        >
          ← Back
        </button>
      </header>

      <main className="settings-main container" id="settings-main">
        <div className="settings-content">

          {/* Page title */}
          <h1 className="settings-title">Settings & <span className="gradient-text">Memory</span></h1>
          <p className="settings-subtitle">
            Manage your AI companion experience, privacy boundaries, and what the AI remembers.
          </p>

          {/* AI COMPANION DISCLOSURE */}
          <section
            className="settings-card settings-card--disclosure"
            aria-labelledby="ai-disclosure-heading"
            id="ai-disclosure"
          >
            <div className="settings-card__icon" aria-hidden="true">🤖</div>
            <div className="settings-card__body">
              <h2 id="ai-disclosure-heading" className="settings-card__title">
                AI Companion Disclosure
              </h2>
              <p className="settings-card__text">
                <strong>Spark is an AI-powered companion platform.</strong> All companions
                (Luna, Ethan, Zara, Diya) are fictional AI personas driven by large language models.
                They are not real people and do not have physical existence or real feelings.
              </p>
              <p className="settings-card__text">
                Spark is created for thoughtful conversation, companionship, and storytelling.
              </p>
            </div>
          </section>

          {/* MEMORY MANAGEMENT SECTION */}
          <section className="settings-card" aria-labelledby="memory-management-heading">
            <div className="settings-card__icon" aria-hidden="true">🧠</div>
            <div className="settings-card__body" style={{ width: '100%' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
                <h2 id="memory-management-heading" className="settings-card__title" style={{ margin: 0 }}>
                  What the AI Remembers About You
                </h2>
                <button
                  className="btn btn--secondary btn--sm"
                  onClick={fetchMemories}
                  disabled={loadingMemories}
                >
                  {loadingMemories ? 'Refreshing...' : '🔄 Refresh'}
                </button>
              </div>

              <p className="settings-card__text" style={{ marginTop: '8px' }}>
                As you chat, companions remember important facts, preferences, and events you share.
                You have full control to view or delete any memory below:
              </p>

              {actionMessage && (
                <div style={{ background: 'rgba(34, 197, 94, 0.15)', color: '#4ade80', padding: '8px 12px', borderRadius: '8px', fontSize: '0.9rem', marginBottom: '12px' }}>
                  ✓ {actionMessage}
                </div>
              )}

              {loadingMemories ? (
                <p style={{ color: 'var(--color-text-muted)' }}>Loading memories...</p>
              ) : memories.length === 0 ? (
                <div style={{ background: 'var(--color-surface-card)', padding: '16px', borderRadius: '12px', border: '1px dashed var(--color-border)', textAlign: 'center', color: 'var(--color-text-muted)' }}>
                  No memories saved yet. Start chatting and mention things you like, your job, or upcoming events!
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '12px', maxHeight: '350px', overflowY: 'auto' }}>
                  {memories.map((mem) => (
                    <div
                      key={mem.id}
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        background: 'rgba(255, 255, 255, 0.03)',
                        border: '1px solid var(--color-border)',
                        borderRadius: '10px',
                        padding: '10px 14px',
                        gap: '12px',
                      }}
                    >
                      <div style={{ flex: 1 }}>
                        <span style={{ fontSize: '0.75rem', textTransform: 'uppercase', background: 'rgba(236, 72, 153, 0.15)', color: '#f472b6', padding: '2px 8px', borderRadius: '6px', marginRight: '8px' }}>
                          {mem.memory_type.replace('_', ' ')}
                        </span>
                        <span style={{ fontSize: '0.92rem', color: 'var(--color-text-primary)' }}>
                          {mem.memory_text}
                        </span>
                      </div>
                      <button
                        className="btn btn--danger btn--sm"
                        style={{ padding: '4px 8px', fontSize: '0.8rem' }}
                        onClick={() => handleDeleteMemory(mem.id)}
                        title="Delete this memory"
                      >
                        Delete
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {/* Memory Action Buttons */}
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginTop: '16px' }}>
                <button
                  className="btn btn--secondary btn--sm"
                  onClick={handleForgetRelationship}
                  disabled={memories.length === 0}
                >
                  Clear Inside Jokes Only
                </button>
                <button
                  className="btn btn--danger btn--sm"
                  onClick={handleForgetAll}
                  disabled={memories.length === 0}
                >
                  Clear All Memories
                </button>
              </div>
            </div>
          </section>

          {/* PRIVACY & DATA EXPORT */}
          <section className="settings-card" aria-labelledby="privacy-heading">
            <div className="settings-card__icon" aria-hidden="true">🔒</div>
            <div className="settings-card__body">
              <h2 id="privacy-heading" className="settings-card__title">
                Your Privacy & Data Rights
              </h2>
              <p className="settings-card__text">
                Your chats are associated with an anonymous session ID stored locally in your browser.
                You can download your entire data footprint or permanently erase all messages and memories anytime.
              </p>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginTop: '12px' }}>
                <button className="btn btn--secondary btn--sm" onClick={handleExportData}>
                  📥 Export My Data (JSON)
                </button>
                <button className="btn btn--danger btn--sm" onClick={handleDeleteAllData}>
                  🗑️ Permanently Erase All Data
                </button>
              </div>
            </div>
          </section>

          {/* Version info */}
          <div className="settings-footer">
            <p>Spark AI Companion Platform · v1.1.0</p>
            <p>FastAPI · React · Groq LLaMA 3.3 70B · Claude 3.5 Haiku · Low-Memory RAG</p>
          </div>

        </div>
      </main>
    </div>
  );
}
