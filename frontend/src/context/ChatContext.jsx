/**
 * context/ChatContext.jsx
 * ------------------------
 * Global React context that manages:
 *  - Active profile selection
 *  - Per-profile message lists
 *  - Typing/loading state
 *  - Persistent anonymous user ID (stored in localStorage)
 *
 * Any component can call useChatContext() to access shared chat state
 * without prop-drilling.
 */

import React, { createContext, useContext, useState, useCallback, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { sendMessage, getChatHistory } from '../services/api';

const ChatContext = createContext(null);

// Persist a random user ID across page reloads
function getOrCreateUserId() {
  let id = localStorage.getItem('spark_user_id');
  if (!id) {
    id = uuidv4();
    localStorage.setItem('spark_user_id', id);
  }
  return id;
}

export function ChatProvider({ children }) {
  const userId = useRef(getOrCreateUserId()).current;

  // Map of profileId → array of message objects
  const [messagesByProfile, setMessagesByProfile] = useState({});
  const [activeProfileId, setActiveProfileId] = useState(null);
  const [isTyping, setIsTyping] = useState(false);
  const [error, setError] = useState(null);

  /**
   * Load conversation history from the backend for a given profile.
   * Called when navigating to a chat page.
   */
  const loadHistory = useCallback(async (profileId) => {
    try {
      const data = await getChatHistory(profileId, userId);
      const messages = (data.messages || []).map((m) => ({
        id: m.id,
        role: m.role,
        content: m.content,
        timestamp: m.created_at,
      }));
      setMessagesByProfile((prev) => ({ ...prev, [profileId]: messages }));
    } catch {
      // Non-fatal: history just won't be pre-populated
    }
  }, [userId]);

  /**
   * Send a message to the backend and append both the user message
   * and the AI reply to local state. Simulates a realistic typing delay.
   */
  const sendChatMessage = useCallback(async (profileId, messageText) => {
    if (!messageText.trim() || isTyping) return;

    setError(null);

    // Append user message immediately (optimistic UI)
    const userMsg = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: messageText.trim(),
      timestamp: new Date().toISOString(),
    };

    setMessagesByProfile((prev) => ({
      ...prev,
      [profileId]: [...(prev[profileId] || []), userMsg],
    }));

    // Show typing indicator
    setIsTyping(true);

    try {
      // Realistic typing delay: 1–2.5s randomized BEFORE showing the reply
      const typingDelay = 1000 + Math.random() * 1500;

      // Fire the API call in parallel with the delay
      const [response] = await Promise.all([
        sendMessage(profileId, userId, messageText.trim()),
        new Promise((resolve) => setTimeout(resolve, typingDelay)),
      ]);

      const aiMsg = {
        id: `ai-${Date.now()}`,
        role: 'assistant',
        content: response.reply,
        timestamp: new Date().toISOString(),
      };

      setMessagesByProfile((prev) => ({
        ...prev,
        [profileId]: [...(prev[profileId] || []), aiMsg],
      }));
    } catch (err) {
      setError(err.message || 'Failed to send message. Please try again.');
    } finally {
      setIsTyping(false);
    }
  }, [userId, isTyping]);

  const getMessages = useCallback(
    (profileId) => messagesByProfile[profileId] || [],
    [messagesByProfile]
  );

  const clearError = useCallback(() => setError(null), []);

  const value = {
    userId,
    activeProfileId,
    setActiveProfileId,
    isTyping,
    error,
    clearError,
    getMessages,
    sendChatMessage,
    loadHistory,
    messagesByProfile,
  };

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}

export function useChatContext() {
  const ctx = useContext(ChatContext);
  if (!ctx) throw new Error('useChatContext must be used inside <ChatProvider>');
  return ctx;
}
