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
import { sendMessage, getChatHistory, clearChatConversation } from '../services/api';

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
  const [typingProfileId, setTypingProfileId] = useState(null);
  const [error, setError] = useState(null);
  const [lastFailedMessage, setLastFailedMessage] = useState(null);

  // Active profile is typing when typingProfileId matches activeProfileId
  const isTyping = Boolean(typingProfileId && typingProfileId === activeProfileId);

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
   * and the AI reply to local state.
   * Natural typing delay is calibrated and served by the backend.
   */
  const sendChatMessage = useCallback(async (profileId, messageText) => {
    const text = messageText.trim();
    if (!text || typingProfileId === profileId) return;

    setError(null);
    setLastFailedMessage(null);

    // Append user message immediately (optimistic UI)
    const userMsg = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: text,
      timestamp: new Date().toISOString(),
    };

    setMessagesByProfile((prev) => ({
      ...prev,
      [profileId]: [...(prev[profileId] || []), userMsg],
    }));

    // Activate typing indicator for this profile
    setTypingProfileId(profileId);

    try {
      const response = await sendMessage(profileId, userId, text);
      const rawReply = (response?.reply || '').trim();

      if (rawReply) {
        const aiMsg = {
          id: `ai-${Date.now()}`,
          role: 'assistant',
          content: rawReply,
          timestamp: new Date().toISOString(),
        };

        setMessagesByProfile((prev) => ({
          ...prev,
          [profileId]: [...(prev[profileId] || []), aiMsg],
        }));
      }

      setLastFailedMessage(null);
    } catch (err) {
      setLastFailedMessage({ profileId, messageText: text });
      setError(err.message || 'Reply aane mein thoda time lag gaya. Ek baar phir try karoge?');
    } finally {
      setTypingProfileId((current) => (current === profileId ? null : current));
    }
  }, [userId, typingProfileId]);

  /**
   * Retry sending the last failed message without adding a duplicate user bubble.
   */
  const retryLastMessage = useCallback(async (profileId) => {
    if (!lastFailedMessage || lastFailedMessage.profileId !== profileId || typingProfileId === profileId) return;

    const text = lastFailedMessage.messageText;
    setError(null);
    setTypingProfileId(profileId);

    try {
      const response = await sendMessage(profileId, userId, text);
      const rawReply = (response?.reply || '').trim();

      if (rawReply) {
        const aiMsg = {
          id: `ai-${Date.now()}`,
          role: 'assistant',
          content: rawReply,
          timestamp: new Date().toISOString(),
        };

        setMessagesByProfile((prev) => ({
          ...prev,
          [profileId]: [...(prev[profileId] || []), aiMsg],
        }));
      }

      setLastFailedMessage(null);
      setError(null);
    } catch (err) {
      setError(err.message || 'Reply aane mein thoda time lag gaya. Ek baar phir try karoge?');
    } finally {
      setTypingProfileId((current) => (current === profileId ? null : current));
    }
  }, [userId, typingProfileId, lastFailedMessage]);

  const getMessages = useCallback(
    (profileId) => messagesByProfile[profileId] || [],
    [messagesByProfile]
  );

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const clearConversation = useCallback(async (profileId) => {
    try {
      await clearChatConversation(profileId, userId);
      setMessagesByProfile((prev) => ({ ...prev, [profileId]: [] }));
      setLastFailedMessage(null);
      setError(null);
    } catch (err) {
      // The clear-chat modal owns this error state; do not show a
      // message-send failure underneath it.
      setError(null);
      throw err;
    }
  }, [userId]);

  const value = {
    userId,
    activeProfileId,
    setActiveProfileId,
    isTyping,
    error,
    clearError,
    lastFailedMessage,
    retryLastMessage,
    getMessages,
    sendChatMessage,
    loadHistory,
    clearConversation,
    messagesByProfile,
  };

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}

export function useChatContext() {
  const ctx = useContext(ChatContext);
  if (!ctx) throw new Error('useChatContext must be used inside <ChatProvider>');
  return ctx;
}
