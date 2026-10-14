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
  const [isTyping, setIsTyping] = useState(false);
  const [error, setError] = useState(null);
  const [lastFailedMessage, setLastFailedMessage] = useState(null);

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
    const text = messageText.trim();
    if (!text || isTyping) return;

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

    // Show typing indicator
    setIsTyping(true);

    try {
      const startTime = Date.now();
      const response = await sendMessage(profileId, userId, text);
      const apiDuration = Date.now() - startTime;

      // Realistic Human Typing Simulation:
      // Real humans read/think for 600-900ms, then type at natural phone texting speed (~26ms per char).
      // Longer responses naturally take proportionally longer to type.
      const replyLength = (response?.reply || '').length;
      const readingTime = 600 + Math.random() * 300;
      const typingTime = replyLength * 26;
      const targetDuration = Math.min(Math.max(readingTime + typingTime, 1400), 4500);

      const remainingTime = targetDuration - apiDuration;
      if (remainingTime > 0) {
        await new Promise((resolve) => setTimeout(resolve, remainingTime));
      }

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
      setLastFailedMessage(null);
    } catch (err) {
      setLastFailedMessage({ profileId, messageText: text });
      setError(err.message || 'Reply aane mein thoda time lag gaya. Ek baar phir try karoge?');
    } finally {
      setIsTyping(false);
    }
  }, [userId, isTyping]);

  /**
   * Retry sending the last failed message without adding a duplicate user bubble.
   */
  const retryLastMessage = useCallback(async (profileId) => {
    if (!lastFailedMessage || lastFailedMessage.profileId !== profileId || isTyping) return;

    const text = lastFailedMessage.messageText;
    setError(null);
    setIsTyping(true);

    try {
      const startTime = Date.now();
      const response = await sendMessage(profileId, userId, text);
      const apiDuration = Date.now() - startTime;

      const replyLength = (response?.reply || '').length;
      const readingTime = 600 + Math.random() * 300;
      const typingTime = replyLength * 26;
      const targetDuration = Math.min(Math.max(readingTime + typingTime, 1400), 4500);

      const remainingTime = targetDuration - apiDuration;
      if (remainingTime > 0) {
        await new Promise((resolve) => setTimeout(resolve, remainingTime));
      }

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
      setLastFailedMessage(null);
      setError(null);
    } catch (err) {
      setError(err.message || 'Reply aane mein thoda time lag gaya. Ek baar phir try karoge?');
    } finally {
      setIsTyping(false);
    }
  }, [userId, isTyping, lastFailedMessage]);

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
      setError(err.message || 'Failed to clear conversation.');
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
