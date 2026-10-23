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
import { sendMessage, getChatHistory, clearChatConversation, nudgeUser } from '../services/api';

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
  const [pendingProfileId, setPendingProfileId] = useState(null);
  const [typingProfileId, setTypingProfileId] = useState(null);
  const [error, setError] = useState(null);
  const [lastFailedMessage, setLastFailedMessage] = useState(null);
  const lastNudgeAttemptRef = useRef({});

  // Active profile is typing when typingProfileId matches activeProfileId
  const isTyping = Boolean(typingProfileId && typingProfileId === activeProfileId);
  // Active profile is pending (reading or typing) when pendingProfileId matches activeProfileId
  const isPending = Boolean(pendingProfileId && pendingProfileId === activeProfileId);

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
   *
   * Flow:
   * 1. User message delivered instantly.
   * 2. Natural READ DELAY: Companion first sees and reads message (no typing indicator yet).
   * 3. TYPING PHASE: Companion starts typing (dots bounce for natural duration).
   * 4. Reply appears cleanly!
   */
  const sendChatMessage = useCallback(async (profileId, messageText) => {
    const text = messageText.trim();
    if (!text || typingProfileId === profileId || pendingProfileId === profileId) return;

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

    // Mark as pending/reading
    setPendingProfileId(profileId);

    // Calculate natural reading delay based on words in user's message
    const wordCount = text.split(/\s+/).filter(Boolean).length;
    const readDelayMs = Math.min(2200, Math.max(1000, 850 + wordCount * 65 + Math.random() * 300));

    // Start API request in parallel
    const requestPromise = sendMessage(profileId, userId, text);

    // Wait for reading delay before starting typing indicator
    await new Promise((resolve) => setTimeout(resolve, readDelayMs));

    // Turn ON typing indicator after reading
    setTypingProfileId(profileId);

    try {
      const response = await requestPromise;
      const rawReply = (response?.reply || '').trim();

      if (rawReply) {
        // Typing delay of 1-4 seconds based on reply length (User Requirement 7)
        // Average texting speed: ~28ms per character, bounded within [1000ms, 4000ms]
        const typingDelayMs = Math.min(4000, Math.max(1000, rawReply.length * 28));
        await new Promise((resolve) => setTimeout(resolve, typingDelayMs));

        // If reply has 2 lines with different thoughts, send as 2 separate messages with short gap
        const lines = rawReply.split('\n').map((l) => l.trim()).filter(Boolean);
        if (lines.length >= 2) {
          // Deliver first message bubble
          const firstMsg = {
            id: `ai-${Date.now()}-1`,
            role: 'assistant',
            content: lines[0],
            timestamp: new Date().toISOString(),
          };
          setMessagesByProfile((prev) => ({
            ...prev,
            [profileId]: [...(prev[profileId] || []), firstMsg],
          }));

          // Intermediate typing indicator with short natural gap for second thought
          setTypingProfileId(profileId);
          const gapDelayMs = Math.min(1500, Math.max(800, lines[1].length * 25));
          await new Promise((resolve) => setTimeout(resolve, gapDelayMs));

          const secondMsg = {
            id: `ai-${Date.now()}-2`,
            role: 'assistant',
            content: lines.slice(1).join('\n'),
            timestamp: new Date().toISOString(),
          };
          setMessagesByProfile((prev) => ({
            ...prev,
            [profileId]: [...(prev[profileId] || []), secondMsg],
          }));
        } else {
          // Single-line message bubble
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
      }

      setLastFailedMessage(null);
    } catch (err) {
      setLastFailedMessage({ profileId, messageText: text });
      setError(err.message || 'Reply aane mein thoda time lag gaya. Ek baar phir try karoge?');
    } finally {
      setTypingProfileId((current) => (current === profileId ? null : current));
      setPendingProfileId((current) => (current === profileId ? null : current));
    }
  }, [userId, typingProfileId, pendingProfileId]);

  /**
   * Retry sending the last failed message without adding a duplicate user bubble.
   */
  const retryLastMessage = useCallback(async (profileId) => {
    if (!lastFailedMessage || lastFailedMessage.profileId !== profileId || typingProfileId === profileId || pendingProfileId === profileId) return;

    const text = lastFailedMessage.messageText;
    setError(null);
    setPendingProfileId(profileId);

    const wordCount = text.split(/\s+/).filter(Boolean).length;
    const readDelayMs = Math.min(2000, Math.max(900, 800 + wordCount * 50));
    const requestPromise = sendMessage(profileId, userId, text);

    await new Promise((resolve) => setTimeout(resolve, readDelayMs));
    setTypingProfileId(profileId);

    try {
      const response = await requestPromise;
      const rawReply = (response?.reply || '').trim();

      if (rawReply) {
        const typingDelayMs = Math.min(4000, Math.max(1000, rawReply.length * 28));
        await new Promise((resolve) => setTimeout(resolve, typingDelayMs));

        const lines = rawReply.split('\n').map((l) => l.trim()).filter(Boolean);
        if (lines.length >= 2) {
          const firstMsg = {
            id: `ai-${Date.now()}-1`,
            role: 'assistant',
            content: lines[0],
            timestamp: new Date().toISOString(),
          };
          setMessagesByProfile((prev) => ({
            ...prev,
            [profileId]: [...(prev[profileId] || []), firstMsg],
          }));

          setTypingProfileId(profileId);
          const gapDelayMs = Math.min(1500, Math.max(800, lines[1].length * 25));
          await new Promise((resolve) => setTimeout(resolve, gapDelayMs));

          const secondMsg = {
            id: `ai-${Date.now()}-2`,
            role: 'assistant',
            content: lines.slice(1).join('\n'),
            timestamp: new Date().toISOString(),
          };
          setMessagesByProfile((prev) => ({
            ...prev,
            [profileId]: [...(prev[profileId] || []), secondMsg],
          }));
        } else {
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
      }

      setLastFailedMessage(null);
      setError(null);
    } catch (err) {
      setError(err.message || 'Reply aane mein thoda time lag gaya. Ek baar phir try karoge?');
    } finally {
      setTypingProfileId((current) => (current === profileId ? null : current));
      setPendingProfileId((current) => (current === profileId ? null : current));
    }
  }, [userId, typingProfileId, pendingProfileId, lastFailedMessage]);

  /**
   * Inactivity Nudge: When the user goes idle without ending the conversation,
   * trigger a single proactive re-engagement message from the companion.
   */
  const checkAndTriggerNudge = useCallback(async (profileId) => {
    if (!profileId || typingProfileId === profileId || pendingProfileId === profileId) return;

    // Rate-limit frontend nudge attempts (at most once every 3 minutes per profile)
    const lastAttempt = lastNudgeAttemptRef.current[profileId];
    if (lastAttempt && Date.now() - lastAttempt < 180000) {
      return;
    }

    try {
      const data = await nudgeUser(profileId, userId, 180);
      if (data && data.nudged && data.reply) {
        lastNudgeAttemptRef.current[profileId] = Date.now();

        // Show typing indicator for 1.6s before nudge message arrives
        setTypingProfileId(profileId);
        await new Promise((resolve) => setTimeout(resolve, 1600));
        setTypingProfileId(null);

        const aiMsg = {
          id: `ai-${Date.now()}`,
          role: 'assistant',
          content: data.reply.trim(),
          timestamp: new Date().toISOString(),
        };

        setMessagesByProfile((prev) => ({
          ...prev,
          [profileId]: [...(prev[profileId] || []), aiMsg],
        }));

        // Notify user if tab is hidden
        if (typeof document !== 'undefined' && document.hidden) {
          const originalTitle = document.title;
          document.title = '💬 New message from your match!';
          const handleFocus = () => {
            document.title = originalTitle;
            window.removeEventListener('focus', handleFocus);
          };
          window.addEventListener('focus', handleFocus);
        }
      }
    } catch {
      // Non-fatal background check
    }
  }, [userId, typingProfileId, pendingProfileId]);

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
    isPending,
    error,
    clearError,
    lastFailedMessage,
    retryLastMessage,
    getMessages,
    sendChatMessage,
    checkAndTriggerNudge,
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
