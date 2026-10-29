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
  const activeTurnRef = useRef({});
  const messageQueueRef = useRef({});
  const isProcessingQueueRef = useRef({});

  /**
   * Load conversation history from the backend for a given profile.
   * Called when navigating to a chat page.
   * Splits any multi-bubble turns and strips '|||' from user-facing text.
   */
  const loadHistory = useCallback(async (profileId) => {
    try {
      const data = await getChatHistory(profileId, userId);
      const messages = [];
      (data.messages || []).forEach((m) => {
        const rawContent = (m.content || '').trim();
        if (m.role === 'assistant' && rawContent.includes('|||')) {
          const parts = rawContent.split('|||').map((s) => s.replace(/\|\|\|/g, '').trim()).filter(Boolean);
          parts.forEach((part, idx) => {
            messages.push({
              id: `${m.id}-${idx}`,
              role: m.role,
              content: part,
              timestamp: m.created_at,
            });
          });
        } else {
          messages.push({
            id: m.id,
            role: m.role,
            content: rawContent.replace(/\|\|\|/g, '').trim(),
            timestamp: m.created_at,
          });
        }
      });
      setMessagesByProfile((prev) => ({ ...prev, [profileId]: messages }));
    } catch {
      // Non-fatal: history just won't be pre-populated
    }
  }, [userId]);

  /**
   * Drain and process message queue sequentially per companion profile.
   * Allows user to send multiple messages in a row without UI lock (WhatsApp-style).
   */
  const drainQueue = useCallback(async (profileId) => {
    if (isProcessingQueueRef.current[profileId]) return;
    isProcessingQueueRef.current[profileId] = true;

    try {
      while ((messageQueueRef.current[profileId] || []).length > 0) {
        const item = messageQueueRef.current[profileId].shift();
        const text = item.text;

        setPendingProfileId(profileId);

        // Natural short reading delay based on user message length
        const wordCount = text.split(/\s+/).filter(Boolean).length;
        const readDelayMs = Math.min(1000, Math.max(400, 300 + wordCount * 30));

        const requestPromise = sendMessage(profileId, userId, text);
        await new Promise((resolve) => setTimeout(resolve, readDelayMs));

        setTypingProfileId(profileId);

        try {
          const response = await requestPromise;
          const rawReply = (response?.reply || '').trim();

          if (rawReply) {
            // Typing delay: natural short duration
            const typingDelayMs = Math.min(1800, Math.max(500, rawReply.length * 15));
            await new Promise((resolve) => setTimeout(resolve, typingDelayMs));

            // Extract bubbles (support ||| and multi-line breaks)
            let bubbles = [];
            if (rawReply.includes('|||')) {
              bubbles = rawReply.split('|||').map((s) => s.replace(/\|\|\|/g, '').trim()).filter(Boolean);
            } else if (rawReply.includes('\n\n')) {
              bubbles = rawReply.split('\n\n').map((s) => s.replace(/\|\|\|/g, '').trim()).filter(Boolean);
            } else {
              bubbles = [rawReply.replace(/\|\|\|/g, '').trim()];
            }
            if (bubbles.length === 0) bubbles = [rawReply.replace(/\|\|\|/g, '').trim()];

            for (let i = 0; i < bubbles.length; i++) {
              const bubbleText = bubbles[i];
              const aiMsg = {
                id: `ai-${Date.now()}-${i}`,
                role: 'assistant',
                content: bubbleText,
                timestamp: new Date().toISOString(),
              };

              setMessagesByProfile((prev) => ({
                ...prev,
                [profileId]: [...(prev[profileId] || []), aiMsg],
              }));

              if (i < bubbles.length - 1) {
                setTypingProfileId(profileId);
                await new Promise((resolve) => setTimeout(resolve, 400));
              }
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
      }
    } finally {
      isProcessingQueueRef.current[profileId] = false;
      setTypingProfileId((current) => (current === profileId ? null : current));
      setPendingProfileId((current) => (current === profileId ? null : current));
    }
  }, [userId]);

  /**
   * Send a message to the backend and append both the user message
   * and the AI reply bubbles to local state.
   */
  const sendChatMessage = useCallback(async (profileId, messageText) => {
    const text = messageText.trim();
    if (!text) return;

    setError(null);
    setLastFailedMessage(null);

    // 1. Immediately append user message to local state (optimistic UI)
    const userMsg = {
      id: `user-${Date.now()}-${Math.random()}`,
      role: 'user',
      content: text,
      timestamp: new Date().toISOString(),
    };

    setMessagesByProfile((prev) => ({
      ...prev,
      [profileId]: [...(prev[profileId] || []), userMsg],
    }));

    // 2. Push message into queue & drain sequentially
    if (!messageQueueRef.current[profileId]) {
      messageQueueRef.current[profileId] = [];
    }
    messageQueueRef.current[profileId].push({ text });

    drainQueue(profileId);
  }, [drainQueue]);

  /**
   * Retry sending the last failed message without adding a duplicate user bubble.
   */
  const retryLastMessage = useCallback(async (profileId) => {
    if (!lastFailedMessage || lastFailedMessage.profileId !== profileId || typingProfileId === profileId || pendingProfileId === profileId) return;

    const text = lastFailedMessage.messageText;
    setError(null);
    setPendingProfileId(profileId);

    const currentTurnToken = (activeTurnRef.current[profileId] || 0) + 1;
    activeTurnRef.current[profileId] = currentTurnToken;

    const wordCount = text.split(/\s+/).filter(Boolean).length;
    const readDelayMs = Math.min(2000, Math.max(900, 800 + wordCount * 50));
    const requestPromise = sendMessage(profileId, userId, text);

    await new Promise((resolve) => setTimeout(resolve, readDelayMs));
    if (activeTurnRef.current[profileId] !== currentTurnToken) return;
    setTypingProfileId(profileId);

    try {
      const response = await requestPromise;
      if (activeTurnRef.current[profileId] !== currentTurnToken) return;
      const rawReply = (response?.reply || '').trim();

      if (rawReply) {
        const typingDelayMs = Math.min(4000, Math.max(1000, rawReply.length * 28));
        await new Promise((resolve) => setTimeout(resolve, typingDelayMs));
        if (activeTurnRef.current[profileId] !== currentTurnToken) return;

        let bubbles = [];
        if (rawReply.includes('|||')) {
          bubbles = rawReply.split('|||').map((s) => s.replace(/\|\|\|/g, '').trim()).filter(Boolean);
        } else {
          bubbles = rawReply.split('\n').map((s) => s.replace(/\|\|\|/g, '').trim()).filter(Boolean);
        }
        if (bubbles.length === 0) bubbles = [rawReply.replace(/\|\|\|/g, '').trim()];

        for (let i = 0; i < bubbles.length; i++) {
          if (activeTurnRef.current[profileId] !== currentTurnToken) break;

          const bubbleText = bubbles[i];
          const aiMsg = {
            id: `ai-${Date.now()}-${i}`,
            role: 'assistant',
            content: bubbleText,
            timestamp: new Date().toISOString(),
          };
          setMessagesByProfile((prev) => ({
            ...prev,
            [profileId]: [...(prev[profileId] || []), aiMsg],
          }));

          if (i < bubbles.length - 1) {
            setTypingProfileId(profileId);
            const gapDelayMs = Math.min(1500, Math.max(800, bubbles[i + 1].length * 25));
            await new Promise((resolve) => setTimeout(resolve, gapDelayMs));
            if (activeTurnRef.current[profileId] !== currentTurnToken) break;
          }
        }
      }

      setLastFailedMessage(null);
      setError(null);
    } catch (err) {
      if (activeTurnRef.current[profileId] === currentTurnToken) {
        setError(err.message || 'Reply aane mein thoda time lag gaya. Ek baar phir try karoge?');
      }
    } finally {
      if (activeTurnRef.current[profileId] === currentTurnToken) {
        setTypingProfileId((current) => (current === profileId ? null : current));
        setPendingProfileId((current) => (current === profileId ? null : current));
      }
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
