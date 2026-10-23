/**
 * services/api.js
 * ----------------
 * Centralized Axios API client. All backend calls go through this module.
 * Components never import axios directly — they import from here.
 *
 * Base URL is read from VITE_API_BASE_URL env variable (defaults to localhost:8000).
 */

import axios from 'axios';

// When running in unified Docker / Hugging Face Spaces, uses relative URL;
// In local dev, automatically uses the current hostname (localhost or LAN IP like 192.168.x.x) on port 8000
const getBaseUrl = () => {
  if (import.meta.env.VITE_API_BASE_URL !== undefined) {
    return import.meta.env.VITE_API_BASE_URL;
  }
  if (typeof window !== 'undefined') {
    const host = window.location.hostname || '';
    // If accessing via ngrok, Cloudflare tunnel, or remote domain, use relative path so Vite proxies to backend port 8000
    if (host.includes('ngrok') || host.includes('trycloudflare') || host.includes('loca.lt')) {
      return '';
    }
    if (import.meta.env.DEV) {
      return `http://${host || 'localhost'}:8000`;
    }
  }
  return '';
};

const BASE_URL = getBaseUrl();

const apiClient = axios.create({
  baseURL: BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30s — generous for LLM response times
});

// ---- Response interceptor: normalize errors ----
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    let message = 'Something went wrong';
    const requestUrl = error.config?.url || '';
    const isChatReplyRequest = error.config?.method?.toLowerCase() === 'post'
      && requestUrl.includes('/chat/');
    const isConnectionFailure = error.code === 'ECONNABORTED'
      || error.message?.includes('timeout')
      || error.message?.includes('Network Error');

    if (isConnectionFailure && isChatReplyRequest) {
      message = 'Reply aane mein thoda time lag gaya. Ek baar phir try karoge?';
    } else if (isConnectionFailure) {
      message = 'Backend se connection nahi ho pa raha. Server check karke dobara try karo.';
    } else if (error.response?.data?.detail) {
      message = error.response.data.detail;
    } else if (error.response?.data?.message) {
      message = error.response.data.message;
    } else if (error.message) {
      message = error.message;
    }
    return Promise.reject(new Error(message));
  }
);


// ============================================================
// Profile API calls
// ============================================================

export const getProfiles = async () => {
  const { data } = await apiClient.get('/profiles');
  return data;
};

export const getProfile = async (profileId) => {
  const { data } = await apiClient.get(`/profiles/${profileId}`);
  return data;
};

export const createProfile = async (profileData) => {
  const { data } = await apiClient.post('/profiles', profileData);
  return data;
};

export const deleteProfile = async (profileId) => {
  const { data } = await apiClient.delete(`/profiles/${profileId}`);
  return data;
};


// ============================================================
// Chat API calls
// ============================================================

export const sendMessage = async (profileId, userId, message) => {
  const { data } = await apiClient.post(`/chat/${profileId}`, {
    user_id: userId,
    message,
  });
  return data;
};

export const getChatHistory = async (profileId, userId) => {
  const { data } = await apiClient.get(`/chat/${profileId}/history`, {
    params: { user_id: userId },
  });
  return data;
};

export const getOpener = async (profileId) => {
  const { data } = await apiClient.get(`/chat/${profileId}/opener`);
  return data;
};

export const clearChatConversation = async (profileId, userId) => {
  const { data } = await apiClient.delete(`/chat/${profileId}/messages`, {
    params: { user_id: userId },
  });
  return data;
};

export const nudgeUser = async (profileId, userId, minIdleSeconds = 180) => {
  const { data } = await apiClient.post(`/chat/${profileId}/nudge`, {
    user_id: userId,
    min_idle_seconds: minIdleSeconds,
  });
  return data;
};


// ============================================================
// Memory & Privacy API calls
// ============================================================

export const getMemories = async (userId, profileId = null, memoryType = null) => {
  const params = { user_id: userId };
  if (profileId) params.profile_id = profileId;
  if (memoryType) params.memory_type = memoryType;
  const { data } = await apiClient.get('/memories', { params });
  return data;
};

export const updateMemory = async (memoryId, userId, memoryText) => {
  const { data } = await apiClient.patch(`/memories/${memoryId}`, { memory_text: memoryText }, {
    params: { user_id: userId },
  });
  return data;
};

export const deleteMemory = async (memoryId, userId) => {
  const { data } = await apiClient.delete(`/memories/${memoryId}`, {
    params: { user_id: userId },
  });
  return data;
};

export const forgetMemories = async (userId, scope = 'all', profileId = null) => {
  const { data } = await apiClient.post('/memories/forget', {
    user_id: userId,
    scope,
    profile_id: profileId,
  });
  return data;
};

export const exportUserData = async (userId) => {
  const { data } = await apiClient.get('/privacy/data-export', {
    params: { user_id: userId },
  });
  return data;
};

export const deleteAccountData = async (userId) => {
  const { data } = await apiClient.delete('/account/data', {
    params: { user_id: userId },
  });
  return data;
};
