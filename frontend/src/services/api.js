/**
 * services/api.js
 * ----------------
 * Centralized Axios API client. All backend calls go through this module.
 * Components never import axios directly — they import from here.
 *
 * Base URL is read from VITE_API_BASE_URL env variable (defaults to localhost:8000).
 */

import axios from 'axios';

// When running in unified Docker / Hugging Face Spaces, uses relative URL; in local dev, points to localhost:8000
const BASE_URL =
  import.meta.env.VITE_API_BASE_URL !== undefined
    ? import.meta.env.VITE_API_BASE_URL
    : import.meta.env.DEV
    ? 'http://localhost:8000'
    : '';

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
    const message =
      error.response?.data?.detail ||
      error.response?.data?.message ||
      error.message ||
      'Something went wrong';
    return Promise.reject(new Error(message));
  }
);


// ============================================================
// Profile API calls
// ============================================================

/**
 * Fetch all available AI persona profiles (for the homepage grid).
 * @returns {Promise<Array<{id, name, short_bio, avatar_url}>>}
 */
export const getProfiles = async () => {
  const { data } = await apiClient.get('/profiles');
  return data;
};

/**
 * Fetch full public info for a single persona profile.
 * @param {string} profileId
 * @returns {Promise<Object>}
 */
export const getProfile = async (profileId) => {
  const { data } = await apiClient.get(`/profiles/${profileId}`);
  return data;
};

/**
 * Create a new AI dating persona.
 * @param {Object} profileData
 * @returns {Promise<Object>}
 */
export const createProfile = async (profileData) => {
  const { data } = await apiClient.post('/profiles', profileData);
  return data;
};

/**
 * Delete an AI persona profile and all its associated chats.
 * @param {string} profileId
 * @returns {Promise<Object>}
 */
export const deleteProfile = async (profileId) => {
  const { data } = await apiClient.delete(`/profiles/${profileId}`);
  return data;
};



// ============================================================
// Chat API calls
// ============================================================

/**
 * Send a user message to an AI persona and receive a reply.
 * @param {string} profileId  - Which persona to chat with.
 * @param {string} userId     - The user's session UUID.
 * @param {string} message    - The user's message text.
 * @returns {Promise<{reply: string, profile_id: string}>}
 */
export const sendMessage = async (profileId, userId, message) => {
  const { data } = await apiClient.post(`/chat/${profileId}`, {
    user_id: userId,
    message,
  });
  return data;
};

/**
 * Fetch conversation history for a user+profile pair.
 * @param {string} profileId
 * @param {string} userId
 * @returns {Promise<{messages: Array}>}
 */
export const getChatHistory = async (profileId, userId) => {
  const { data } = await apiClient.get(`/chat/${profileId}/history`, {
    params: { user_id: userId },
  });
  return data;
};

/**
 * Fetch a natural opening greeting for a persona.
 * @param {string} profileId
 * @returns {Promise<{profile_id: string, name: string, opener: string}>}
 */
export const getOpener = async (profileId) => {
  const { data } = await apiClient.get(`/chat/${profileId}/opener`);
  return data;
};
