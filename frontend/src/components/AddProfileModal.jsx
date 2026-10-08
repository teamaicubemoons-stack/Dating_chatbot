/**
 * components/AddProfileModal.jsx
 * --------------------------------
 * Modal dialog for creating a new dating persona.
 * Takes all standard dating profile information:
 * Name, Age, Gender, Country (mandatory), City, Bio, Backstory,
 * Tone, Interests, Dating Intent, and Avatar selection.
 */

import React, { useState } from 'react';
import { createProfile } from '../services/api';

const PRESET_AVATARS = [
  { id: 'luna', label: 'Luna', url: '/avatars/luna.png' },
  { id: 'ethan', label: 'Ethan', url: '/avatars/ethan.png' },
  { id: 'zara', label: 'Zara', url: '/avatars/zara.png' },
  { id: 'diya', label: 'Diya', url: '/avatars/diya.jpg' },
];

const POPULAR_COUNTRIES = [
  'India',
  'United States',
  'United Kingdom',
  'Canada',
  'Australia',
  'Germany',
  'France',
  'Japan',
  'Singapore',
  'United Arab Emirates',
  'Other',
];

const INTEREST_OPTIONS = [
  'Travel', 'Photography', 'Music', 'Coffee', 'Fitness',
  'Art & Design', 'Foodie', 'Books', 'Movies', 'Tech & Startups',
  'Nature & Treks', 'Late Night Drives'
];

const TONE_OPTIONS = [
  'warm & playful',
  'witty & sarcastic',
  'chill & easygoing',
  'bold & energetic',
  'intellectual & deep',
  'sweet & romantic',
];

export default function AddProfileModal({ isOpen, onClose, onProfileCreated }) {
  const [formData, setFormData] = useState({
    name: '',
    age: 24,
    gender: 'Female',
    country: 'India',
    customCountry: '',
    city: '',
    short_bio: '',
    tone: 'warm & playful',
    interests: ['Travel', 'Music'],
    relationship_intent: 'Looking for fun conversations & genuine connection',
    backstory: '',
    avatar_url: '/avatars/luna.png',
  });

  const [selectedPresetId, setSelectedPresetId] = useState('luna');
  const [customAvatarUrl, setCustomAvatarUrl] = useState('');
  const [customImgStatus, setCustomImgStatus] = useState('idle'); // 'idle' | 'loading' | 'valid' | 'invalid'
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  if (!isOpen) return null;

  const handleChange = (field, value) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const toggleInterest = (tag) => {
    setFormData((prev) => {
      const exists = prev.interests.includes(tag);
      return {
        ...prev,
        interests: exists
          ? prev.interests.filter((t) => t !== tag)
          : [...prev.interests, tag],
      };
    });
  };

  const handleSelectPreset = (av) => {
    setSelectedPresetId(av.id);
    setCustomAvatarUrl('');
    setCustomImgStatus('idle');
    setFormData((prev) => ({ ...prev, avatar_url: av.url }));
  };

  const handleCustomUrlChange = (val) => {
    setCustomAvatarUrl(val);
    const trimmed = val.trim();
    if (!trimmed) {
      setSelectedPresetId('luna');
      setCustomImgStatus('idle');
      setFormData((prev) => ({ ...prev, avatar_url: '/avatars/luna.png' }));
      return;
    }
    setSelectedPresetId(null);
    setFormData((prev) => ({ ...prev, avatar_url: trimmed }));
    setCustomImgStatus('loading');

    const img = new Image();
    img.onload = () => setCustomImgStatus('valid');
    img.onerror = () => setCustomImgStatus('invalid');
    img.src = trimmed;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);

    const country = formData.country === 'Other'
      ? formData.customCountry.trim()
      : formData.country;

    if (!formData.name.trim()) {
      setError('Please enter a name for the profile.');
      return;
    }
    if (!country) {
      setError('Please specify a country.');
      return;
    }
    if (!formData.short_bio.trim()) {
      setError('Please add a short bio or headline.');
      return;
    }

    let finalAvatarUrl = formData.avatar_url;
    if (customAvatarUrl.trim()) {
      if (customImgStatus === 'invalid') {
        setError('The custom avatar image URL could not be loaded. Please enter a valid direct image link (.jpg/.png/.webp) or choose one of the AI avatars.');
        return;
      }
      finalAvatarUrl = customAvatarUrl.trim();
    } else if (selectedPresetId) {
      const preset = PRESET_AVATARS.find((p) => p.id === selectedPresetId);
      finalAvatarUrl = preset ? preset.url : '/avatars/luna.png';
    }

    setLoading(true);
    try {
      const payload = {
        name: formData.name.trim(),
        age: parseInt(formData.age, 10) || 24,
        gender: formData.gender,
        country: country,
        city: formData.city.trim(),
        short_bio: formData.short_bio.trim(),
        tone: formData.tone,
        interests: formData.interests,
        relationship_intent: formData.relationship_intent,
        backstory: formData.backstory.trim(),
        avatar_url: finalAvatarUrl,
      };

      const newProfile = await createProfile(payload);
      onProfileCreated(newProfile);
      onClose();
    } catch (err) {
      setError(err.message || 'Failed to create profile. Please check the backend.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose} role="dialog" aria-modal="true">
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        {/* Modal Header */}
        <div className="modal-header">
          <div>
            <h2 className="modal-title">Create Dating Profile</h2>
            <p className="modal-subtitle">Add a unique companion persona with authentic personality & location</p>
          </div>
          <button className="modal-close-btn" onClick={onClose} aria-label="Close modal">
            ✕
          </button>
        </div>

        {error && (
          <div className="modal-error-banner" role="alert">
            <span>⚠ {error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="modal-form">
          {/* Row 1: Name, Age, Gender */}
          <div className="form-row form-row--three">
            <div className="form-group">
              <label className="form-label">Full Name *</label>
              <input
                type="text"
                className="form-input"
                placeholder="e.g. Maya Sharma"
                value={formData.name}
                onChange={(e) => handleChange('name', e.target.value)}
                required
              />
            </div>
            <div className="form-group">
              <label className="form-label">Age *</label>
              <input
                type="number"
                className="form-input"
                min="18"
                max="99"
                value={formData.age}
                onChange={(e) => handleChange('age', e.target.value)}
                required
              />
            </div>
            <div className="form-group">
              <label className="form-label">Gender</label>
              <select
                className="form-select"
                value={formData.gender}
                onChange={(e) => handleChange('gender', e.target.value)}
              >
                <option value="Female">Female</option>
                <option value="Male">Male</option>
                <option value="Non-binary">Non-binary</option>
              </select>
            </div>
          </div>

          {/* Row 2: Country & City */}
          <div className="form-row form-row--two">
            <div className="form-group">
              <label className="form-label">Country *</label>
              <select
                className="form-select"
                value={formData.country}
                onChange={(e) => handleChange('country', e.target.value)}
              >
                {POPULAR_COUNTRIES.map((c) => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
              {formData.country === 'Other' && (
                <input
                  type="text"
                  className="form-input"
                  style={{ marginTop: '8px' }}
                  placeholder="Enter country name..."
                  value={formData.customCountry}
                  onChange={(e) => handleChange('customCountry', e.target.value)}
                  required
                />
              )}
            </div>
            <div className="form-group">
              <label className="form-label">City</label>
              <input
                type="text"
                className="form-input"
                placeholder="e.g. Mumbai, New York, London"
                value={formData.city}
                onChange={(e) => handleChange('city', e.target.value)}
              />
            </div>
          </div>

          {/* Row 3: Short Bio */}
          <div className="form-group">
            <label className="form-label">Short Bio / Profile Headline *</label>
            <input
              type="text"
              className="form-input"
              placeholder="e.g. Architect by day, vinyl collector by night. Loves chai & long drives."
              value={formData.short_bio}
              onChange={(e) => handleChange('short_bio', e.target.value)}
              maxLength={180}
              required
            />
            <span className="form-hint">{formData.short_bio.length}/180 characters</span>
          </div>

          {/* Row 4: Tone & Dating Intent */}
          <div className="form-row form-row--two">
            <div className="form-group">
              <label className="form-label">Conversational Tone</label>
              <select
                className="form-select"
                value={formData.tone}
                onChange={(e) => handleChange('tone', e.target.value)}
              >
                {TONE_OPTIONS.map((t) => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Dating Intent</label>
              <input
                type="text"
                className="form-input"
                placeholder="e.g. Meaningful connection & chill weekend banter"
                value={formData.relationship_intent}
                onChange={(e) => handleChange('relationship_intent', e.target.value)}
              />
            </div>
          </div>

          {/* Row 5: Interests Chips */}
          <div className="form-group">
            <label className="form-label">Interests & Hobbies (select all that apply)</label>
            <div className="form-chips">
              {INTEREST_OPTIONS.map((tag) => {
                const selected = formData.interests.includes(tag);
                return (
                  <button
                    type="button"
                    key={tag}
                    className={`form-chip ${selected ? 'form-chip--selected' : ''}`}
                    onClick={() => toggleInterest(tag)}
                  >
                    {selected ? '✓ ' : '+ '}{tag}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Row 6: Detailed Backstory */}
          <div className="form-group">
            <label className="form-label">Backstory & Personality (AI Knowledge & Memory)</label>
            <textarea
              className="form-textarea"
              rows={3}
              placeholder="Where they grew up, passions, quirks, what makes them unique... This gives depth and authentic memory to your companion."
              value={formData.backstory}
              onChange={(e) => handleChange('backstory', e.target.value)}
            />
          </div>

          {/* Row 7: Avatar Selection & Live Preview */}
          <div className="form-group">
            <label className="form-label">Profile Avatar</label>
            <div className="form-avatar-section">
              {/* Presets */}
              <div style={{ fontSize: '0.78rem', fontWeight: 600, color: '#475569' }}>
                Select AI Avatar:
              </div>
              <div className="form-avatar-presets">
                {PRESET_AVATARS.map((av) => (
                  <div
                    key={av.id}
                    className={`form-avatar-option ${selectedPresetId === av.id ? 'form-avatar-option--selected' : ''}`}
                    onClick={() => handleSelectPreset(av)}
                    role="button"
                    tabIndex={0}
                  >
                    <img src={av.url} alt={av.label} />
                    <span>{av.label}</span>
                  </div>
                ))}
              </div>

              {/* Custom URL */}
              <div className="form-avatar-custom">
                <div style={{ fontSize: '0.78rem', fontWeight: 600, color: '#475569' }}>
                  Or enter Custom Image Link:
                </div>
                <div className="form-avatar-input-row">
                  <input
                    type="url"
                    className="form-input"
                    placeholder="Paste direct image link (e.g. https://.../photo.jpg)"
                    value={customAvatarUrl}
                    onChange={(e) => handleCustomUrlChange(e.target.value)}
                  />
                  {customAvatarUrl && (
                    <button
                      type="button"
                      className="btn btn--secondary"
                      style={{ padding: '6px 12px', fontSize: '0.78rem', flexShrink: 0 }}
                      onClick={() => handleCustomUrlChange('')}
                    >
                      Clear
                    </button>
                  )}
                </div>

                {/* Pinterest warning if user pastes a Pinterest webpage link */}
                {(customAvatarUrl.includes('pin.it') || (customAvatarUrl.includes('pinterest.') && !customAvatarUrl.match(/\.(jpg|jpeg|png|webp|gif)/i))) && (
                  <div className="form-avatar-alert-box">
                    ℹ <strong>Note on Pinterest:</strong> Links like <code>pin.it/...</code> are webpages, not images. To use a Pinterest photo, right-click the image on Pinterest and choose <em>"Copy image address"</em> (direct link ending in .jpg), or pick an AI Avatar above.
                  </div>
                )}

                {/* Live Preview Card */}
                <div className="form-avatar-live-preview">
                  <div className="form-avatar-preview-thumb">
                    {selectedPresetId ? (
                      <img
                        src={PRESET_AVATARS.find((p) => p.id === selectedPresetId)?.url}
                        alt="Preset Avatar"
                      />
                    ) : customImgStatus === 'valid' ? (
                      <img src={customAvatarUrl.trim()} alt="Custom Avatar Preview" />
                    ) : customImgStatus === 'loading' ? (
                      '⏳'
                    ) : (
                      (formData.name?.[0]?.toUpperCase() || '✦')
                    )}
                  </div>
                  <div className="form-avatar-status-text">
                    <div className="form-avatar-status-title">
                      {selectedPresetId
                        ? `Using AI Avatar: ${PRESET_AVATARS.find((p) => p.id === selectedPresetId)?.label}`
                        : customImgStatus === 'valid'
                          ? '✓ Custom avatar loaded successfully'
                          : customImgStatus === 'loading'
                            ? 'Verifying image link...'
                            : customImgStatus === 'invalid'
                              ? '⚠️ Image could not be loaded from this link'
                              : 'Paste a link above or select an AI Avatar'}
                    </div>
                    <div className="form-avatar-status-sub">
                      {selectedPresetId
                        ? 'High-resolution AI portrait will display on profile & chat'
                        : customImgStatus === 'valid'
                          ? 'Preview confirmed! This image will be saved.'
                          : customImgStatus === 'invalid'
                            ? 'Direct image URL (.jpg/.png) required — or pick an AI avatar above.'
                            : 'Supports direct URLs (.jpg, .png, .webp, Imgur, Unsplash)'}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Modal Actions */}
          <div className="modal-actions">
            <button
              type="button"
              className="btn btn--secondary"
              onClick={onClose}
              disabled={loading}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="btn btn--primary"
              disabled={loading}
            >
              {loading ? 'Creating Companion…' : 'Create Profile ✦'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
