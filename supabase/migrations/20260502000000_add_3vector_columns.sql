-- Add two new columns to user_pref for the 3-vector recommendation architecture.
--
-- original_profile_vector: the user's signup-text embedding, stored once and never overwritten.
--   NULL for users who pre-date this migration — those users fall back to the existing
--   preference_vector in the blending logic.
--
-- recent_choices: JSONB array of the last 5 dish embeddings (most-recent first), each stored
--   as a flat JSON array of 768 floats. Defaults to [] so existing rows are immediately valid.

ALTER TABLE user_pref
  ADD COLUMN IF NOT EXISTS original_profile_vector vector(768),
  ADD COLUMN IF NOT EXISTS recent_choices jsonb NOT NULL DEFAULT '[]'::jsonb;
