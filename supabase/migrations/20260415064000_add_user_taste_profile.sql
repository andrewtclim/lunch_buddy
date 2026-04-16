-- Add user taste profile (embedding storage) + minor profile schema alignment.

-- Optional: keep naming consistent if older code used different column names.
-- This is a no-op if the column already exists.
alter table if exists public.profiles
  add column if not exists display_name text;

-- Store the model-owned preference vector separately from user-editable profile fields.
create table if not exists public.user_taste_profile (
  user_id uuid primary key references auth.users(id) on delete cascade,
  -- 768-dim embedding stored as float array (easy to round-trip with Python lists/NumPy).
  -- We keep it nullable until the user has enough interactions to initialize it.
  pref_vec real[],
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint pref_vec_dim_768 check (pref_vec is null or array_length(pref_vec, 1) = 768)
);

-- Reuse the same updated_at trigger function created in the earlier migration.
drop trigger if exists set_user_taste_profile_updated_at on public.user_taste_profile;
create trigger set_user_taste_profile_updated_at
before update on public.user_taste_profile
for each row
execute function public.set_current_timestamp_updated_at();

alter table public.user_taste_profile enable row level security;

-- Users can read only their own taste profile.
drop policy if exists "user_taste_profile_select_own" on public.user_taste_profile;
create policy "user_taste_profile_select_own"
on public.user_taste_profile
for select
using (auth.uid() = user_id);

-- Users can insert only their own taste profile row.
drop policy if exists "user_taste_profile_insert_own" on public.user_taste_profile;
create policy "user_taste_profile_insert_own"
on public.user_taste_profile
for insert
with check (auth.uid() = user_id);

-- Users can update only their own taste profile row.
drop policy if exists "user_taste_profile_update_own" on public.user_taste_profile;
create policy "user_taste_profile_update_own"
on public.user_taste_profile
for update
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

