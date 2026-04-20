-- Align taste profile vector type and add recommendation event logging.

create extension if not exists vector;
create extension if not exists pgcrypto;

-- Move preference vectors to pgvector for consistency with existing embedding tables.
alter table public.user_taste_profile
  drop constraint if exists pref_vec_dim_768;

alter table public.user_taste_profile
  alter column pref_vec type vector(768)
  using case
    when pref_vec is null then null
    else pref_vec::vector(768)
  end;

-- One row per recommendation request/response event.
create table if not exists public.recommendation_events (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  pref_vec vector(768),
  allergies text[] not null default '{}'::text[],
  diets text[] not null default '{}'::text[],
  input_text text not null,
  model_output jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists recommendation_events_user_created_idx
  on public.recommendation_events (user_id, created_at desc);

alter table public.recommendation_events enable row level security;

-- Users can read only their own recommendation history.
drop policy if exists "recommendation_events_select_own" on public.recommendation_events;
create policy "recommendation_events_select_own"
on public.recommendation_events
for select
using (auth.uid() = user_id);

-- Users can create only their own recommendation events.
drop policy if exists "recommendation_events_insert_own" on public.recommendation_events;
create policy "recommendation_events_insert_own"
on public.recommendation_events
for insert
with check (auth.uid() = user_id);

