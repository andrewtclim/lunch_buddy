-- Add email column to profiles.
alter table public.profiles add column if not exists email text;

-- Auto-create a profiles row when a new user signs up.
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer          -- runs with table-owner privileges so it can insert into profiles
set search_path = ''      -- prevent search_path hijacking
as $$
begin
  insert into public.profiles (user_id, display_name, email)
  values (
    new.id,
    coalesce(new.raw_user_meta_data ->> 'display_name', ''),
    new.email
  );
  return new;
end;
$$;

-- Fire after every insert on auth.users.
drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
after insert on auth.users
for each row
execute function public.handle_new_user();

-- Backfill existing users who don't have a profiles row yet.
insert into public.profiles (user_id, email)
select id, email from auth.users
where id not in (select user_id from public.profiles)
on conflict do nothing;
\