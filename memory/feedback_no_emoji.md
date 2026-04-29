---
name: No emoji in any output
description: User strongly prefers no emojis in code, docs, README, commit messages, or chat replies. Strip them on sight.
type: feedback
---

Do not use emojis anywhere — code, comments, docs (README/CHANGELOG/etc.), commit messages, PR bodies, or chat replies.

**Why:** User stated "no emoji" after I included emoji bullets (🔎 ⚛️ 🗣️) in a README. They flagged it explicitly the first time it happened, suggesting strong preference (not a one-off).

**How to apply:** Default to plain text bullets / headings. If a code style or template you're working from has emojis, strip them. This applies even when emojis would conventionally fit (e.g., Keep-a-Changelog snippets, status badges, "✨ Features" sections). The base CLAUDE.md guidance ("only use emojis if user explicitly requests it") applies — and this user has now reinforced that explicitly for this project.
