---
name: feedback-methodical-pptx-review
description: When building PPTX from a draft .md, extract and compare line-by-line before finalizing
metadata:
  type: feedback
---

When building a PPTX from a content draft (.md), do NOT eyeball the result. Follow this process:

1. Extract full text from every slide using python-pptx into a plain readable format
2. Compare it line-by-line against the draft .md — check every heading, every bullet, every date, every phrase
3. List every discrepancy found before making any changes
4. Fix all discrepancies in one pass
5. Do NOT add anything not in the draft — no "(due July 4)", no extra words, no "helpful" embellishments
6. Extract and verify again after the fix

**Why:** User had to debug me through multiple rounds of missed discrepancies (duplicate headings, missing body text on slide 11, added dates not in draft, wrong section labels, wrong task format). This wasted the user's time and showed lack of care.

**How to apply:** Before reporting "done" on any PPTX build, run the extraction+comparison script and confirm zero discrepancies against the source draft.
