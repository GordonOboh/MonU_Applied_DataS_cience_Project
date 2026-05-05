# ⚠️ MASTER BRANCH RULE - READ THIS FIRST

## The Rule

**NEVER interact with the `master` branch under any circumstances.**

This is not a suggestion. This is a hard rule.

## What "Never Interact" Means

Do NOT:
- ❌ `git checkout master`
- ❌ `git checkout main`
- ❌ `git merge master`
- ❌ `git merge main`
- ❌ `git push origin master`
- ❌ `git push origin main`
- ❌ Create pull requests to master
- ❌ Create pull requests from master
- ❌ View master branch
- ❌ Rebase onto master
- ❌ Cherry-pick from master
- ❌ Fetch master
- ❌ Pull master

## What You Should Do Instead

**Always work on the `new` branch.**

```bash
# When starting work
git checkout new

# Before any operation, verify you're on 'new':
git branch

# Output should show:
# * new
#   master
```

## If You Accidentally Checkout Master

If you realize you checked out `master`, immediately return to `new`:

```bash
git checkout new
```

Do NOT make any commits on master. If you did:
- Contact the project lead immediately
- Do NOT push anything to master

## Why This Rule Exists

Master is protected. Changes to master affect the entire project. By restricting all work to `new` branch, we prevent accidental breakage to production or shared state.

## Remember

**`new` branch = your work space**  
**`master` branch = do not touch**

---

When in doubt: `git checkout new`
