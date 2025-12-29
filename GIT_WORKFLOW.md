# Git Workflow Guide - Horse Health Analysis

This guide covers all Git operations for this project, including submodule management.

## Table of Contents
- [Initial Setup](#initial-setup)
- [Daily Workflow](#daily-workflow)
- [Submodule Management](#submodule-management)
- [Branching Strategy](#branching-strategy)
- [Common Issues](#common-issues)

---

## Initial Setup

### First Time Clone (For New Team Members)

```bash
# Clone the main repository
git clone https://github.com/Kshitijvyas-chicmic/horse_health_analysis.git
cd horse_health_analysis

# Initialize and update submodules
git submodule update --init --recursive

# Verify submodule is loaded
cd mmpose
git status  # Should show "On branch main"
cd ..
```

### Configure Git Identity

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

---

## Daily Workflow

### 1. Before Starting Work

```bash
# Pull latest changes from main repo
git pull origin main

# Update submodules to latest committed version
git submodule update --remote --merge
```

### 2. Making Changes

#### Changes in Main Repo (Outside mmpose/)

```bash
# Check what changed
git status

# Add specific files
git add README.md fix_bbox_from_keypoints.py

# Or add all changes (use carefully!)
git add .

# Commit with descriptive message
git commit -m "Add bbox expansion script and update README"

# Push to remote
git push origin main
```

#### Changes in MMPose Submodule

```bash
# Navigate to submodule
cd mmpose

# Check status
git status

# Add your custom files
git add custom_configs/rtmpose_hoof_4kp.py
git add demo/debug_visualizer.py
git add demo/inference_on_new_image.py

# Commit in submodule
git commit -m "Update custom configs and demo scripts"

# Go back to main repo
cd ..

# The main repo now sees mmpose has new commits
git status  # Will show "modified: mmpose (new commits)"

# Add the submodule reference update
git add mmpose

# Commit the submodule pointer update
git commit -m "Update mmpose submodule with latest custom configs"

# Push both
git push origin main
```

### 3. After Making Changes

```bash
# Always check status before pushing
git status

# Review changes
git diff

# Push to remote
git push origin main
```

---

## Submodule Management

### Understanding Submodules

The `mmpose/` directory is a **submodule** - a separate Git repository inside your main repo. It has its own commit history.

### Update Submodule to Latest Upstream (MMPose Official Repo)

```bash
cd mmpose

# Add upstream remote (first time only)
git remote add upstream https://github.com/open-mmlab/mmpose.git

# Fetch latest from official MMPose
git fetch upstream

# Merge upstream changes (be careful - may conflict with custom changes!)
git merge upstream/main

# Go back to main repo and update the submodule pointer
cd ..
git add mmpose
git commit -m "Update mmpose to latest upstream version"
git push origin main
```

### View Submodule Status

```bash
# From main repo
git submodule status

# Detailed submodule info
git submodule summary
```

### Reset Submodule to Committed Version

If you accidentally modified the submodule:

```bash
# Reset submodule to the version committed in main repo
git submodule update --init --recursive
```

---

## Branching Strategy

### Create a Feature Branch

```bash
# Create and switch to new branch
git checkout -b feature/add-new-augmentation

# Make changes and commit
git add custom_configs/rtmpose_hoof_4kp.py
git commit -m "Add rotation augmentation to training pipeline"

# Push branch to remote
git push -u origin feature/add-new-augmentation
```

### Merge Feature Branch

```bash
# Switch to main
git checkout main

# Pull latest
git pull origin main

# Merge feature branch
git merge feature/add-new-augmentation

# Push merged changes
git push origin main

# Delete feature branch (optional)
git branch -d feature/add-new-augmentation
git push origin --delete feature/add-new-augmentation
```

### Work on Submodule Branch

```bash
cd mmpose

# Create branch in submodule
git checkout -b custom/hoof-detection

# Make changes
git add custom_configs/
git commit -m "Add custom hoof detection configs"

# Push submodule branch
git push origin custom/hoof-detection

# Go back to main repo
cd ..

# Update main repo to track this submodule branch
git add mmpose
git commit -m "Update mmpose submodule to custom/hoof-detection branch"
git push origin main
```

---

## Common Issues

### Issue 1: "Permission Denied" When Pushing

**Problem:** Git is using cached credentials for wrong user.

**Solution:**
```bash
# Clear cached credentials
git credential reject <<EOF
protocol=https
host=github.com
EOF

# Try push again - it will ask for credentials
git push origin main
```

### Issue 2: Submodule Shows as "Modified" But You Didn't Change It

**Problem:** Submodule HEAD is detached or at different commit.

**Solution:**
```bash
# Reset submodule to committed version
git submodule update --init --recursive

# Or if you want to keep changes:
cd mmpose
git checkout main  # Or your branch name
cd ..
git add mmpose
git commit -m "Update submodule pointer"
```

### Issue 3: "fatal: Not a git repository" in mmpose/

**Problem:** Submodule not initialized.

**Solution:**
```bash
git submodule update --init --recursive
```

### Issue 4: Merge Conflicts in Submodule

**Problem:** Both you and upstream modified same files in mmpose.

**Solution:**
```bash
cd mmpose

# See conflicted files
git status

# Resolve conflicts manually in each file
# Then:
git add <resolved-files>
git commit -m "Resolve merge conflicts"

cd ..
git add mmpose
git commit -m "Update submodule after resolving conflicts"
```

### Issue 5: Accidentally Committed Large Files (work_dirs/, *.pth)

**Problem:** Training outputs or model checkpoints in Git.

**Solution:**
```bash
# Remove from Git but keep locally
git rm --cached -r mmpose/work_dirs/
git rm --cached mmpose/*.pth

# Add to .gitignore
echo "work_dirs/" >> mmpose/.gitignore
echo "*.pth" >> mmpose/.gitignore

# Commit the removal
git add .gitignore
git commit -m "Remove large training files from Git"
git push origin main
```

---

## Best Practices

### ✅ DO:
- Commit often with clear messages
- Pull before pushing
- Use `.gitignore` for large files (data/, work_dirs/, *.pth)
- Keep main repo and submodule commits separate
- Test code before pushing

### ❌ DON'T:
- Use `git add .` without checking `git status` first
- Commit sensitive data (API keys, passwords)
- Commit large binary files (model checkpoints, datasets)
- Force push (`git push -f`) unless absolutely necessary
- Modify core MMPose files (except documented patches)

---

## Quick Reference

```bash
# Check status
git status

# View changes
git diff

# View commit history
git log --oneline -10

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Discard all local changes (DANGEROUS!)
git reset --hard HEAD

# View remote URL
git remote -v

# Change remote URL
git remote set-url origin <new-url>

# Submodule status
git submodule status

# Update all submodules
git submodule update --remote --merge
```

---

## Important Files to Track

### Main Repo
- ✅ `README.md`, `SETUP_DOC.md`, `GIT_WORKFLOW.md`
- ✅ `fix_bbox_from_keypoints.py`, `test_pipeline.py`
- ✅ `src/` directory (all source code)
- ❌ `data/` (images - too large)
- ❌ `env/`, `env_mmpose/` (virtual environments)

### MMPose Submodule
- ✅ `custom_configs/rtmpose_hoof_4kp.py`
- ✅ `demo/debug_visualizer.py`, `demo/inference_on_new_image.py`
- ✅ Patches to `mmpose/datasets/transforms/common_transforms.py` (documented)
- ❌ `work_dirs/` (training outputs)
- ❌ `*.pth` (model checkpoints)

---

## Support

For Git issues, check:
- GitHub Docs: https://docs.github.com/en/get-started/using-git
- Git Submodules: https://git-scm.com/book/en/v2/Git-Tools-Submodules
- This project's README: `README.md`
