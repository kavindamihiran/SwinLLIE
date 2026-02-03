# 🛠️ Maintenance & Deployment Guide

## ⚡ Quick Update (The Easy Way)

I've created a script `easy_update.ps1` that does everything for you.

**Whenever you change code or fix errors:**
1. Open PowerShell in `SwinIR` folder
2. Run:
   ```powershell
   .\easy_update.ps1
   ```
3. It will:
   - Copy all your latest files to the Space folder
   - Commit them
   - Push them to Hugging Face automatically

---

## 🔙 How to Recover / Rollback

If you break something and want to go back to a previous version:

1. Go to your Space folder:
   ```powershell
   cd ..\swinlle
   ```

2. See history of changes:
   ```powershell
   git log --oneline
   ```
   *You'll see a list like `a1b2c3d Update from script...`*

3. Reset to a previous commit (replace hash with the one you want):
   ```powershell
   git reset --hard a1b2c3d
   git push --force
   ```

---

## 🔑 Authentication

If the script asks for a password, use your **Access Token** (starts with `hf_...`).
