# Environment Setup Instructions

These steps recreate the `ben-splat-env` environment used for development.

---

## 🧰 Create the Environment

From the project root (e.g. `~/encode/code`):

```bash
# Create the environment (one-time)
uv venv ben-splat-env

# Activate it
source ben-splat-env/bin/activate
```

---

## ✅ Verify

Check you’re using the right interpreter:

```bash
which python
# should show: ~/encode/code/ben-splat-env/bin/python
```

---

## 📦 Install Dependencies

To install all required packages (creates a lockfile for reproducibility):

```bash
# Add core packages
uv add torch torchvision torchaudio pycolmap open3d numpy wildflow
```

This will automatically update both `pyproject.toml` and `uv.lock`.

If you only want to install what’s already listed:

```bash
uv sync
```

---

## ➕ Add New Packages Later

To add more packages later:

```bash
uv add <package-name>
```

Example:

```bash
uv add matplotlib seaborn
```

This updates both the TOML and lock files automatically.

---

## 🔄 Update Existing Packages

To update everything to the latest compatible versions:

```bash
uv lock --upgrade
uv sync
```

Or to upgrade just one package:

```bash
uv add --upgrade <package-name>
```

---

## 🚪 Exit the Environment

To deactivate:

```bash
deactivate
```

To reactivate later:

```bash
source ben-splat-env/bin/activate
```

---

## 🧹 Optional Clean-up

If you ever want to remove and recreate the environment:

```bash
deactivate  # if active
rm -rf ben-splat-env
uv venv ben-splat-env
source ben-splat-env/bin/activate
uv sync
```

---

## 💡 Notes

- Do **not** activate both Conda and this venv at once.
- `uv` automatically manages dependencies and lockfiles.
- Keep `pyproject.toml` and `uv.lock` under version control.
- You can also run scripts directly without activating using:
  ```bash
  uv run python your_script.py
  ```