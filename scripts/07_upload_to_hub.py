"""Step 7 - upload the trained weights to a Hugging Face Hub model repo so the
app can download them (weights are never committed to git).

Usage:  hf auth login          (once, with a write token)
        python scripts/07_upload_to_hub.py <username>/swe015-chat-moderation
Then put the same repo id into HF_REPO_ID in moderation/config.py.
"""
import sys
from pathlib import Path

from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moderation.config import MODELS_DIR  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    repo_id = sys.argv[1]
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=str(MODELS_DIR), repo_type="model",
                      commit_message="Upload trained moderation models")
    print(f"Uploaded to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
