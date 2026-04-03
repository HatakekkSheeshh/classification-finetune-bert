from pathlib import Path
import re

import streamlit as st
import torch
import wandb
from transformers import BertTokenizer, DistilBertForSequenceClassification

wandb.login(key=st.secrets["WANDB_API_KEY"])
ARTIFACT_NAME = (
    "nguyenquochieujff7-ho-chi-minh-city-university-of-technology/"
    "bert-models/DistilBERT_Full:v0"
)
BASE_MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 20
MAX_LENGTH = 512
MAX_WORDS = 400
LOCAL_MODEL_DIR = Path("artifacts") / "DistilBERT_Full-v0"

CLASS_NAMES = [
    "alt.atheism",
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "misc.forsale",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
    "soc.religion.christian",
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
    "talk.religion.misc",
]

CLASS_DISPLAY_NAMES = {
    "alt.atheism": "Atheism",
    "comp.graphics": "Computer Graphics",
    "comp.os.ms-windows.misc": "MS Windows",
    "comp.sys.ibm.pc.hardware": "IBM PC Hardware",
    "comp.sys.mac.hardware": "Mac Hardware",
    "comp.windows.x": "X Window System",
    "misc.forsale": "For Sale",
    "rec.autos": "Autos",
    "rec.motorcycles": "Motorcycles",
    "rec.sport.baseball": "Baseball",
    "rec.sport.hockey": "Hockey",
    "sci.crypt": "Cryptography",
    "sci.electronics": "Electronics",
    "sci.med": "Medicine",
    "sci.space": "Space",
    "soc.religion.christian": "Christianity",
    "talk.politics.guns": "Politics - Guns",
    "talk.politics.mideast": "Politics - Middle East",
    "talk.politics.misc": "Politics - Misc",
    "talk.religion.misc": "Religion - Misc",
}


def truncate_text(text: str, max_words: int = MAX_WORDS) -> str:
    words = str(text).split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return str(text)


def clean_text(text: str) -> str:
    lines = str(text).split("\n")
    cleaned = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(">") or stripped.startswith(":") or stripped.startswith("|"):
            continue

        if any(
            line.startswith(header)
            for header in [
                "From:",
                "Subject:",
                "Organization:",
                "Lines:",
                "Reply-To:",
                "NNTP-Posting-Host:",
            ]
        ):
            continue

        if stripped == "--":
            break

        cleaned.append(line)

    cleaned_text = " ".join(cleaned).strip()
    cleaned_text = re.sub(r"http\S+|ftp\S+|www\.\S+", "", cleaned_text)
    cleaned_text = re.sub(r"\S+@\S+", "", cleaned_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    cleaned_text = re.sub(r"\[.*?deletia.*?\]", "", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\[.*?snip.*?\]", "", cleaned_text, flags=re.IGNORECASE)
    return cleaned_text


def preprocess_text(text: str) -> str:
    return truncate_text(clean_text(text))


def resolve_checkpoint_path(artifact_dir: Path) -> Path:
    preferred_name = "best_model_distilbert_full_fine_tune.pt"
    direct_preferred = artifact_dir / preferred_name
    if direct_preferred.exists():
        return direct_preferred

    nested_preferred = sorted(artifact_dir.rglob(preferred_name))
    if nested_preferred:
        return nested_preferred[0]

    checkpoints = sorted(artifact_dir.rglob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(
            f"No .pt checkpoint found inside artifact folder: {artifact_dir}"
        )
    return checkpoints[0]


def download_wandb_artifact() -> Path:
    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if any(LOCAL_MODEL_DIR.rglob("*.pt")):
        return LOCAL_MODEL_DIR

    api = wandb.Api()
    artifact = api.artifact(ARTIFACT_NAME, type="model")
    artifact_path = artifact.download(root=str(LOCAL_MODEL_DIR))
    return Path(artifact_path)


@st.cache_resource(show_spinner="Loading DistilBERT model...")
def load_model_and_tokenizer():
    try:
        artifact_dir = download_wandb_artifact()
    except Exception as exc:
        raise RuntimeError(
            "Cannot download the W&B artifact. If the artifact is private, run "
            "`wandb login` in terminal first."
        ) from exc

    checkpoint_path = resolve_checkpoint_path(artifact_dir)
    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=NUM_LABELS,
        seq_classif_dropout=0.1,
    )

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model, tokenizer, checkpoint_path


def predict_text(model, tokenizer, text: str, top_k: int = 5):
    normalized_text = preprocess_text(text)
    encoded = tokenizer(
        normalized_text,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**encoded)
        probs = torch.softmax(outputs.logits, dim=1).squeeze(0)

    k = min(top_k, NUM_LABELS)
    top_probs, top_indices = torch.topk(probs, k=k)
    predictions = [
        {
            "rank": rank + 1,
            "class_id": int(idx),
            "label": CLASS_DISPLAY_NAMES.get(CLASS_NAMES[int(idx)], CLASS_NAMES[int(idx)]),
            "raw_label": CLASS_NAMES[int(idx)],
            "probability": float(prob),
        }
        for rank, (prob, idx) in enumerate(zip(top_probs, top_indices))
    ]
    return normalized_text, predictions


st.set_page_config(page_title="20 Newsgroups DistilBERT Classifier", layout="centered")
st.title("20 Newsgroups Text Classification")
st.caption("DistilBERT_Full fine-tuned checkpoint from W&B artifact")

try:
    model, tokenizer, checkpoint_path = load_model_and_tokenizer()
    st.success(f"Loaded checkpoint: {checkpoint_path}")
except Exception as exc:
    st.error(str(exc))
    st.stop()

sample_text = (
    "NASA recently announced a new space telescope mission focused on observing "
    "distant galaxies and studying exoplanet atmospheres."
)
text_input = st.text_area(
    "Input text to classify",
    value=sample_text,
    height=220,
)
top_k = st.slider("Top-K labels", min_value=1, max_value=10, value=5)

if st.button("Predict", type="primary"):
    if not text_input.strip():
        st.warning("Please enter text before prediction.")
        st.stop()

    cleaned_input, predictions = predict_text(model, tokenizer, text_input, top_k=top_k)
    best = predictions[0]

    st.subheader("Prediction")
    st.metric("Predicted label", best["label"], f"{best['probability']:.2%}")

    st.subheader("Top-K probabilities")
    st.dataframe(predictions, use_container_width=True, hide_index=True)

    with st.expander("Preprocessed text"):
        st.write(cleaned_input)
