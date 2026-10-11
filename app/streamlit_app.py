"""
PHASE 9 - Streamlit dashboard.

Run from project root:
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils import load_config, get_device                     # noqa: E402
from preprocessing import get_eval_transforms                  # noqa: E402
from models import build_model                                 # noqa: E402
from gradcam import get_target_layer                            # noqa: E402

st.set_page_config(page_title="Acute Leukemia Screening (Research Prototype)", layout="wide")

CONFIG_PATH = "configs/config.yaml"
DEFAULT_CHECKPOINT = "models/baseline_best.pt"


@st.cache_resource
def load_model_and_config(checkpoint_path: str):
    cfg = load_config(CONFIG_PATH)
    device = get_device()
    ckpt = torch.load(checkpoint_path, map_location=device)
    run_name = Path(checkpoint_path).stem.replace("_best", "")
    model_type = "baseline" if "baseline" in run_name else "advanced"
    model = build_model(model_type, cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, device, model_type


def predict_and_explain(model, cfg, device, model_type, pil_img: Image.Image):
    image_size = cfg["data"]["image_size"]
    transforms = get_eval_transforms(image_size)
    rgb_img = np.array(pil_img.resize((image_size, image_size))).astype(np.float32) / 255.0
    tensor_img = transforms(image=np.array(pil_img))["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor_img)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    pred_class = int(np.argmax(probs))
    pred_prob = float(probs[pred_class])
    class_name = cfg["data"]["class_names"][pred_class]

    target_layer = get_target_layer(model, model_type, cfg)
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(
        input_tensor=tensor_img,
        targets=[ClassifierOutputTarget(pred_class)]
    )[0]
    overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return class_name, pred_prob, probs, rgb_img, grayscale_cam, overlay


def main():
    st.title("🔬 AI-Based Acute Leukemia Screening — Research Prototype")
    st.warning(
        "**This is an educational/research prototype only.** It has NOT been clinically "
        "validated and must NOT be used for real diagnosis or patient-care decisions. "
        "Always consult a qualified hematologist/oncologist and confirmatory laboratory tests."
    )

    with st.sidebar:
        st.header("Settings")
        checkpoint_path = st.text_input("Model checkpoint path", value=DEFAULT_CHECKPOINT)
        st.caption("Train a model first with `python src/train_baseline.py` if this file doesn't exist yet.")

    if not Path(checkpoint_path).exists():
        st.error(f"Checkpoint not found at `{checkpoint_path}`. Train the model first (see README PART U).")
        st.stop()

    model, cfg, device, model_type = load_model_and_config(checkpoint_path)
    low_conf_threshold = cfg["evaluation"]["low_confidence_threshold"]

    uploaded_file = st.file_uploader("Upload a peripheral blood smear (single-cell) image",
                                      type=["bmp", "png", "jpg", "jpeg", "tif", "tiff"])

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Uploaded image")
            st.image(pil_img, use_container_width=True)

        with st.spinner("Running preprocessing, inference, and Grad-CAM..."):
            class_name, pred_prob, probs, rgb_img, cam_map, overlay = predict_and_explain(
                model, cfg, device, model_type, pil_img
            )

        with col2:
            st.subheader("Prediction")
            label_display = "🩸 Leukemic blast (ALL) — suspicious" if class_name == "all" else "✅ Normal (Hem)"
            st.markdown(f"### {label_display}")
            st.metric("Confidence (predicted class)", f"{pred_prob:.1%}")

            max_prob = max(probs)
            if max_prob < low_conf_threshold:
                st.error(
                    f"⚠️ LOW CONFIDENCE (max softmax probability = {max_prob:.1%}, "
                    f"below threshold {low_conf_threshold:.0%}). This prediction is flagged "
                    f"for mandatory human expert review."
                )
            else:
                st.info(
                    "Confidence above the review threshold — still requires confirmation by "
                    "a qualified professional; this is not a diagnosis."
                )

            st.write("Class probabilities:")
            for cname, p in zip(cfg["data"]["class_names"], probs):
                st.progress(float(p), text=f"{cname}: {p:.1%}")

        st.subheader("Explainability (Grad-CAM)")
        st.caption(
            "The heatmap shows which regions most influenced the model's prediction. "
            "It does NOT prove the prediction is medically correct — use it as a "
            "debugging/interpretability aid alongside the confidence score above."
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(rgb_img, caption="Original (resized)", use_container_width=True)
        with c2:
            fig, ax = plt.subplots()
            ax.imshow(cam_map, cmap="jet")
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)
            st.caption("Grad-CAM heatmap")
        with c3:
            st.image(overlay, caption="Heatmap overlay", use_container_width=True)
    else:
        st.info("Upload an image to run the pipeline.")

    st.divider()
    st.caption(
        "Project: AI/ML-Based Detection and Recognition of Acute Leukemia from Peripheral "
        "Blood Smear Images. Trained on the C-NMC 2019 dataset (binary ALL vs. Hem classification "
        "only — AML recognition is not supported by this dataset/model). Educational prototype, "
        "not a medical device."
    )


if __name__ == "__main__":
    main()
