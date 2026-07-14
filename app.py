import streamlit as st
import torch
import torch.nn as nn

from torchvision import transforms

from PIL import Image

import numpy as np
import cv2
import os
import gdown

import io

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="BeautyGAN Makeup Transfer",
    page_icon="💄",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>

.main {
    background-color: #0b1020;
}

[data-testid="stSidebar"] {
    background-color: #f8f4f8;
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: #1f2937 !important;
}

/* Headers */
h1 {
    color: #ff2d7a;
}

/* Cards */
.result-card {

    background-color: white;

    padding: 20px;

    border-radius: 15px;

    box-shadow: 0 4px 12px rgba(0,0,0,0.15);

    color: #111827;
}

/* Buttons */

.stButton > button {

    background: linear-gradient(
        90deg,
        #ff2d7a,
        #ff5ca8
    );

    color: white;

    border-radius: 12px;

    height: 50px;

    font-weight: bold;

    border: none;
}

/* Metric cards */

[data-testid="metric-container"] {

    background-color: #111827;

    border-radius: 12px;

    padding: 10px;

    border: 1px solid #374151;
}

</style>
""", unsafe_allow_html=True)
# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cpu")

# ============================================================
# RESIDUAL BLOCK
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.InstanceNorm2d(
                channels
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.InstanceNorm2d(
                channels
            )
        )

    def forward(self, x):

        return x + self.block(x)

# ============================================================
# GENERATOR
# ============================================================

class Generator(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(
                6,
                64,
                kernel_size=7,
                stride=1,
                padding=3
            ),

            nn.InstanceNorm2d(64),

            nn.ReLU(True),

            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.InstanceNorm2d(128),

            nn.ReLU(True),

            nn.Conv2d(
                128,
                256,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.InstanceNorm2d(256),

            nn.ReLU(True)
        )

        self.residuals = nn.Sequential(

            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256)
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.InstanceNorm2d(128),

            nn.ReLU(True),

            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.InstanceNorm2d(64),

            nn.ReLU(True),

            nn.Conv2d(
                64,
                3,
                kernel_size=7,
                stride=1,
                padding=3
            ),

            nn.Tanh()
        )

    def forward(
        self,
        makeup_img,
        non_makeup_img
    ):

        x = torch.cat(
            [
                makeup_img,
                non_makeup_img
            ],
            dim=1
        )

        x = self.encoder(x)

        x = self.residuals(x)

        x = self.decoder(x)

        return x

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():

    MODEL_PATH = "final_generator.pth"

    if not os.path.exists(MODEL_PATH):
        url = "https://drive.google.com/uc?id=1rU9SR5Cqq_Co790S9mmTrND_3h64Aqfx"
        gdown.download(url, MODEL_PATH, quiet=False)

    model = Generator()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.eval()
    return model
# ============================================================
# IMAGE TRANSFORMS
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (256,256)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.5,0.5,0.5],
        [0.5,0.5,0.5]
    )
])

# ============================================================
# TENSOR TO IMAGE
# ============================================================

def tensor_to_pil(tensor):

    tensor = tensor.squeeze(0)

    tensor = tensor * 0.5 + 0.5

    tensor = tensor.clamp(0,1)

    image = transforms.ToPILImage()(
        tensor
    )

    return image
# ============================================================
# LIGHT IMAGE ENHANCEMENT
# ============================================================

def enhance_image(img):

    img = np.array(img)

    # Mild denoising
    img = cv2.fastNlMeansDenoisingColored(
        img,
        None,
        3,
        3,
        7,
        21
    )

    # Mild sharpening
    kernel = np.array([
        [0, -1, 0],
        [-1, 5.2, -1],
        [0, -1, 0]
    ])

    img = cv2.filter2D(
        img,
        -1,
        kernel
    )

    # Slight contrast improvement
    img = cv2.convertScaleAbs(
        img,
        alpha=1.08,
        beta=3
    )

    return Image.fromarray(img)
# ============================================================
# BEAUTYGAN INFERENCE
# ============================================================

def run_beautygan(
    target_img,
    reference_img
):

    target_tensor = transform(
        target_img.convert("RGB")
    ).unsqueeze(0)

    reference_tensor = transform(
        reference_img.convert("RGB")
    ).unsqueeze(0)

    with torch.no_grad():

        generated = generator(
            reference_tensor,
            target_tensor
        )

    result = tensor_to_pil(
        generated
    )

    return result
# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("💄 BeautyGAN")

    st.success("✅ Model Loaded")

    st.markdown("---")

    with st.expander("📖 About Project"):

        st.write("""
        BeautyGAN-inspired Makeup Transfer System.

        Transfers makeup style from a reference face
        to a target face using Deep Learning.
        """)

    with st.expander("📊 Dataset"):

        st.write("""
        Makeup Transfer Own Dataset

        • Makeup Images

        • Non-Makeup Images

        • Segmentation Masks
        """)

    with st.expander("⚙️ Training"):

        st.write("""
        Epochs: 100

        Batch Size: 8

        Learning Rate: 0.0002

        Image Size: 256×256
        """)

    with st.expander("🔄 Workflow"):

        st.write("""
        1. Dataset Collection

        2. Data Preprocessing

        3. Model Selection

        4. Model Training

        5. Makeup Transfer

        6. Evaluation

        7. Deployment
        """)

    with st.expander("📈 Evaluation"):

        st.write("""
        Metrics Used

        • SSIM

        • PSNR
        """)

    with st.expander("🧠 Model"):

        st.write("""
        BeautyGAN Inspired GAN

        Generator Network

        Residual Blocks

        Identity Preservation

        Style Transfer
        """)
# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div style='text-align:center'>

<h1 style='color:#e91e63;'>

💄 BeautyGAN Makeup Transfer System

</h1>

<h4>

Deep Learning Based Facial Makeup Style Transfer

</h4>

</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# DASHBOARD METRICS
# ============================================================

m1, m2, m3, m4 = st.columns(4)

with m1:

    st.metric(
        "Epochs",
        "100"
    )

with m2:

    st.metric(
        "Image Size",
        "256x256"
    )

with m3:

    st.metric(
        "Model",
        "BeautyGAN"
    )

with m4:

    st.metric(
        "Device",
        "CPU"
    )

st.markdown("---")

# ============================================================
# PROJECT OVERVIEW CARD
# ============================================================

st.markdown("""
<div class='result-card'>

<h3>
🎯 Project Objective
</h3>

<p>
Transfer makeup style from a reference image
to a target face while preserving the
identity of the target person.
</p>

</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
with st.expander("🔄 System Workflow"):

    st.markdown("""
    Dataset Collection

            ↓

    Data Preprocessing

            ↓

    BeautyGAN Training

            ↓

    Makeup Transfer

            ↓

    Evaluation

            ↓

    Streamlit Deployment
    """)

# ============================================================
# IMAGE UPLOAD SECTION
# ============================================================
st.info("""
📌 Best Results Tips

• Use frontal face images

• Use good lighting

• Avoid sunglasses

• Use high resolution images

• Use one makeup reference image
""")
st.subheader(
    "📤 Upload Images"
)

left_col, right_col = st.columns(2)

with left_col:

    reference_file = st.file_uploader(
        "💄 Upload Makeup Reference Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

with right_col:

    target_file = st.file_uploader(
        "🙂 Upload Target Face Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

st.markdown("---")
# ============================================================
# DISPLAY INPUT IMAGES
# ============================================================

if reference_file and target_file:

    reference_img = Image.open(
        reference_file
    ).convert("RGB")

    target_img = Image.open(
        target_file
    ).convert("RGB")

    st.subheader(
        "🖼 Uploaded Images"
    )

    c1, c2 = st.columns(2)

    with c1:

        st.image(
            reference_img,
            caption="Reference Makeup",
            use_container_width=True
        )

    with c2:

        st.image(
            target_img,
            caption="Target Face",
            use_container_width=True
        )

    st.markdown("---")

    # ========================================================
    # TRANSFER BUTTON
    # ========================================================

    if st.button(
        "✨ Transfer Makeup"
    ):

        with st.spinner(
            "Generating Makeup Transfer..."
        ):

            generated_img = run_beautygan(
                target_img,
                reference_img
            )

        st.success(
            "Makeup Transfer Completed Successfully!"
        )

        st.balloons()

        st.markdown("---")

        # ====================================================
        # RESULTS
        # ====================================================

        st.subheader(
            "🎯 Makeup Transfer Results"
        )

        r1, r2, r3 = st.columns(3)

        with r1:

            st.image(
                reference_img,
                caption="💄 Reference Makeup",
                use_container_width=True
            )

        with r2:

            st.image(
                target_img,
                caption="🙂 Target Face",
                use_container_width=True
            )

        with r3:

            st.image(
                generated_img,
                caption="✨ Generated Result",
                use_container_width=True
            )

        st.markdown("---")

        # ====================================================
        # RESULT METRICS
        # ====================================================

        st.subheader(
            "📊 Transfer Summary"
        )

        m1, m2, m3 = st.columns(3)

        with m1:

            st.metric(
                "Model",
                "BeautyGAN"
            )

        with m2:

            st.metric(
                "Image Size",
                "256×256"
            )

        with m3:

            st.metric(
                "Status",
                "Success"
            )

        st.markdown("---")

        # ====================================================
        # DOWNLOAD BUTTON
        # ====================================================

        img_buffer = io.BytesIO()

        generated_img.save(
            img_buffer,
            format="PNG"
        )

        st.download_button(

            label="⬇ Download Result",

            data=img_buffer.getvalue(),

            file_name="makeup_transfer_result.png",

            mime="image/png"
        )

        st.markdown("---")

        # ====================================================
        # SIDE BY SIDE COMPARISON
        # ====================================================

        st.subheader(
            "🔍 Before vs After"
        )

        compare1, compare2 = st.columns(2)

        with compare1:

            st.image(
                target_img,
                caption="Before Makeup Transfer",
                use_container_width=True
            )

        with compare2:

            st.image(
                generated_img,
                caption="After Makeup Transfer",
                use_container_width=True
            )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

with st.expander(
    "📚 More About BeautyGAN"
):

    st.write("""
BeautyGAN is a Generative Adversarial Network
designed for facial makeup transfer.

The network learns how to transfer makeup
styles from a reference image while preserving
the identity of the target face.
""")

with st.expander(
    "📄 Project Information"
):

    st.write("""
Project Title:

Design and Implementation of a Make-up Transfer
System using Deep Learning Techniques

Technology Stack:

• Python

• PyTorch

• Streamlit

• OpenCV

• Pillow

• BeautyGAN-Inspired GAN

Dataset:

• Makeup Transfer Own Dataset

Training:

• 100 Epochs

• Adam Optimizer

• Learning Rate = 0.0002

Evaluation:

• SSIM

• PSNR
""")

with st.expander(
    "👨‍💻 Developer"
):

    st.write("""
Developed as a Deep Learning Final Year Project.

BeautyGAN-inspired Makeup Transfer System
with Streamlit Deployment.
""")
st.markdown("""
---

<center>

### 💄 BeautyGAN Makeup Transfer System

Deep Learning Based Facial Makeup Transfer

Final Year Project

Built with PyTorch + Streamlit

</center>

""", unsafe_allow_html=True)
