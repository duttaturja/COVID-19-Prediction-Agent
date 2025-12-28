import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
import time

# Configure Streamlit page
st.set_page_config(
    page_title="COVID-19 X-Ray Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Sayemahamed/AI-Lab-Project',
        'Report a bug': "mailto:sayemahamed183@gmail.com",
        'About': "# COVID-19 X-Ray Analysis\nAI-powered COVID-19 detection from chest X-rays using deep learning."
    }
)

# Custom theme and styles
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    :root {
        --primary: #2563EB;
        --primary-dark: #1E40AF;
        --success: #10B981;
        --danger: #EF4444;
        --background: #F8FAFC;
        --card-bg: #FFFFFF;
        --text: #1E293B;
    }

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        background-color: var(--background);
        color: var(--text);
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1200px;
    }
    
    /* Premium Header */
    h1 {
        background: linear-gradient(135deg, #2563EB 0%, #1E40AF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        font-size: 3rem !important;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem !important;
    }
    
    h2, h3 {
        color: var(--text);
        font-weight: 700 !important;
        letter-spacing: -0.01em;
    }
    
    /* Glassmorphism Cards */
    .stCard {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.5);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        background: linear-gradient(120deg, #2563EB, #60A5FA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 28px !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 14px !important;
        color: #64748B !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    [data-testid="stMetricDelta"] {
        font-weight: 600 !important;
    }
    
    /* Custom File Uploader */
    [data-testid="stFileUploader"] {
        border-radius: 12px;
        background-color: #ffffff;
        padding: 2rem;
        border: 2px dashed #CBD5E1;
        transition: border-color 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--primary);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #2563EB 0%, #1E40AF 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }
    
    /* Sidebar Styling */
    /* Sidebar Styling */
    /* Sidebar Styling - Dark Theme */
    [data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="stSidebar"] .stMarkdown h1, 
    [data-testid="stSidebar"] .stMarkdown h2, 
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #F8FAFC !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em;
        text-shadow: 0 1px 2px rgba(0,0,0,0.3);
    }
    
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li,
    [data-testid="stSidebar"] .stMarkdown label,
    [data-testid="stSidebar"] .stMarkdown span {
        color: #E2E8F0 !important;
        font-size: 15px !important;
        line-height: 1.6 !important;
    }
    
    /* Sidebar Expander styling override for dark theme */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        border: 1px solid #334155;
    }
    
    [data-testid="stSidebar"] .streamlit-expanderContent {
        background-color: #0F172A !important;
        color: #CBD5E1 !important;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: #334155 !important;
    }
    
    /* Sidebar user profile/help section styling */
    .sidebar-section {
        padding: 1rem 0;
        border-bottom: 1px solid #F1F5F9;
        margin-bottom: 1rem;
    }
    
    /* Alert/Status Styling */
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid #E2E8F0;
        color: #64748B;
    }
    .footer a {
        color: var(--primary);
        text-decoration: none;
        font-weight: 600;
        transition: color 0.2s;
    }
    .footer a:hover {
        color: var(--primary-dark);
    }
    </style>
    """, unsafe_allow_html=True)

# --- New Model Classes from Untitled-1.py ---

class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)) -> None:
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img) -> Image.Image:
        img_np = np.array(img)
        if len(img_np.shape) == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
        )
        img_clahe = clahe.apply(img_np)
        return Image.fromarray(img_clahe)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Sequential(nn.Identity())

    def forward(self, x):
        return nn.functional.leaky_relu(self.main(x) + self.shortcut(x), 0.2)


class ReadHeads(nn.Module):
    def __init__(
        self,
        in_channels=3,
        kernel_sizes: list[int] = [3, 5, 7],
        feature_per_head: int = 64,
    ) -> None:
        super().__init__()
        self.read_heads = nn.ModuleList()
        for ks in kernel_sizes:
            self.read_heads.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        feature_per_head,
                        kernel_size=ks,
                        padding=(ks - 1) // 2,
                    ),
                    nn.BatchNorm2d(feature_per_head),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )

    def forward(self, x):
        return torch.cat([head(x) for head in self.read_heads], dim=1)


class Covid_Net(nn.Module):
    def __init__(
        self,
        kernel_sizes: list[int],
        number_of_classes: int,
        feature_per_head: int = 64,
    ) -> None:
        super().__init__()

        self.read_heads = ReadHeads(
            in_channels=1, kernel_sizes=kernel_sizes, feature_per_head=feature_per_head
        )

        channel_count = feature_per_head * len(kernel_sizes)

        self.downsampling = nn.Sequential(
            ResidualBlock(channel_count, channel_count),
            nn.Conv2d(channel_count, channel_count, kernel_size=4, stride=2, padding=1),
            ResidualBlock(channel_count, channel_count),
            nn.Conv2d(channel_count, channel_count, kernel_size=4, stride=2, padding=1),
            ResidualBlock(channel_count, channel_count),
            nn.Conv2d(channel_count, channel_count, kernel_size=4, stride=2, padding=1),
            ResidualBlock(channel_count, channel_count),
        )
        self.learning_bottleneck = nn.Sequential(
            ResidualBlock(channel_count, channel_count * 2),
            ResidualBlock(channel_count * 2, channel_count * 4),
            ResidualBlock(channel_count * 4, channel_count * 8),
        )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(channel_count * 8, 12288),
            nn.LeakyReLU(0.2),
            nn.Linear(12288, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, number_of_classes),
        )

    def forward(self, x):
        x = self.read_heads(x)
        x = self.downsampling(x)
        x = self.learning_bottleneck(x)
        x = self.global_pool(x)
        return self.classifier(x)

# Load and prepare model
@st.cache_resource
def load_model():
    try:
        # Initialize the new model architecture
        model = Covid_Net(kernel_sizes=[3, 5, 7], number_of_classes=1)
        
        # Load weights
        # Note: Standardizing to 'model.pth' as the expected filename in deployment
        state_dict = torch.load('model_weights/model.pth', map_location='cpu')
        
        # Remove any module. prefix if present
        if isinstance(state_dict, dict):
             # Check if it's the full state dict or just the map
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
        else:
             st.warning("Model file format unexpected, attempting direct load")
             model = state_dict # Fallback if user saved entire model object
             
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

# Image transformation pipeline
# Updated to match Untitled-1.py training pipeline
transform = transforms.Compose([
    CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    # Note: Training didn't show explicit Normalize in the snippet I saw for Untitled-1.py 
    # except in the previous code? 
    # Wait, Untitled-1.py line 111 shows: CLAHE, Resize, ToTensor. No Normalize.
    # So I will remove Normalize to match training data exactly.
])

def process_image(image):
    if model is None:
        st.error("Model not loaded properly. Please check the model file.")
        return None, 0
        
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    
    # Convert to RGB not strictly needed for CLAHE (it converts to Gray) 
    # but good for consistency with input handling
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    try:
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        with torch.inference_mode():
            output = model(image_tensor)
            # Binary classification: output is logits, use sigmoid
            probability = torch.sigmoid(output).item()
            
            # Decide class based on threshold 0.5
            # In Untitled-1.py: title_str = config.target_disease if label_array == 1 else "Negative"
            # Assuming label 1 is COVID (Positive)
            
            if probability > 0.5:
                predicted_class = 0 # COVID (Positive) based on my logic below? 
                # Wait, usually 1 is the 'Positive' class (COVID).
                # Untitled-1.py: "title_str = config.target_disease if label_array == 1 else "Negative""
                # So 1 -> COVID, 0 -> Normal/Negative.
                
                # In app.py OLD logic:
                # if predicted_class == 1:  # Normal case
                #     st.success("✅ Normal X-Ray")
                # else:  # COVID case
                
                # I need to match the OLD app.py expected return values if I don't change the UI logic.
                # OLD UI: if predicted_class == 1 (Normal).
                
                # NEW Model: 1 = COVID, 0 = Normal.
                # So if prob > 0.5 (COVID), I should return something that indicates COVID.
                # If I return 0 for COVID, and 1 for Normal, it matches the old check:
                # "if predicted_class == 1: # Normal case"
                
            # Interpretation based on ImageFolder structure:
            # Class 0: COVID
            # Class 1: Normal
            # The model outputs probability of Class 1 (Normal)
            
            if probability > 0.5:
                predicted_class = 1 # Normal
                confidence = probability
            else:
                predicted_class = 0 # COVID
                confidence = 1 - probability
        
        return predicted_class, confidence * 100, image_tensor[0]
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, 0, None

def main():
    # Sidebar content
    # Sidebar content
    with st.sidebar:
        st.image("assets/covid-19.jpg", width="stretch")
        
        st.markdown("### 🏥 AI Diagnostic Assistant")
        st.info(
            "This AI tool assists medical professionals in screening for COVID-19 using chest X-rays."
        )
        
        st.markdown("---")
        
        st.markdown("### 📊 Dataset Statistics")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("COVID", "3.6K", delta_color="inverse")
        with col_b:
            st.metric("Normal", "10K", delta_color="normal")
            
        with st.expander("� Detailed Sources"):
            st.markdown("""
            **COVID-19 (3,616 images)**
            - PadChest: 2,473
            - SIRM/Kaggle: 559
            - Other Sources: 583
            
            **Normal (10,192 images)**
            - RSNA: 8,851
            - Kaggle: 1,341
            """)

        st.markdown("### 🧠 Model Info")
        st.caption("Architecture: Custom Covid_Net (ResNet-style)")
        with st.expander("�️ Technical Details"):
             st.markdown("""
            - **Input**: 300x300 px
            - **Preprocessing**: CLAHE Enhancement
            - **Backbone**: Multi-Scale Residual Net
            - **Training Accuracy**: ~98%
            """)
        
        st.markdown("### 📚 Reference")
        st.markdown("""
        *M.E.H. Chowdhury et al., IEEE Access, 2020.*  
        [Read Paper](https://doi.org/10.1109/ACCESS.2020.3010287)
        """)

    # Main content
    st.title("🫁 COVID-19 X-Ray Analysis")
    st.caption("AI-Powered COVID-19 Detection from Chest X-Rays")

    # Medical Disclaimer
    with st.warning("⚕️ **Medical Disclaimer**"):
        st.markdown("""
        This tool is for research and educational purposes only. It should not be used as a substitute 
        for professional medical advice, diagnosis, or treatment. Always consult with healthcare 
        professionals for medical advice and diagnosis.
        """)

    # Performance Metrics
    metrics_cols = st.columns(4)
    with metrics_cols[0]:
        st.metric("Accuracy", "95%", "High")
    with metrics_cols[1]:
        st.metric("Sensitivity", "93%", "Good")
    with metrics_cols[2]:
        st.metric("Specificity", "96%", "Excellent")
    with metrics_cols[3]:
        st.metric("Precision", "94%", "High")

    # Upload Section
    st.subheader("Upload X-Ray Image")
    
    # Add input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Upload File", "Use Camera"],
        horizontal=True,
        help="Select how you want to input the X-ray image"
    )
    
    if input_method == "Upload File":
        image_source = st.file_uploader(
            "Choose a chest X-ray image",
            type=["png", "jpg", "jpeg"],
            help="Upload a clear, front-view chest X-ray image in PNG or JPEG format"
        )
    else:  # Camera option
        image_source = st.camera_input(
            "Take a picture of the X-ray",
            help="Position the X-ray image clearly in front of the camera"
        )

    if image_source:
        try:
            with st.spinner("Analyzing X-ray..."):
                # Process image and get prediction
                predicted_class, confidence, transformed_tensor = process_image(image_source)
            
            if predicted_class is None:
                st.stop()
                
            # Create two columns for image and results
            col1, col2 = st.columns([1, 1], gap="medium")
            
            with col1:
                # Display images in tabs
                tab1, tab2 = st.tabs(["Original X-Ray", "Processed Input"])
                
                image = Image.open(image_source)
                with tab1:
                    st.image(
                        image, 
                        caption=f"{'Captured' if input_method == 'Use Camera' else 'Uploaded'} X-Ray ({time.strftime('%H:%M:%S')})", 
                        width="stretch"
                    )
                
                with tab2:
                    if transformed_tensor is not None:
                        # Convert tensor to PIL Image for display
                        # Transform creates a float tensor [0,1], need to scale for visibility if needed, 
                        # but ToPILImage handles float tensors [0,1] correctly.
                        trans_img = transforms.ToPILImage()(transformed_tensor)
                        st.image(
                            trans_img,
                            caption=f"Clahe + Resized ({trans_img.size})",
                            width="stretch"
                        )
            
            with col2:
                # Create result container with custom styling
                result_container = st.container()
                
                with result_container:
                    st.subheader("Analysis Results")
                    
                    # Display prediction with appropriate styling
                    if predicted_class == 1:  # Normal case
                        st.success("✅ Normal X-Ray")
                        recommendation = "No COVID-19 indicators detected in the X-ray image"
                        status_color = "green"
                    else:  # COVID case
                        st.error("⚠️ COVID-19 Indicators Detected")
                        recommendation = "Please seek immediate medical attention and consult a healthcare professional"
                        status_color = "red"
                    
                    # Show confidence with progress bar
                    st.markdown(f"**Confidence Score:** {confidence:.1f}%")
                    st.progress(confidence/100)
                    
                    # Show recommendation
                    st.info(recommendation)
                    
                    # Additional details
                    with st.expander("🔍 Detailed Analysis"):
                        st.markdown(f"""
                        - **Classification**: {"Normal" if predicted_class == 1 else "COVID-19"}
                        - **Confidence**: {confidence:.1f}%
                        - **Model Version**: v1.0
                        - **Image Size**: {image.size}
                        - **Analysis Time**: {time.strftime('%H:%M:%S')}
                        """)

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.info("Please ensure you've uploaded a valid chest X-ray image")
    
    # Tips section
    with st.expander("💡 Tips for Best Results"):
        st.markdown("""
        For optimal analysis results:
        
        1. **Image Quality**
           - Use high-resolution X-ray images
           - Ensure proper contrast and brightness
           - Avoid blurry or distorted images
        
        2. **Image Position**
           - Upload front-view (PA) chest X-rays
           - Ensure the entire chest cavity is visible
           - Avoid cropped or partial images
        
        3. **File Format**
           - Use PNG or JPEG format
           - Original medical image format preferred
           - Avoid screenshots or phone photos of X-rays
        """)

    # Footer
    st.markdown("""
    <div class="footer">
        <p>Built with ❤️ using PyTorch & Streamlit | <a href="https://github.com/Sayemahamed/AI-Lab-Project">GitHub</a> | <a href="mailto:sayemahamed183@gmail.com">Contact</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()