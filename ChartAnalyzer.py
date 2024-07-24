import streamlit as st
import numpy as np
import easyocr
import openai
import cv2
import torch
import urllib.request
import os
from PIL import Image
from matplotlib import pyplot as plt
import requests
from pathlib import Path
from yolov5.utils.general import non_max_suppression, scale_coords, check_img_size
from yolov5.utils.datasets import letterbox
from yolov5.utils.torch_utils import select_device

st.set_page_config(layout="wide")

# Set up OpenAI API key
openai_api_key_input = st.text_input("Enter OpenAI API Key", type="password")
openai.api_key = openai_api_key_input

if not openai.api_key:
    st.error("OpenAI API key is not set. Please set it as an environment variable 'OPENAI_API_KEY'.")

# Streamlit UI
st.sidebar.title("Dashboard Analyzer")

uploaded_file = st.sidebar.file_uploader("Upload a Screenshot", type=["png", "jpg", "jpeg"])

# Initialize EasyOCR reader
with st.spinner("Downloading OCR model... This may take a few minutes."):
    reader = easyocr.Reader(['en'])

# Define a function to download YOLOv5 model
def download_yolov5_model():
    model_url = "https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt"
    model_path = Path('yolov5s.pt')
    if not model_path.exists():
        with st.spinner("Downloading YOLOv5 model weights..."):
            response = requests.get(model_url, stream=True)
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    return model_path

# Download YOLOv5 model if not already present
model_path = download_yolov5_model()

# Load YOLOv5 model
device = select_device('')
model = torch.load(model_path, map_location=device)['model']
model.to(device).eval()

def detect_visuals(image):
    img = letterbox(image, new_shape=640)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

    visuals = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = map(int, xyxy)
                class_name = model.names[int(cls)]
                if conf > 0.5:  # Confidence threshold
                    visuals.append((image[y1:y2, x1:x2], class_name))
    return visuals

def extract_chart_data(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x_axis_label = ""
    y_axis_label = ""
    data_points = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        
        if aspect_ratio > 2 and h < 50:
            roi = image[y:y+h, x:x+w]
            text = reader.readtext(roi, detail=0)
            if y > image.shape[0] / 2:
                x_axis_label = ' '.join(text)
            else:
                y_axis_label = ' '.join(text)
        elif aspect_ratio < 1.2 and 10 < w < 50 and 10 < h < 50:
            data_points.append((x + w // 2, y + h // 2))

    data_points = sorted(data_points, key=lambda pt: pt[0])

    if data_points:
        x_values = np.linspace(0, 10, len(data_points))
        y_max = max(pt[1] for pt in data_points)
        y_values = [y_max - pt[1] for pt in data_points]

    chart_data = {
        "x_axis_label": x_axis_label,
        "y_axis_label": y_axis_label,
        "x_values": x_values.tolist(),
        "y_values": y_values,
    }
    return chart_data

def analyze_visual(visual, class_name):
    if class_name in ['chart', 'table']:
        chart_data = extract_chart_data(visual)
        insights = generate_insights_from_gpt(chart_data)
        recommendations = generate_recommendations_from_gpt(insights)
        return insights, recommendations
    else:
        return "Unsupported visual type", "No action items available"

def generate_insights_from_gpt(chart_data):
    detailed_prompt = (
        "You are a data analyst. Here is the data extracted from a chart: \n\n"
        f"{chart_data}\n\n"
        "Based on this information, provide a clear and concise summary that explains the key insights from the chart."
    )
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data analyst who provides clear and concise insights from charts."},
                {"role": "user", "content": detailed_prompt}
            ]
        )
        insights = response.choices[0].message['content'].strip()
        return insights
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

def generate_recommendations_from_gpt(insights):
    detailed_prompt = (
        "You are a business consultant. Based on the following insights from a chart, provide actionable recommendations: \n\n"
        f"{insights}\n\n"
        "Provide clear and actionable recommendations based on the insights."
    )
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a business consultant who provides actionable recommendations based on insights."},
                {"role": "user", "content": detailed_prompt}
            ]
        )
        recommendations = response.choices[0].message['content'].strip()
        return recommendations
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

if uploaded_file is not None and openai.api_key:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_np = np.array(image)
    st.image(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB), caption='Uploaded Screenshot', use_column_width=True)

    visuals = detect_visuals(image_np)
    analysis_results = [analyze_visual(visual, class_name) for visual, class_name in visuals]

    st.header("Analysis")

    for i, (insights, recommendations) in enumerate(analysis_results):
        if insights and recommendations:
            st.subheader(f"Insights for Visual {i+1}")
            st.write(insights)
            st.subheader(f"Action Items for Visual {i+1}")
            st.write(recommendations)
        else:
            st.write(f"Error analyzing visual {i+1}")
