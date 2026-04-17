import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os
import warnings
from datetime import datetime

# --- TẮT CẢNH BÁO ---
warnings.filterwarnings("ignore")

# --- 1. CẤU HÌNH GIAO DIỆN CHUẨN FRONTEND ---
st.set_page_config(
    page_title="AgriDashboard | Nhóm D2T", 
    page_icon="📈", 
    layout="wide", 
    initial_sidebar_state="expanded" # Luôn mở Sidebar khi bắt đầu
)

# Nhúng CSS mô phỏng Tailwind & Glassmorphism
st.markdown("""
<style>
    .stApp { background-color: #020617; color: #F8FAFC; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #0F172A; border-right: 1px solid rgba(255,255,255,0.05); }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 20px; padding: 24px;
        transition: all 0.3s ease; margin-bottom: 20px;
    }
    .glass-card:hover { transform: translateY(-4px); border-color: rgba(56, 189, 248, 0.4); box-shadow: 0 0 25px -5px rgba(56, 189, 248, 0.2); }
    
    .kpi-title { color: #94A3B8; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;}
    .kpi-value { color: #F8FAFC; font-size: 2.2rem; font-weight: 800; margin: 8px 0; font-family: 'Courier New', monospace; }
    
    /* FIX LỖI 1: Hiện lại nút Sidebar nhưng ẩn các thanh màu xám thừa của Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {
        background: rgba(0,0,0,0);
        color: #F8FAFC;
    }
</style>
""", unsafe_allow_html=True)

def render_kpi(title, value, label, color="#38BDF8"):
    st.markdown(f"""
    <div class="glass-card" style="border-top: 3px solid {color};">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{int(value):,}<span style="font-size:1.1rem; color:#64748B; font-family: 'Inter', sans-serif;"> VNĐ/kg</span></div>
        <div style="font-size: 0.85rem; color: #64748B; font-weight: 500;">{label}</div>
    </div>
    """, unsafe_allow_html=True)

# --- 2. ĐỌC VÀ XỬ LÝ DỮ LIỆU ---
@st.cache_data
def load_data(commodity_name):
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        if "Cà phê" in commodity_name:
            file_path = os.path.join(base_path, "coffee_sugar_banana_prices_per_pound.csv")
            df = pd.read_csv(file_path)
            df = df[df['product'] == 'coffee'].copy()
            df['Date'] = pd.to_datetime(df['date'])
            df['Price'] = df['price_in_dollars'] * 26335
        else:
            file_path = os.path.join(base_path, "ho_tieu.csv")
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df['Price'] = pd.to_numeric(df['Price_VND_kg'], errors='coerce')
        
        df = df[df['Date'].dt.year >= 2021] 
        df = df[['Date', 'Price']].dropna().sort_values('Date').set_index('Date')
        monthly_df = df['Price'].resample('ME').mean().dropna()
        tech_df = pd.DataFrame({'Price': monthly_df})
        tech_df['SMA_3'] = tech_df['Price'].rolling(window=3).mean()
        tech_df['EMA_6'] = tech_df['Price'].ewm(span=6, adjust=False).mean()
        return tech_df, False
    except Exception as e:
        return None, str(e)

# --- 3. LÕI AI ---
@st.cache_data
def run_forecast(series, months):
    model = ExponentialSmoothing(series, trend='add', seasonal=None, damped_trend=True, initialization_method="estimated").fit()
    forecast = model.forecast(months)
    forecast.index = pd.date_range(start=series.index[-1] + pd.offsets.MonthEnd(1), periods=months, freq='ME')
    std_err = (series - model.fittedvalues).std()
    expanding_variance = np.sqrt(np.arange(1, months + 1))
    upper = forecast + (1.96 * std_err * expanding_variance)
    lower = forecast - (1.96 * std_err * expanding_variance)
    lower = lower.clip(lower=0) 
    return forecast, upper, lower

# --- 4. GIAO DIỆN WEB APP ---
def main():
    # --- SIDEBAR (SETTINGS) ---
    with st.sidebar:
        st.markdown("<h2 style='color: #F8FAFC; margin-bottom: 0;'>Agri<span style='color: #38BDF8;'>Terminal</span></h2>", unsafe_allow_html=True)
        st.caption("**Nhóm D2T | Developer**")
        st.write("---")
        
        commodity = st.selectbox("🎯 Chọn Nông sản", ["🌶️ Hồ tiêu", "☕ Cà phê"])
        
        st.write("---")
        st.markdown("### ⚙️ Bộ lọc & Chỉ báo")
        show_sma = st.toggle("📈 Hiển thị Đường SMA (3T)", value=True)
        show_ema = st.toggle("📉 Hiển thị Đường EMA (6T)", value=False)
        show_risk = st.toggle("🛡️ Hiển thị Vùng Rủi Ro", value=True)
        
        st.write("---")
        horizon = st.slider("⏱️ Chu kỳ Dự báo (Tháng)", 3, 24, 12)

    # --- MAIN CONTENT ---
    col_header1, col_header2 = st.columns([2, 1])
    with col_header1:
        # FIX LỖI 2: Xử lý tên hiển thị linh hoạt
        if "Cà phê" in commodity:
            asset_name = "cafe" # Chuyển thành cafe viết thường
        else:
            asset_name = ' '.join(commodity.split(' ')[1:]) # Lấy đầy đủ "Hồ tiêu"
            
        st.markdown(f"<h1 style='color: #F8FAFC; margin-bottom: 5px; font-weight: 800;'>Thị trường <span style='color: #38BDF8;'>{asset_name}</span></h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #64748B;'>Dữ liệu đồng bộ thời gian thực & Phân tích định lượng AI.</p>", unsafe_allow_html=True)
    
    with col_header2:
        st.write("")
        st.write("")
        view_range = st.radio("Phạm vi hiển thị:", ["Tất cả", "2 Năm qua", "1 Năm qua"], horizontal=True, label_visibility="collapsed")

    tech_df, err = load_data(commodity)
    if err:
        st.error(f"Lỗi hệ thống: {err}"); st.stop()
        
    series = tech_df['Price']
    forecast, upper, lower = run_forecast(series, horizon)

    plot_df = tech_df.copy()
    if view_range == "1 Năm qua": plot_df = plot_df.iloc[-12:]
    elif view_range == "2 Năm qua": plot_df = plot_df.iloc[-24:]

    # KPI Cards
    latest_val = series.iloc[-1]
    c1, c2, c3 = st.columns(3)
    with c1: render_kpi("Giá Đóng Phiên", latest_val, f"Cập nhật: {series.index[-1].strftime('%m/%Y')}", color="#34D399")
    with c2: render_kpi(f"Mục Tiêu AI (+{horizon}T)", forecast.iloc[-1], "Kịch bản Trọng tâm", color="#38BDF8")
    with c3: render_kpi("Ngưỡng Hỗ Trợ Đáy", lower.iloc[-1], "Ranh giới Rủi ro 95%", color="#F87171")

    # --- BIỂU ĐỒ TƯƠNG TÁC (PLOTLY) ---
    st.markdown("<div style='background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 20px; padding: 20px;'>", unsafe_allow_html=True)
    
    fig = go.Figure()
    last_date, last_val = series.index[-1], series.iloc[-1]
    x_forecast = [last_date] + list(forecast.index)
    y_forecast = [last_val] + list(forecast.values)
    y_upper = [last_val] + list(upper.values)
    y_lower = [last_val] + list(lower.values)

    if show_risk:
        fig.add_trace(go.Scatter(
            x=x_forecast + x_forecast[::-1], y=y_upper + y_lower[::-1],
            fill='toself', fillcolor='rgba(56, 189, 248, 0.12)', line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip", name='Vùng Rủi Ro (95%)'
        ))

    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df['Price'], name="Giá thực tế", mode='lines+markers',
        line=dict(color='#F8FAFC', width=3), marker=dict(size=6, color='#0F172A', line=dict(width=2, color='#F8FAFC')),
        hovertemplate="<b>%{x|%m/%Y}</b><br>Thực tế: %{y:,.0f} VNĐ/kg<extra></extra>"
    ))

    if show_sma:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA_3'], name="SMA (3T)", line=dict(color='#F59E0B', width=2, dash='dot'), hovertemplate="SMA: %{y:,.0f} VNĐ/kg<extra></extra>"))
    if show_ema:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['EMA_6'], name="EMA (6T)", line=dict(color='#A855F7', width=2, dash='dot'), hovertemplate="EMA: %{y:,.0f} VNĐ/kg<extra></extra>"))

    fig.add_trace(go.Scatter(
        x=x_forecast, y=y_forecast, name="Dự báo AI", mode='lines+markers',
        line=dict(color='#38BDF8', width=3, dash='dash'), marker=dict(size=7, color='#0F172A', line=dict(width=2, color='#38BDF8')),
        hovertemplate="<b>%{x|%m/%Y}</b><br>Dự báo: %{y:,.0f} VNĐ/kg<extra></extra>"
    ))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=500, margin=dict(l=0, r=0, t=40, b=0), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)', tickformat="%m/%Y")
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    
    st.plotly_chart(fig, use_container_width=True)
    
    if "Cà phê" in commodity:
        source_text = "Nguồn: Viet Nam - Food Prices | HDX Dataset"
    else:
        source_text = "Nguồn: Hiệp hội Hồ tiêu và cây gia vị Việt Nam (VPSA)"

    st.markdown(f"<p style='text-align: right; font-size: 0.8rem; color: #64748B; margin-top: -10px; font-style: italic;'>{source_text}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
