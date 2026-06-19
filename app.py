import streamlit as st
import numpy as np
import scipy.stats as si
import plotly.graph_objects as go

# ==========================================
# 页面配置 (移动端优化)
# ==========================================
st.set_page_config(
    page_title="期权 IV 反推与 PnL 分析器",
    layout="centered", # 居中布局更适合手机屏幕
    initial_sidebar_state="collapsed"
)

# ==========================================
# 核心业务逻辑：期权定价与 IV 反推
# ==========================================
def bs_price(S, K, T, r, sigma, option_type):
    """Black-Scholes 期权定价模型"""
    if T <= 0:
        return np.maximum(S - K, 0) if option_type == 'Call' else np.maximum(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'Call':
        return S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)

def calculate_iv_bisection(target_price, S, K, T, r, option_type):
    """使用二分法稳定反推隐含波动率 (Bisection Method)"""
    # 边界与内在价值检查
    intrinsic_value = max(0.0, S - K * np.exp(-r * T)) if option_type == 'Call' else max(0.0, K * np.exp(-r * T) - S)
    if target_price < intrinsic_value:
        return None # 市场价格低于内在价值，无解或存在套利空间
    if T <= 0:
        return None
        
    low, high = 1e-4, 5.0 # IV 搜索范围 0.01% - 500%
    tol = 1e-5
    
    for _ in range(100): # 最大迭代次数 100 次
        mid = (low + high) / 2.0
        price_mid = bs_price(S, K, T, r, mid, option_type)
        
        if abs(price_mid - target_price) < tol:
            return mid
        if price_mid > target_price:
            high = mid
        else:
            low = mid
            
    return (low + high) / 2.0

# ==========================================
# 前端界面构建 (流线型上下结构)
# ==========================================
st.title("📈 期权 IV 与动态盈亏分析")
st.markdown("---")

st.subheader("1. 输入期权市场参数")
col1, col2 = st.columns(2)
with col1:
    option_type = st.selectbox("期权类型", ["Call", "Put"])
    S = st.number_input("标的价格 (S)", value=100.0, step=1.0)
    K = st.number_input("行权价 (K)", value=100.0, step=1.0)
with col2:
    market_price = st.number_input("市场现价", value=3.00, step=0.1)
    T_days = st.number_input("到期天数 (T)", value=30, step=1)
    r_pct = st.number_input("无风险利率 (%)", value=3.0, step=0.1)

T_years = T_days / 365.0
r = r_pct / 100.0

# 计算 IV
iv = calculate_iv_bisection(market_price, S, K, T_years, r, option_type)

if iv is None:
    st.error("⚠️ 无法计算 IV：期权市场价格低于理论内在价值（或已到期）。请检查输入数据！")
    st.stop()

st.success(f"**反推隐含波动率 (IV): {iv * 100:.2f}%**")

st.markdown("---")
st.subheader("2. 情景模拟与 PnL 分析")

# 模拟参数滑块
sim_days_passed = st.slider("时间流逝 (天)", min_value=0, max_value=int(T_days), value=0, step=1)
sim_iv_change_pct = st.slider("IV 变动预测 (绝对百分比 %)", min_value=-50.0, max_value=50.0, value=0.0, step=1.0)
# 虽然直接在图表上标记了 -5%，但也提供自定义的标的价格预测滑块用于灵活性
sim_S_change_pct = st.slider("标的价格预测变动 (%)", min_value=-30.0, max_value=30.0, value=-5.0, step=1.0)

# ==========================================
# 数据计算与可视化
# ==========================================
T_sim_years = max((T_days - sim_days_passed) / 365.0, 1e-5) # 防止时间为0导致除零错误
iv_sim = max(iv + (sim_iv_change_pct / 100.0), 1e-4) # 防止 IV 为负

# 生成 X 轴数据 (标的价格变动区间 -30% 到 +30%)
S_array = np.linspace(S * 0.7, S * 1.3, 200)

# 1. 计算到期结算 PnL (虚线)
if option_type == 'Call':
    pnl_expiry = np.maximum(S_array - K, 0) - market_price
else:
    pnl_expiry = np.maximum(K - S_array, 0) - market_price

# 2. 计算动态持有期 PnL (实线)
prices_dynamic = np.array([bs_price(s, K, T_sim_years, r, iv_sim, option_type) for s in S_array])
pnl_dynamic = prices_dynamic - market_price

# 计算特定预测点的动态 PnL
# (a) 当前价格点
current_pnl = bs_price(S, K, T_sim_years, r, iv_sim, option_type) - market_price
# (b) 指定预测变化点 (默认体现下跌5%的设计要求)
target_S = S * (1 + sim_S_change_pct / 100.0)
target_pnl = bs_price(target_S, K, T_sim_years, r, iv_sim, option_type) - market_price

# 绘制 Plotly 图表
fig = go.Figure()

# 到期 PnL 虚线
fig.add_trace(go.Scatter(
    x=S_array, y=pnl_expiry, 
    mode='lines', name='到期结算 PnL',
    line=dict(dash='dash', color='gray')
))

# 动态 PnL 实线
fig.add_trace(go.Scatter(
    x=S_array, y=pnl_dynamic, 
    mode='lines', name=f'{sim_days_passed}天后动态 PnL',
    line=dict(color='#00BFFF', width=3)
))

# 标记【当前价格点】
fig.add_trace(go.Scatter(
    x=[S], y=[current_pnl],
    mode='markers+text', name='当前标的价格',
    marker=dict(color='orange', size=12, symbol='star'),
    text=[f"S={S:.2f}<br>PnL={current_pnl:.2f}"],
    textposition="top center"
))

# 标记【预测价格点】
fig.add_trace(go.Scatter(
    x=[target_S], y=[target_pnl],
    mode='markers+text', name=f'预测价格 ({sim_S_change_pct}%)',
    marker=dict(color='red', size=10),
    text=[f"S={target_S:.2f}<br>PnL={target_pnl:.2f}"],
    textposition="bottom center"
))

# 图表布局美化，适配移动端窄屏
fig.update_layout(
    title=f"持仓盈亏模拟 (期权成本: {market_price:.2f})",
    xaxis_title="标的价格",
    yaxis_title="盈亏 (PnL)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=50, b=10) # 减少边缘留白，适配手机端
)
fig.add_hline(y=0, line_dash="solid", line_color="red", opacity=0.5)

# 渲染图表：使用动态唯一 Key 彻底解决 React DOM Bug
unique_chart_key = f"pnl_chart_{sim_days_passed}_{sim_iv_change_pct}_{sim_S_change_pct}"
st.plotly_chart(fig, use_container_width=True, key=unique_chart_key)
