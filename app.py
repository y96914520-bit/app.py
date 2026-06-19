import streamlit as st
import numpy as np
import scipy.stats as si
import plotly.graph_objects as go

# ==========================================
# 页面配置 (移动端 & 窄屏优化)
# ==========================================
st.set_page_config(
    page_title="期权多腿策略分析器",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 注入自定义 CSS 以优化移动端间距
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stButton>button { width: 100%; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 核心业务逻辑：BS 模型与 IV 计算
# ==========================================
def bs_price(S, K, T, r, sigma, option_type):
    if T <= 0:
        if option_type == 'Call': return np.maximum(S - K, 0)
        else: return np.maximum(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'Call':
        return S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)

def calculate_iv(target_price, S, K, T, r, option_type):
    intrinsic = max(0.0, S - K * np.exp(-r * T)) if option_type == 'Call' else max(0.0, K * np.exp(-r * T) - S)
    if target_price <= intrinsic: return 0.15 # 默认值
    
    low, high = 1e-4, 5.0
    for _ in range(50):
        mid = (low + high) / 2.0
        p = bs_price(S, K, T, r, mid, option_type)
        if p > target_price: high = mid
        else: low = mid
    return mid

# ==========================================
# 状态管理：管理多腿持仓
# ==========================================
if 'legs' not in st.session_state:
    st.session_state.legs = []

def add_leg():
    st.session_state.legs.append({
        'side': 'Long',
        'type': 'Call',
        'strike': 100.0,
        'mkt_price': 3.0,
        'qty': 1
    })

def remove_leg(index):
    st.session_state.legs.pop(index)

# ==========================================
# UI 布局
# ==========================================
st.title("🧩 自定义期权策略分析")

# 1. 基础环境参数
with st.expander("🌍 市场环境设置", expanded=True):
    col_env1, col_env2 = st.columns(2)
    with col_env1:
        S_curr = st.number_input("标的当前价格", value=100.0, step=1.0)
    with col_env2:
        r_val = st.number_input("无风险利率 (%)", value=3.0) / 100.0
    t_days = st.number_input("距离到期总天数", value=30, step=1)

st.markdown("---")

# 2. 持仓腿管理
st.subheader("🛠️ 策略构成 (Legs)")
if st.button("➕ 添加新期权腿"):
    add_leg()

total_entry_cost = 0.0
processed_legs = []

for i, leg in enumerate(st.session_state.legs):
    with st.container():
        # 为每条腿创建一个卡片式布局
        st.markdown(f"**Leg #{i+1}**")
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            leg['side'] = st.selectbox(f"方向##{i}", ["Long", "Short"], index=0 if leg['side']=='Long' else 1, key=f"side_{i}")
            leg['type'] = st.selectbox(f"类型##{i}", ["Call", "Put"], index=0 if leg['type']=='Call' else 1, key=f"type_{i}")
        with c2:
            leg['strike'] = st.number_input(f"行权价##{i}", value=leg['strike'], key=f"k_{i}")
            leg['mkt_price'] = st.number_input(f"成交价格##{i}", value=leg['mkt_price'], key=f"p_{i}")
        with c3:
            leg['qty'] = st.number_input(f"手数##{i}", value=leg['qty'], min_value=1, key=f"q_{i}")
            if st.button(f"🗑️ 删除##{i}", key=f"del_{i}"):
                remove_leg(i)
                st.rerun()
        
        # 计算该腿的 IV 和成本
        leg_iv = calculate_iv(leg['mkt_price'], S_curr, leg['strike'], t_days/365.0, r_val, leg['type'])
        multiplier = 1 if leg['side'] == 'Long' else -1
        total_entry_cost += leg['mkt_price'] * leg['qty'] * multiplier
        
        processed_legs.append({**leg, 'iv': leg_iv, 'multiplier': multiplier})
        st.markdown("---")

if not processed_legs:
    st.info("请点击上方按钮添加期权腿来构建策略。")
    st.stop()

# 3. 模拟滑块 (情景分析)
st.subheader("📈 模拟预测分析")
col_sim1, col_sim2 = st.columns(2)
with col_sim1:
    days_passed = st.slider("时间经过 (天)", 0, t_days, 0)
with col_sim2:
    iv_move = st.slider("未来 IV 变动 (%)", -50, 50, 0)

# ==========================================
# 计算总盈亏曲线
# ==========================================
s_range = np.linspace(S_curr * 0.7, S_curr * 1.3, 200)
t_remaining = max((t_days - days_passed) / 365.0, 1e-5)

def get_combined_pnl(S_target, is_expiry=False):
    total_val = 0.0
    for leg in processed_legs:
        if is_expiry:
            # 到期价值
            if leg['type'] == 'Call': val = max(S_target - leg['strike'], 0)
            else: val = max(leg['strike'] - S_target, 0)
        else:
            # 动态 BS 价值
            sim_iv = max(leg['iv'] + iv_move/100.0, 1e-4)
            val = bs_price(S_target, leg['strike'], t_remaining, r_val, sim_iv, leg['type'])
        
        total_val += val * leg['qty'] * leg['multiplier']
    return total_val - total_entry_cost

pnl_expiry = [get_combined_pnl(s, True) for s in s_range]
pnl_dynamic = [get_combined_pnl(s, False) for s in s_range]
curr_pnl = get_combined_pnl(S_curr, False)

# ==========================================
# 可视化 (严格执行 Unique Key 防崩溃)
# ==========================================
fig = go.Figure()

fig.add_trace(go.Scatter(x=s_range, y=pnl_expiry, name="到期盈亏", line=dict(dash='dash', color='gray')))
fig.add_trace(go.Scatter(x=s_range, y=pnl_dynamic, name=f"{days_passed}天后盈亏", line=dict(color='#00BFFF', width=3)))

# 标记当前点
fig.add_trace(go.Scatter(
    x=[S_curr], y=[curr_pnl],
    mode='markers+text',
    text=[f"当前: {curr_pnl:.2f}"],
    textposition="top center",
    marker=dict(size=12, color='orange', symbol='diamond')
))

fig.update_layout(
    title="策略合成损益图",
    xaxis_title="标的价格",
    yaxis_title="组合总盈亏",
    template="plotly_white",
    legend=dict(orientation="h", y=1.1),
    margin=dict(l=10, r=10, t=40, b=10),
    hovermode="x unified"
)
fig.add_hline(y=0, line_color="red", opacity=0.3)

# 彻底根除 DOM 渲染 Bug
chart_key = f"multileg_chart_{len(processed_legs)}_{days_passed}_{iv_move}_{S_curr}"
st.plotly_chart(fig, use_container_width=True, key=chart_key)

# 4. 统计摘要
st.sidebar.markdown(f"### 策略摘要")
st.sidebar.write(f"初期总支出/收入: {total_entry_cost:.2f}")
st.sidebar.write(f"当前模拟盈亏: {curr_pnl:.2f}")
