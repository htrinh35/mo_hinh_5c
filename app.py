"""
Ứng dụng đánh giá rủi ro tín dụng khách hàng vay theo mô hình 5C
Chuyển thể từ notebook: Model_cho_vay_kh_nhóm_nợ.ipynb
Thuật toán: Logistic Regression (phân loại nhị phân: 0 = Không rủi ro, 1 = Có rủi ro)
"""

import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    classification_report,
)
from sklearn.model_selection import train_test_split

# ==========================================================================
# 1) CẤU HÌNH TRANG (phải là lệnh Streamlit đầu tiên)
# ==========================================================================
st.set_page_config(
    layout="wide",
    page_title="Đánh giá Rủi ro Tín dụng 5C",
    page_icon="🏦",
)

# ==========================================================================
# 2) HẰNG SỐ & HÀM DÙNG CHUNG
# ==========================================================================

# Nhóm biến theo mô hình 5C (Tư cách - Năng lực - Điều kiện - Vốn - Tài sản đảm bảo)
FEATURE_GROUPS = {
    "TC — Tư cách (Character)": ["TC1", "TC2", "TC3", "TC4", "TC5"],
    "NL — Năng lực (Capacity)": ["NL1", "NL2", "NL3", "NL4"],
    "DK — Điều kiện (Condition)": ["DK1", "DK2", "DK3", "DK4", "DK5"],
    "V — Vốn (Capital)": ["V1", "V2", "V3", "V4", "V5", "V6"],
    "TS — Tài sản đảm bảo (Collateral)": ["TS1", "TS2", "TS3", "TS4"],
}
FEATURE_COLS = [c for cols in FEATURE_GROUPS.values() for c in cols]
TARGET_COL = "PD"
TARGET_LABELS = {0: "Không rủi ro", 1: "Có rủi ro"}


@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes) -> pd.DataFrame:
    """Nạp dữ liệu từ bytes của file CSV (để hashable cho cache)."""
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = [c.strip() for c in df.columns]
    return df


def validate_columns(df: pd.DataFrame, required_cols: list) -> list:
    """Trả về danh sách cột còn thiếu trong df so với required_cols."""
    return [c for c in required_cols if c not in df.columns]


# ==========================================================================
# 3) SIDEBAR — VÙNG CẤU HÌNH
# ==========================================================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")

    uploaded_file = st.file_uploader(
        "Tải lên tệp dữ liệu khảo sát (.csv)",
        type=["csv"],
        help=(
            "Tệp CSV cần có 24 biến đầu vào theo mô hình 5C "
            "(TC1-5, NL1-4, DK1-5, V1-6, TS1-4) và cột nhãn 'PD' "
            "(0 = không rủi ro, 1 = có rủi ro)."
        ),
    )

    st.subheader("Tham số mô hình AI")

    test_size = st.slider(
        "Tỷ lệ tập kiểm tra (test size)",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
        help="Tỷ lệ dữ liệu dùng để kiểm định mô hình (giống train_test_split trong notebook, mặc định 0.2).",
    )

    random_state = st.number_input(
        "Random state",
        min_value=0,
        max_value=9999,
        value=23,
        step=1,
        help="Hạt giống ngẫu nhiên để tái lập kết quả chia dữ liệu (notebook dùng random_state=23).",
    )

    with st.expander("Tham số nâng cao (Logistic Regression)"):
        C_value = st.number_input(
            "C (nghịch đảo độ mạnh regularization)",
            min_value=0.01,
            max_value=10.0,
            value=1.0,
            step=0.01,
            help="Giá trị càng nhỏ, regularization càng mạnh. Notebook dùng mặc định của scikit-learn (C=1.0).",
        )
        max_iter = st.number_input(
            "Số vòng lặp tối đa (max_iter)",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            help="Số vòng lặp tối đa để thuật toán hội tụ.",
        )
        solver = st.selectbox(
            "Solver",
            options=["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
            index=0,
            help="Thuật toán tối ưu hóa dùng để huấn luyện Logistic Regression (mặc định scikit-learn: lbfgs).",
        )

    st.divider()
    train_clicked = st.button(
        "🚀 Huấn luyện mô hình",
        type="primary",
        use_container_width=True,
    )

# ==========================================================================
# 4) HEADER — VÙNG ĐỊNH HƯỚNG
# ==========================================================================
st.title("🏦 Đánh giá Rủi ro Tín dụng Khách hàng vay — Mô hình 5C")
st.caption(
    "Ứng dụng dự báo khả năng rủi ro (nhóm nợ) của khách hàng vay dựa trên 24 tiêu chí "
    "khảo sát theo mô hình 5C (Tư cách, Năng lực, Điều kiện, Vốn, Tài sản đảm bảo), "
    "sử dụng thuật toán Logistic Regression. Vui lòng tải lên tệp dữ liệu khảo sát (.csv) ở thanh bên trái."
)

if uploaded_file is None:
    st.info("👈 Vui lòng tải lên tệp dữ liệu (.csv) ở thanh bên trái để bắt đầu.")
    st.stop()

file_bytes = uploaded_file.getvalue()
try:
    df = load_data(file_bytes)
except Exception as e:
    st.error(f"Không thể đọc tệp dữ liệu. Lỗi: {e}")
    st.stop()

if df.empty:
    st.error("Tệp dữ liệu rỗng. Vui lòng kiểm tra lại.")
    st.stop()

missing_cols = validate_columns(df, FEATURE_COLS + [TARGET_COL])
if missing_cols:
    st.error(
        "Tệp dữ liệu thiếu các cột bắt buộc: " + ", ".join(missing_cols) +
        ". Vui lòng kiểm tra lại cấu trúc dữ liệu."
    )
    st.stop()

st.caption(f"📁 Đang dùng tệp: **{uploaded_file.name}**")
st.caption(f"📊 Dữ liệu gồm **{df.shape[0]}** dòng và **{df.shape[1]}** cột.")
st.divider()

# ==========================================================================
# 5) KHỐI HUẤN LUYỆN — chạy khi bấm nút, lưu vào session_state
# ==========================================================================
if train_clicked:
    try:
        X = df[FEATURE_COLS]
        y = df[TARGET_COL]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state)
        )

        model = LogisticRegression(C=C_value, max_iter=int(max_iter), solver=solver)
        model.fit(X_train, y_train)

        yhat_test = model.predict(X_test)
        yhat_proba = model.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(y_test, yhat_test)
        acc = accuracy_score(y_test, yhat_test)
        prec = precision_score(y_test, yhat_test, zero_division=0)
        rec = recall_score(y_test, yhat_test, zero_division=0)
        f1 = f1_score(y_test, yhat_test, zero_division=0)
        try:
            auc = roc_auc_score(y_test, yhat_proba)
            fpr, tpr, _ = roc_curve(y_test, yhat_proba)
        except ValueError:
            auc, fpr, tpr = None, None, None
        report = classification_report(y_test, yhat_test, zero_division=0, output_dict=True)

        st.session_state["model"] = model
        st.session_state["feature_cols"] = FEATURE_COLS
        st.session_state["results"] = {
            "cm": cm,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc,
            "fpr": fpr,
            "tpr": tpr,
            "report": report,
            "X_test": X_test,
            "y_test": y_test,
            "yhat_test": yhat_test,
            "yhat_proba": yhat_proba,
        }
        st.session_state["train_params"] = {
            "test_size": test_size,
            "random_state": int(random_state),
            "C": C_value,
            "max_iter": int(max_iter),
            "solver": solver,
        }
        st.success("✅ Huấn luyện mô hình thành công! Xem kết quả ở tab 'Kết quả huấn luyện & kiểm định mô hình'.")
    except Exception as e:
        st.error(f"Đã xảy ra lỗi trong quá trình huấn luyện: {e}")

# ==========================================================================
# 6) CÁC TAB NỘI DUNG CHÍNH
# ==========================================================================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📋 Tổng quan dữ liệu",
        "📈 Trực quan hóa dữ liệu",
        "🎯 Kết quả huấn luyện & kiểm định mô hình",
        "🔮 Sử dụng mô hình",
    ]
)

# --------------------------------------------------------------------
# TAB 1: TỔNG QUAN DỮ LIỆU
# --------------------------------------------------------------------
with tab1:
    st.subheader("Tổng quan dữ liệu")

    col1, col2, col3 = st.columns(3)
    col1.metric("Số dòng", f"{df.shape[0]:,}")
    col2.metric("Số cột", f"{df.shape[1]:,}")
    col3.metric("Dung lượng tệp", f"{len(file_bytes) / 1024:.1f} KB")

    st.markdown("**Xem trước dữ liệu thô**")
    with st.container(height=300):
        st.dataframe(df.head(20), use_container_width=True)

    st.markdown("**Thống kê mô tả các biến đưa vào mô hình (biến đầu vào X & biến mục tiêu PD)**")
    st.dataframe(df[FEATURE_COLS + [TARGET_COL]].describe(), use_container_width=True)

# --------------------------------------------------------------------
# TAB 2: TRỰC QUAN HÓA DỮ LIỆU
# --------------------------------------------------------------------
with tab2:
    st.subheader("Trực quan hóa dữ liệu")
    st.caption("Phân phối biến mục tiêu (PD) và các biến đầu vào theo mô hình 5C.")

    # Biến mục tiêu luôn được vẽ đầu tiên
    st.markdown("**Biến mục tiêu — PD (Nhóm rủi ro)**")
    target_counts = df[TARGET_COL].map(TARGET_LABELS).value_counts().reset_index()
    target_counts.columns = ["Nhóm", "Số lượng"]
    fig_target = px.bar(
        target_counts, x="Nhóm", y="Số lượng", color="Nhóm",
        title="Phân phối lớp khách hàng theo rủi ro (PD)",
    )
    fig_target.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_target, use_container_width=True)

    st.markdown("**Chọn thêm biến đầu vào để trực quan hóa (tối đa 3 biến, hiển thị dạng lưới 2x2 cùng biến mục tiêu)**")
    default_vars = [FEATURE_COLS[0], FEATURE_COLS[5], FEATURE_COLS[9]]
    selected_vars = st.multiselect(
        "Biến đầu vào (thang điểm 1-5)",
        options=FEATURE_COLS,
        default=default_vars,
        max_selections=3,
        help="Các biến khảo sát 5C theo thang điểm Likert 1-5. Biểu đồ dạng cột thể hiện tần suất từng mức điểm.",
    )

    if selected_vars:
        rows = [selected_vars[i:i + 2] for i in range(0, len(selected_vars), 2)]
        for row_vars in rows:
            cols = st.columns(2)
            for c, var in zip(cols, row_vars):
                counts = df[var].value_counts().sort_index().reset_index()
                counts.columns = ["Mức điểm", "Số lượng"]
                fig = px.bar(
                    counts, x="Mức điểm", y="Số lượng",
                    title=f"Phân phối điểm — {var}",
                )
                fig.update_layout(height=320)
                c.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Chọn ít nhất một biến đầu vào để xem biểu đồ.")

# --------------------------------------------------------------------
# TAB 3: KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH
# --------------------------------------------------------------------
with tab3:
    st.subheader("Kết quả huấn luyện & kiểm định mô hình")

    if "results" not in st.session_state:
        st.info("👈 Vui lòng bấm nút **'Huấn luyện mô hình'** ở thanh bên trái để xem kết quả.")
    else:
        res = st.session_state["results"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{res['accuracy'] * 100:.2f}%")
        m2.metric("Precision", f"{res['precision'] * 100:.2f}%")
        m3.metric("Recall", f"{res['recall'] * 100:.2f}%")
        m4.metric("F1-score", f"{res['f1'] * 100:.2f}%")

        if res["auc"] is not None:
            st.metric("ROC-AUC", f"{res['auc']:.3f}")

        col_cm, col_roc = st.columns(2)

        with col_cm:
            st.markdown("**Ma trận nhầm lẫn (Confusion Matrix)**")
            cm = res["cm"]
            labels = [TARGET_LABELS.get(0, "0"), TARGET_LABELS.get(1, "1")]
            fig_cm = px.imshow(
                cm, text_auto=True,
                x=labels, y=labels,
                labels=dict(x="Dự báo", y="Thực tế", color="Số lượng"),
                color_continuous_scale="Blues",
            )
            fig_cm.update_layout(height=380)
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_roc:
            if res["fpr"] is not None:
                st.markdown("**Đường cong ROC**")
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=res["fpr"], y=res["tpr"], mode="lines", name="ROC"))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ngẫu nhiên", line=dict(dash="dash")))
                fig_roc.update_layout(
                    height=380, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate"
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.info("Không đủ dữ liệu để vẽ đường cong ROC (chỉ có 1 lớp trong tập kiểm tra).")

        st.markdown("**Báo cáo phân loại chi tiết (Classification Report)**")
        report_df = pd.DataFrame(res["report"]).transpose()
        st.dataframe(report_df.round(3), use_container_width=True)

# --------------------------------------------------------------------
# TAB 4: SỬ DỤNG MÔ HÌNH
# --------------------------------------------------------------------
with tab4:
    st.subheader("Sử dụng mô hình để dự báo")

    if "model" not in st.session_state:
        st.info("👈 Vui lòng bấm nút **'Huấn luyện mô hình'** ở thanh bên trái trước khi dự báo.")
    else:
        model = st.session_state["model"]
        feature_cols = st.session_state["feature_cols"]

        mode = st.radio(
            "Chọn chế độ dự báo",
            options=["Nhập trực tiếp thông tin khách hàng", "Tải lên tệp dữ liệu hàng loạt"],
            horizontal=True,
        )

        if mode == "Nhập trực tiếp thông tin khách hàng":
            with st.form("predict_form"):
                st.markdown("Nhập điểm khảo sát cho từng tiêu chí (thang điểm 1-5):")
                input_values = {}
                for group_name, cols in FEATURE_GROUPS.items():
                    st.markdown(f"**{group_name}**")
                    g_cols = st.columns(len(cols))
                    for gc, col in zip(g_cols, cols):
                        default_val = int(round(df[col].median()))
                        input_values[col] = gc.number_input(
                            col,
                            min_value=int(df[col].min()),
                            max_value=int(df[col].max()),
                            value=default_val,
                            step=1,
                            key=f"input_{col}",
                        )
                submitted = st.form_submit_button("Dự báo", type="primary", use_container_width=True)

            if submitted:
                x_new = pd.DataFrame([[input_values[c] for c in feature_cols]], columns=feature_cols)
                pred = model.predict(x_new)[0]
                proba = model.predict_proba(x_new)[0]

                if pred == 1:
                    st.error(f"⚠️ Dự báo: **{TARGET_LABELS[pred]}**")
                else:
                    st.success(f"✅ Dự báo: **{TARGET_LABELS[pred]}**")

                p1, p2 = st.columns(2)
                p1.metric("Xác suất không có rủi ro", f"{proba[0] * 100:.2f}%")
                p2.metric("Xác suất có rủi ro", f"{proba[1] * 100:.2f}%")

        else:
            batch_file = st.file_uploader(
                "Tải lên tệp CSV chứa các cột biến đầu vào (giống cấu trúc X_test)",
                type=["csv"],
                key="batch_predict_uploader",
                help="Tệp cần chứa đầy đủ 24 cột biến đầu vào: " + ", ".join(feature_cols),
            )
            if batch_file is not None:
                try:
                    new_df = load_data(batch_file.getvalue())
                    missing = validate_columns(new_df, feature_cols)
                    if missing:
                        st.error("Tệp thiếu các cột bắt buộc: " + ", ".join(missing))
                    else:
                        X_new = new_df[feature_cols]
                        preds = model.predict(X_new)
                        probas = model.predict_proba(X_new)[:, 1]
                        result_df = new_df.copy()
                        result_df["Dự báo"] = [TARGET_LABELS[p] for p in preds]
                        result_df["Xác suất rủi ro (%)"] = (probas * 100).round(2)

                        st.markdown("**Kết quả dự báo**")
                        with st.container(height=350):
                            st.dataframe(result_df, use_container_width=True)

                        csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            "⬇️ Tải kết quả (CSV)",
                            data=csv_bytes,
                            file_name="ket_qua_du_bao.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                except Exception as e:
                    st.error(f"Không thể xử lý tệp: {e}")
