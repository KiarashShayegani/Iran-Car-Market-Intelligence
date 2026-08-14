"""
ICMI v2.0 - Gradio Application
Multi-tab interface for HF Spaces free tier deployment.
"""

import json
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from catboost import CatBoostRegressor
from loguru import logger

# --- Paths ---
ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = ARTIFACTS_DIR / "models"
META_DIR = ARTIFACTS_DIR / "metadata"
PROCESSED_PATH = Path("data/processed/processed_latest.parquet")


# --- Model Loading ---
def load_model_artifacts():
    """Load latest model and metadata from artifacts/."""
    model_path = MODELS_DIR / "latest_model"
    meta_path = META_DIR / "latest_metadata"
    feat_path = META_DIR / "latest_features"

    if not model_path.exists():
        raise RuntimeError("No trained model found. Run training first.")

    model = CatBoostRegressor()
    # Resolve symlink
    actual_model = MODELS_DIR / model_path.readlink() if model_path.is_symlink() else model_path
    model.load_model(str(actual_model))

    actual_meta = META_DIR / meta_path.readlink() if meta_path.is_symlink() else meta_path
    with open(actual_meta, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    actual_feat = META_DIR / feat_path.readlink() if feat_path.is_symlink() else feat_path
    features = joblib.load(actual_feat)

    return model, metadata, features


def load_market_data():
    """Load processed data for analytics."""
    if not PROCESSED_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(PROCESSED_PATH)


try:
    model, metadata, feature_info = load_model_artifacts()
    market_df = load_market_data()
    MODEL_READY = True
except Exception as e:
    logger.error("Failed to load model: {}", e)
    MODEL_READY = False
    model, metadata, feature_info = None, {}, {}
    market_df = pd.DataFrame()


# --- Price Prediction Logic ---
BODY_STATUS_MAP = {
    "اتاق تعویض": 1,
    "درب تعویض": 2,
    "گلگیر تعویض": 3,
    "کاپوت تعویض": 4,
    "کامل رنگ": 5,
    "صافکاری بدون رنگ": 6,
    "دور رنگ": 7,
    "گلگیر رنگ": 8,
    "کاپوت رنگ": 9,
    "دو درب رنگ": 10,
    "یک درب رنگ": 11,
    "چند لکه رنگ": 12,
    "دو لکه رنگ": 13,
    "یک لکه رنگ": 14,
    "بدون رنگ": 15,
}


def predict_price(brand, name, trim, year, mileage, fuel, transmission, body_status):
    """Predict car price with confidence interval."""
    if not MODEL_READY:
        return gr.HTML(
            "⚠️ مدل آماده نیست. لطفاً ابتدا pipeline آموزش را اجرا کنید."
        )

    age = 1404 - int(year)
    mileage_unknown = 1 if mileage is None or mileage == "" else 0
    mileage_val = float(mileage) if mileage not in (None, "") else 50000

    body_ordinal = BODY_STATUS_MAP.get(body_status, 8)

    input_df = pd.DataFrame(
        [
            {
                "brand_slug": brand,
                "name": name,
                "trim": trim,
                "year": int(year),
                "mileage": mileage_val,
                "mileage_unknown": mileage_unknown,
                "fuel": fuel,
                "transmission": transmission,
                "body_status": body_status,
                "body_status_ordinal": body_ordinal,
                "age": age,
            }
        ]
    )

    pred = model.predict(input_df[feature_info["features"]])[0]

    # Confidence interval from similar cars
    similar = market_df[
        (market_df["brand_slug"] == brand)
        & (market_df["year"].between(int(year) - 2, int(year) + 2))
    ]

    std = similar["price"].std() if len(similar) > 5 else pred * 0.15
    ci_low = max(0, pred - 1.96 * std)
    ci_high = pred + 1.96 * std

    price_m = pred / 1e6
    ci_low_m = ci_low / 1e6
    ci_high_m = ci_high / 1e6

    html = f"""
    <div style="background: #f8f9fa; padding: 20px; border-radius: 10px;
                border: 2px solid #28a745;">
        <h3 style="color: #28a745; text-align: center;">
            💰 قیمت پیش‌بینی شده
        </h3>
        <div style="background: white; padding: 15px; border-radius: 8px;
                    text-align: center; margin-bottom: 15px;">
            <p style="font-size: 32px; font-weight: bold; color: #dc3545;
                      margin: 10px 0;">
                {pred:,.0f} تومان
            </p>
            <p style="font-size: 20px; color: #6c757d;">
                ({price_m:.1f} میلیون تومان)
            </p>
            <p style="font-size: 14px; color: #6c757d;">
                بازه اطمینان: {ci_low_m:.1f} - {ci_high_m:.1f}
                میلیون تومان
            </p>
        </div>
        <p style="text-align: center; color: #6c757d; font-size: 12px;">
            تعداد آگهی‌های مشابه در پایگاه داده: {len(similar)}
        </p>
    </div>
    """
    return gr.HTML(html)


def get_brand_options():
    """Get available brands from data."""
    if market_df.empty:
        return ["pride"]
    return sorted(market_df["brand_slug"].unique().tolist())


def get_name_options(brand):
    """Get car names for a brand."""
    if market_df.empty:
        return ["صندوق دار"]
    names = market_df[market_df["brand_slug"] == brand]["name"].unique().tolist()
    return sorted(names)


def get_trim_options(brand, name):
    """Get trims for a brand+name."""
    if market_df.empty:
        return ["ساده"]
    trims = market_df[
        (market_df["brand_slug"] == brand) & (market_df["name"] == name)
    ]["trim"].unique().tolist()
    return sorted(trims)


# --- Dashboard Functions ---
def create_price_trend_plot(brand):
    """Create price trend plot for a brand."""
    if market_df.empty:
        fig = go.Figure()
        fig.update_layout(title="داده‌ای موجود نیست")
        return fig

    df = market_df[market_df["brand_slug"] == brand].copy()
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"داده‌ای برای {brand} موجود نیست")
        return fig

    df["price_million"] = df["price"] / 1e6

    fig = px.scatter(
        df,
        x="year",
        y="price_million",
        color="trim",
        title=f"قیمت vs سال تولید - {brand}",
        labels={
            "price_million": "قیمت (میلیون تومان)",
            "year": "سال تولید",
        },
        height=500,
    )
    return fig


def create_brand_comparison():
    """Create brand comparison bar chart."""
    if market_df.empty:
        fig = go.Figure()
        fig.update_layout(title="داده‌ای موجود نیست")
        return fig

    stats = (
        market_df.groupby("brand_slug")
        .agg({"price": "mean", "listing_id": "count"})
        .reset_index()
    )
    stats["price_million"] = stats["price"] / 1e6
    stats = stats.sort_values("price_million", ascending=True)

    fig = px.bar(
        stats,
        x="price_million",
        y="brand_slug",
        orientation="h",
        title="میانگین قیمت برندها",
        labels={
            "price_million": "میانگین قیمت (میلیون تومان)",
            "brand_slug": "برند",
        },
        height=500,
    )
    return fig


# --- Gradio Interface ---
with gr.Blocks(title="ICMI v2.0 - هوشمند بازار خودرو ایران") as demo:
    gr.Markdown("# 🚗 ICMI v2.0 - ایران کار مارکت اینتلیجنس")
    gr.Markdown(
        "### پیش‌بینی قیمت خودرو با یادگیری ماشین | پوشش چندبرندی"
    )

    with gr.Tab("🎯 تخمین قیمت"):
        with gr.Row():
            with gr.Column():
                brand_dd = gr.Dropdown(
                    choices=get_brand_options(),
                    value=get_brand_options()[0]
                    if get_brand_options()
                    else "pride",
                    label="برند",
                )
                name_dd = gr.Dropdown(label="مدل")
                trim_dd = gr.Dropdown(label="تریم")
                year_s = gr.Slider(
                    1350, 1404, value=1398, step=1, label="سال تولید"
                )
                mileage_n = gr.Number(
                    value=50000, label="کارکرد (کیلومتر)"
                )
                fuel_dd = gr.Dropdown(
                    choices=["بنزینی", "دوگانه سوز", "پلاگین هیبرید", "برقی"],
                    value="بنزینی",
                    label="نوع سوخت",
                )
                trans_dd = gr.Dropdown(
                    choices=["دنده ای", "اتوماتیک"],
                    value="دنده ای",
                    label="گیربکس",
                )
                body_dd = gr.Dropdown(
                    choices=list(BODY_STATUS_MAP.keys()),
                    value="بدون رنگ",
                    label="وضعیت بدنه",
                )
                predict_btn = gr.Button(
                    "🎯 محاسبه قیمت", variant="primary"
                )

            with gr.Column():
                result_html = gr.HTML()

        # Dynamic dropdown updates
        def update_names(brand):
            return gr.Dropdown(choices=get_name_options(brand))

        def update_trims(brand, name):
            return gr.Dropdown(choices=get_trim_options(brand, name))

        brand_dd.change(update_names, inputs=brand_dd, outputs=name_dd)
        name_dd.change(
            update_trims, inputs=[brand_dd, name_dd], outputs=trim_dd
        )

        predict_btn.click(
            predict_price,
            inputs=[
                brand_dd,
                name_dd,
                trim_dd,
                year_s,
                mileage_n,
                fuel_dd,
                trans_dd,
                body_dd,
            ],
            outputs=result_html,
        )

    with gr.Tab("📊 داشبورد بازار"):
        with gr.Row():
            brand_select = gr.Dropdown(
                choices=get_brand_options(),
                value=get_brand_options()[0]
                if get_brand_options()
                else "pride",
                label="انتخاب برند برای تحلیل",
            )
            refresh_btn = gr.Button("🔄 به‌روزرسانی نمودار")

        trend_plot = gr.Plot(label="روند قیمت")
        compare_plot = gr.Plot(label="مقایسه برندها")

        def update_dashboard(brand):
            return create_price_trend_plot(brand), create_brand_comparison()

        refresh_btn.click(
            update_dashboard,
            inputs=brand_select,
            outputs=[trend_plot, compare_plot],
        )
        demo.load(
            update_dashboard,
            inputs=brand_select,
            outputs=[trend_plot, compare_plot],
        )

    with gr.Tab("ℹ️ اطلاعات مدل"):
        if MODEL_READY:
            gr.Markdown(
                f"""
                **نوع مدل:** {metadata.get('model_type', 'N/A')}

                **تاریخ آموزش:** {metadata.get('trained_at', 'N/A')}

                **تعداد نمونه آموزشی:**
                {metadata.get('n_samples_train', 'N/A')}

                **تعداد برندها:** {metadata.get('n_brands', 'N/A')}

                **عملکرد:**
                - R² (تست):
                  {metadata.get('performance', {}).get('test_r2', 'N/A')}
                - MAE (تست):
                  {metadata.get('performance', {}).get('test_mae_million', 'N/A')}
                  میلیون تومان
                - CV R²:
                  {metadata.get('performance', {}).get('cv_r2_mean', 'N/A')}
                  ±
                  {metadata.get('performance', {}).get('cv_r2_std', 'N/A')}
                """
            )
        else:
            gr.Markdown(
                "⚠️ مدل یافت نشد. لطفاً ابتدا pipeline را اجرا کنید."
            )

        gr.Markdown("---")
        gr.Markdown(
            """
            **⚠️ سلب مسئولیت:**
            - این پیش‌بینی بر اساس داده‌های تاریخی است
            - قیمت واقعی می‌تواند متفاوت باشد
            - برای معاملات واقعی با کارشناس مشورت کنید
            """
        )


if __name__ == "__main__":
    demo.launch()
