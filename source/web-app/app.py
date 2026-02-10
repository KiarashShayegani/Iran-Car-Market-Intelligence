import gradio as gr
import numpy as np
import joblib
import pandas as pd

# Load model and scaler
scaler = joblib.load('input_scaler.pkl')
model = joblib.load('random_forest_model.pkl')

# Current Persian year for age calculation
CURRENT_YEAR = 1404

# Feature mappings (Persian for UI only)
name_options = {
    1: "هاچ بک",
    2: "صندوق دار", 
    3: "141",
    4: "132",
    5: "131",
    6: "111",
    7: "151"
}

trim_options = {
    1: "ساده",
    2: "LE",
    3: "SL",
    4: "LX",
    5: "SX",
    6: "EX",
    7: "TL",
    8: "پلاس",
    9: "SE",
    10: "GX"
}

fuel_options = {
    1: "بنزینی",
    2: "دوگانه سوز"
}

# Transmission fixed to manual for Pride cars
transmission_value = 1  # 'دنده ای'

body_status_options = {
    1: "اتاق تعویض",
    2: "درب تعویض",
    3: "گلگیر تعویض",
    4: "کاپوت تعویض",
    5: "کامل رنگ",
    6: "صافکاری بدون رنگ",
    7: "دور رنگ",
    8: "گلگیر رنگ",
    9: "کاپوت رنگ",
    10: "دو درب رنگ",
    11: "یک درب رنگ",
    12: "چند لکه رنگ",
    13: "دو لکه رنگ",
    14: "یک لکه رنگ",
    15: "بدون رنگ"
}

def predict_price(year, mileage, name, trim, fuel, body_status):
    """
    Predict car price based on input features
    """
    # Calculate car age
    age = CURRENT_YEAR - year
    
    # Extract numeric values from selected options
    name_value = int(name.split(":")[0])
    trim_value = int(trim.split(":")[0])
    fuel_value = int(fuel.split(":")[0])
    body_value = int(body_status.split(":")[0])
    
    # Create input array in EXACT training order:
    # ['name', 'trim', 'mileage', 'fuel', 'transmission', 'body_status', 'age']
    features = np.array([[
        name_value,         # name_encoded
        trim_value,         # trim_encoded
        mileage,            # mileage
        fuel_value,         # fuel_encoded
        transmission_value, # transmission_encoded (always 1)
        body_value,         # body_status_encoded
        age                 # age
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict price
    predicted_price = model.predict(features_scaled)[0]
    
    # Format output
    if predicted_price < 0:
        return "⚠️ خطا در پیش‌بینی قیمت"
    
    # Format price in million Toman
    price_million = predicted_price / 1_000_000
    
    # Create output HTML
    output = f"""
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 2px solid #28a745;">
        <h3 style="color: #28a745; text-align: center; margin-bottom: 20px;">💰 قیمت پیش‌بینی شده</h3>
        
        <div style="background-color: white; padding: 15px; border-radius: 8px; margin-bottom: 15px; text-align: center;">
            <p style="font-size: 28px; font-weight: bold; color: #dc3545; margin: 10px 0;">
                {predicted_price:,.0f} تومان
            </p>
            <p style="font-size: 18px; color: #6c757d;">
                ({price_million:.1f} میلیون تومان)
            </p>
        </div>
        
        <div style="background-color: #e9ecef; padding: 15px; border-radius: 8px;">
            <h4 style="color: #495057; margin-top: 0;">📋 جزئیات ورودی</h4>
            <table style="width: 100%; color: #495057;">
                <tr>
                    <td style="padding: 5px;">سال تولید:</td>
                    <td style="padding: 5px;"><strong>{year}</strong></td>
                </tr>
                <tr>
                    <td style="padding: 5px;">سن خودرو:</td>
                    <td style="padding: 5px;"><strong>{age} سال</strong></td>
                </tr>
                <tr>
                    <td style="padding: 5px;">کارکرد:</td>
                    <td style="padding: 5px;"><strong>{mileage:,.0f} کیلومتر</strong></td>
                </tr>
                <tr>
                    <td style="padding: 5px;">نام:</td>
                    <td style="padding: 5px;"><strong>{name_options[name_value]}</strong></td>
                </tr>
                <tr>
                    <td style="padding: 5px;">تریم:</td>
                    <td style="padding: 5px;"><strong>{trim_options[trim_value]}</strong></td>
                </tr>
                <tr>
                    <td style="padding: 5px;">سوخت:</td>
                    <td style="padding: 5px;"><strong>{fuel_options[fuel_value]}</strong></td>
                </tr>
                <tr>
                    <td style="padding: 5px;">وضعیت بدنه:</td>
                    <td style="padding: 5px;"><strong>{body_status_options[body_value]}</strong></td>
                </tr>
                <tr>
                    <td style="padding: 5px;">گیربکس:</td>
                    <td style="padding: 5px;"><strong>دنده‌ای</strong></td>
                </tr>
            </table>
        </div>
    </div>
    """
    
    return output

# Create dropdown lists for UI
name_dropdown = [f"{key}: {value}" for key, value in name_options.items()]
trim_dropdown = [f"{key}: {value}" for key, value in trim_options.items()]
fuel_dropdown = [f"{key}: {value}" for key, value in fuel_options.items()]
body_dropdown = [f"{key}: {value}" for key, value in body_status_options.items()]

# Create Gradio interface
with gr.Blocks(title="پیش‌بینی قیمت پراید") as app:
    gr.Markdown("# 🚗 پیش‌بینی قیمت خودروهای پراید")
    gr.Markdown("### مدل هوش مصنوعی آموزش‌دیده بر اساس داده‌های واقعی")
    
    with gr.Row():
        with gr.Column(scale=1):
            year_input = gr.Slider(
                minimum=1350,
                maximum=1404,
                value=1398,
                step=1,
                label="سال تولید",
                info="سال تولید خودرو (هجری شمسی)"
            )
            
            mileage_input = gr.Number(
                value=50000,
                label="کارکرد (کیلومتر)",
                info="کارکرد خودرو بر حسب کیلومتر"
            )
            
            name_input = gr.Dropdown(
                choices=name_dropdown,
                value=name_dropdown[1],  # صندوق دار as default
                label="نام خودرو",
                info="انتخاب مدل خودرو"
            )
        
        with gr.Column(scale=1):
            trim_input = gr.Dropdown(
                choices=trim_dropdown,
                value=trim_dropdown[8],  # SE as default
                label="نوع تریم",
                info="انتخاب تریم خودرو"
            )
            
            fuel_input = gr.Dropdown(
                choices=fuel_dropdown,
                value=fuel_dropdown[0],  # بنزینی as default
                label="نوع سوخت",
                info="انتخاب نوع سوخت"
            )
            
            body_input = gr.Dropdown(
                choices=body_dropdown,
                value=body_dropdown[14],  # بدون رنگ as default
                label="وضعیت بدنه",
                info="وضعیت رنگ و بدنه خودرو"
            )
    
    predict_button = gr.Button("🎯 محاسبه قیمت", variant="primary", size="lg")
    
    output_html = gr.HTML(label="نتیجه")
    
    # Connect function
    predict_button.click(
        fn=predict_price,
        inputs=[year_input, mileage_input, name_input, trim_input, fuel_input, body_input],
        outputs=output_html
    )
    
    # Example inputs
    gr.Examples(
        examples=[
            [1398, 50000, name_dropdown[1], trim_dropdown[8], fuel_dropdown[0], body_dropdown[14]],
            [1400, 20000, name_dropdown[1], trim_dropdown[9], fuel_dropdown[0], body_dropdown[14]],
            [1390, 150000, name_dropdown[0], trim_dropdown[0], fuel_dropdown[1], body_dropdown[4]]
        ],
        inputs=[year_input, mileage_input, name_input, trim_input, fuel_input, body_input],
        outputs=output_html,
        label="🎮 مثال‌های آماده",
        fn=predict_price
    )
    
    # Footer
    gr.Markdown("---")
    gr.Markdown("""
    **📊 اطلاعات مدل:**  
    🎯 دقت: 83% | 📉 خطای متوسط: 45 میلیون تومان | 📈 داده آموزشی: 580 خودرو  
    
    **⚠️ توجه:**  
    - این پیش‌بینی بر اساس داده‌های تاریخی است  
    - قیمت واقعی می‌تواند متفاوت باشد  
    - برای معاملات واقعی با کارشناس مشورت کنید
    """)

# Launch app
if __name__ == "__main__":
    app.launch()