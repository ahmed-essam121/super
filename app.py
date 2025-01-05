from flask import Flask, request, jsonify
import pickle
import numpy as np
import math

# إنشاء تطبيق Flask
app = Flask(__name__)

# تحميل النموذج المدرب (تأكد من وجود الملف 'model.pkl' في نفس المجلد)
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# نقطة النهاية للتنبؤ
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # الحصول على البيانات المدخلة من الطلب
        data = request.get_json(force=True)
        
        # استخراج المدخلات من البيانات (مثل أن تكون قائمة من القيم)
        features = np.array(data['features']).reshape(1, -1)  # تحويل المدخلات لمصفوفة
       
        # إجراء التنبؤ باستخدام النموذج المدرب
        prediction = model.predict(features)

        # إرجاع التنبؤ في تنسيق JSON
        return jsonify(prediction=prediction[0])

    except Exception as e:
        # إذا كان هناك خطأ في العملية، إرجاع رسالة الخطأ
        return jsonify(error=str(e)), 400

# تشغيل الخادم
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)
