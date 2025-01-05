# استيراد المكتبات
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, request, jsonify
import pickle
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# -------------------------------------------------------
# تحميل البيانات
file_path = r'C:\Users\Elbostan\Desktop\full project\super market eda\supermarket_sales - Sheet1.csv'

# التأكد من صحة المسار
assert os.path.exists(file_path), "الملف غير موجود!"

data = pd.read_csv(file_path)

# عرض المعلومات الأساسية
print(data.head())
print(data.tail())
print(data.info())
print(data.isna().sum())

# ----------------------------------------------------
# الإحصائيات الوصفية
summary = data.describe()

# رسم Heatmap للإحصائيات الوصفية
fig = px.imshow(
    summary,
    color_continuous_scale="RdYlGn",
    title='خريطة الحرارة للإحصائيات الوصفية للبيانات',
    labels={"x": "أعمدة البيانات", "y": "الإحصائيات"}
)
fig.update_layout(
    title_font_size=20,
    xaxis_title='أعمدة البيانات',
    yaxis_title='الإحصائيات',
    xaxis_tickangle=45
)
fig.show()

# ----------------------------------------------------
# تجميع البيانات حسب نوع العميل والجنس
gender_customer_counts = data.groupby(['Customer type', 'Gender']).size().reset_index(name='Count')
print(gender_customer_counts)

# رسم باربلوت لعدد العملاء حسب الجنس ونوع العميل
sns.barplot(x='Customer type', y='Count', hue='Gender', data=gender_customer_counts)
plt.title('عدد الجنسين حسب نوع العميل')
plt.xlabel('نوع العميل')
plt.ylabel('العدد')
plt.show()

# ----------------------------------------------------
# تجميع البيانات حسب نوع العميل والفروع
customer_type_in_branches = data.groupby(["Customer type", "Branch"]).size().reset_index(name='Count')
print(customer_type_in_branches)

# رسم باربلوت لتوزيع العملاء حسب الفروع
fig = px.bar(customer_type_in_branches, x="Branch", y="Count", color="Customer type", barmode="group")
fig.show()

# ----------------------------------------------------
# رسم باربلوت لأسعار المنتجات
plt.figure(figsize=(25, 8))
sns.barplot(x='Product line', y='Unit price', data=data, palette='Blues')
plt.title('أسعار المنتجات')
plt.xlabel('خط الإنتاج')
plt.ylabel('سعر الوحدة')
plt.show()

# -----------------------------------------------------
# حساب تكرارات خطوط المنتجات
product_line_counts = data['Product line'].value_counts().reset_index()
product_line_counts.columns = ['Product line', 'Count']
print(product_line_counts)

# رسم كميات المنتجات حسب خط الإنتاج وسعر الوحدة
fig = px.bar(data, x="Product line", y="Quantity", color='Unit price', barmode="group")
fig.show()

# --------------------------------------------------------
# رسم كميات المنتجات حسب التاريخ وخط الإنتاج
fig = px.bar(data, x="Date", y="Quantity", color="Product line", title="مجموع الكميات حسب خط الإنتاج والتاريخ")
fig.show()

# رسم كميات المنتجات حسب التاريخ والفروع
fig = px.bar(data, x="Date", y="Quantity", color="Branch", title="مجموع الكميات حسب الفرع والتاريخ")
fig.show()

# --------------------------------------------------------
# تحليل COGS وخطوط الإنتاج
cogs_product_line = data.groupby(['cogs', 'Product line']).size().reset_index(name='Count')
cogs_product_line.drop(['Count'], axis=1, inplace=True)
print(cogs_product_line)

# -------------------------------------------------------
# رسم بياني لعدد الكميات مقابل COGS ودخل إجمالي
fig = go.Figure()

# إضافة العلاقة بين الكميات و COGS
fig.add_trace(go.Scatter(x=data['Quantity'], y=data['cogs'], mode='markers', name='Quantity vs COGS'))

# إضافة العلاقة بين الكميات والدخل الإجمالي
fig.add_trace(go.Scatter(x=data['Quantity'], y=data['gross income'], mode='markers', name='Quantity vs Gross Income'))

fig.update_layout(
    title='علاقات متعددة بين الأعمدة',
    xaxis_title='الكمية',
    yaxis_title='القيم',
    legend_title='العلاقات'
)
fig.show()

# ----------------------------------------------------
# رسم بياني لخطوط الإنتاج مقابل التقييم
plt.figure(figsize=(25, 8))
sns.barplot(x='Product line', y='Rating', data=data, palette='Blues')
plt.title('خطوط الإنتاج مقابل التقييم')
plt.xlabel('خط الإنتاج')
plt.ylabel('التقييم')
plt.show()

# ----------------------------------------------------
# رسم Pairplot للأعمدة الرقمية
sns.pairplot(
    data[["Quantity", "Tax 5%", "gross margin percentage", "gross income"]],
    palette="viridis"
)
plt.show()

# -----------------------------------------------------
# خريطة الحرارة للتباين بين الأعمدة الرقمية
numerical_columns = data.select_dtypes(include=['int', 'float']).columns
correlation_matrix = data[numerical_columns].corr()

# استخدام أنماط Seaborn المدمجة
sns.set_style('darkgrid')

# رسم خريطة الحرارة للتباين بين الأعمدة
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap='viridis',
    vmin=-1,
    vmax=1
)
plt.title('خريطة الحرارة للتباين بين الأعمدة')
plt.show()

#---------------------------------------------------
# حذف العمود غير المطلوب
data.drop(["Invoice ID"], axis=1, inplace=True)

# ترميز الأعمدة النصية باستخدام LabelEncoder
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':  
        data[col] = le.fit_transform(data[col])

# -----------------------------------------------------
# تطبيق MinMaxScaler على البيانات
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# -------------------------------------------------------
# تحديد المتغيرات المستقلة والتابعة
col_to_drop = data.columns[-2]
X = data.drop(columns=[col_to_drop])
y = data.iloc[:, -2]

# -------------------------------------------------------
# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# إنشاء نموذج الانحدار الخطي
model = LinearRegression()

# تدريب النموذج
model.fit(X_train, y_train)

# التنبؤ بالقيم باستخدام البيانات الاختبارية
y_pred = model.predict(X_test)

# تقييم النموذج
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# عرض النتائج
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# ----------------------------------------------------
# حفظ النموذج باستخدام joblib
joblib.dump(model, 'model.pkl')

