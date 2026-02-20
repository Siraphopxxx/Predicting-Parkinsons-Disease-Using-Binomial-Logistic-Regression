# Parkinson’s Disease Prediction Pipeline
- Binary Classification project to assist in early-stage Parkinson's diagnosis using vocal feature analysis.

# Project Overview
- โปรเจกต์นี้ใช้ชุดข้อมูลจาก UCI Machine Learning Repository ซึ่งรวบรวมค่าจากการบันทึกเสียง (Biomedical voice measurements) ของผู้ป่วยโรคพาร์กินสันและคนปกติ เพื่อสร้าง Model ที่สามารถจำแนกสถานะของโรคได้อย่างแม่นยำ

# Data Preprocessing (หัวใจสำคัญของโปรเจกต์)
- เนื่องจากข้อมูลทางการแพทย์มักมีความคลาดเคลื่อนสูง ผมจึงเน้นที่การทำ Data Preparation:

- Outlier Pruning: พัฒนา Function สำหรับ Pruning ข้อมูลที่ผิดปกติออกแบบ Iterative โดยใช้ค่า IQR จนกว่าจะได้เปอร์เซ็นต์ข้อมูลที่ยอมรับได้

- Feature Scaling: ปรับค่าฟีเจอร์ที่มีความต่างของระดับ (Scale) สูงให้เป็นมาตรฐานเดียวกันด้วย Z-score Normalization

- Class Handling: จัดการเป้าหมาย (Target) ให้เป็น Categorical data เพื่อความถูกต้องในการทำ Classification

# Model & Evaluation
- ผมเลือกใช้ Binomial Logistic Regression เนื่องจากเป็น Model ที่สามารถให้ค่าความน่าจะเป็น (Probability) ในการวินิจฉัยได้อย่างชัดเจน

  Performance Metrics:

  Accuracy: 80.00%

  Precision: 72.50%

  Recall: 69.74%
 
  F1-Score: 0.7086
