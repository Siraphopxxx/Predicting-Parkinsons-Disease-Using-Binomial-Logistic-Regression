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


  <img width="1676" height="583" alt="image" src="https://github.com/user-attachments/assets/5245cf9e-639c-477a-8815-94815488edfb" />
<img width="1676" height="583" alt="image" src="https://github.com/user-attachments/assets/5245cf9e-639c-477a-8815-94815488edfb" />
<img width="1662" height="580" alt="image" src="https://github.com/user-attachments/assets/378a1801-a8fd-4248-81e5-0939fa7b37f7" />
<img width="1662" height="580" alt="image" src="https://github.com/user-attachments/assets/378a1801-a8fd-4248-81e5-0939fa7b37f7" />
<img width="1652" height="585" alt="image" src="https://github.com/user-attachments/assets/d52b7b4c-2c2a-49b1-8073-62c6d113a2a9" />
<img width="1652" height="585" alt="image" src="https://github.com/user-attachments/assets/d52b7b4c-2c2a-49b1-8073-62c6d113a2a9" />
<img width="776" height="439" alt="image" src="https://github.com/user-attachments/assets/fbd1e24f-4d56-4d94-b208-ef2ee85e0c0d" />
<img width="776" height="439" alt="image" src="https://github.com/user-attachments/assets/fbd1e24f-4d56-4d94-b208-ef2ee85e0c0d" />




