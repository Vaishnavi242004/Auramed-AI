import streamlit as st
import pandas as pd
import numpy as np
import shap
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from io import BytesIO

st.set_page_config(layout="wide", page_title="AURAMED Clinical AI", page_icon="🏥")

# ================= CUSTOM CSS =================
st.markdown("""
<style>
body {font-family: 'Times New Roman', Times, serif;}
h1,h2,h3{color:#1E3A8A;font-weight:700;}
div.stMetric{
background-color: black;
border-radius:12px;
padding:15px;
box-shadow:0 4px 6px rgba(0,0,0,0.05);
border-left:5px solid #3b82f6;
}
div.stButton>button{
border-radius:8px;
font-weight:600;
background-color:white;
color:black;
transition:all 0.2s ease-in-out;
}
div.stButton>button:hover{
background-color:#1D4ED8;
transform:translateY(-2px);
}
[data-testid="stSidebar"]{
background-color:black;
border-right:1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# ================= LOGIN =================
users={
"vaishnuvaishnavi2004@gmail.com":{"password":"vaish123","role":"Doctor"},
"pravallikadoc001@gmail.com":{"password":"pd1_AM","role":"Doctor"},
"sravanthinur001@gmail.com":{"password":"sn1_AM","role":"Nurse"}
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in=False

if not st.session_state.logged_in:

    st.markdown("## 🔐 AURAMED-AI Login")

    email=st.text_input("Email")
    password=st.text_input("Password",type="password")

    if st.button("Login"):
        if email in users and users[email]["password"]==password:
            st.session_state.logged_in=True
            st.session_state.role=users[email]["role"]
            st.success(f"Logged in as {users[email]['role']}")
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.stop()

if st.sidebar.button("Logout"):
    st.session_state.logged_in=False
    st.rerun()

st.sidebar.write("Role:",st.session_state.role)
st.title("🏥 AURAMED: An Explainable AI system for Personalized Drug & Dosage Recommendation")

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    return pd.read_excel("auramed_clinicaldataset(1).xlsx")

data=load_data()

# ================= ENCODERS =================
le_gender=LabelEncoder()
le_severity=LabelEncoder()
le_drug=LabelEncoder()
le_adr=LabelEncoder()

data["Gender"]=le_gender.fit_transform(data["Gender"])
data["Severity"]=le_severity.fit_transform(data["Severity"])
data["Drug"]=le_drug.fit_transform(data["Drug"])
data["ADR_Risk"]=le_adr.fit_transform(data["ADR_Risk"])

X=data.drop(columns=["Drug","ADR_Risk","Dosage_mg"])
X_adr=data.drop(columns=["ADR_Risk","Dosage_mg"])

y_drug=data["Drug"]
y_adr=data["ADR_Risk"]
y_dosage=data["Dosage_mg"]

# ================= TABLET LOOKUP =================
tablet_map={
"Paracetamol":{500:"Crocin 500 mg",650:"Dolo 650 mg"},
"Metformin":{500:"Glucophage 500 mg",850:"Gluformin 850 mg"},
"Lisinopril":{10:"Coversyl 10 mg",20:"Zestril 20 mg"}
}

def get_tablet_name(drug,dosage):
    if drug in tablet_map:
        closest=min(tablet_map[drug].keys(),key=lambda x:abs(x-dosage))
        return tablet_map[drug][closest]
    return f"{drug} {int(dosage)} mg Tablet"

# ================= MODELS =================
@st.cache_resource
def train_models():
    drug_model=RandomForestClassifier(n_estimators=200,random_state=42)
    adr_model=RandomForestClassifier(n_estimators=200,random_state=42)
    dosage_model=RandomForestRegressor(n_estimators=200,random_state=42)

    drug_model.fit(X,y_drug)
    adr_model.fit(X_adr,y_adr)
    dosage_model.fit(X,y_dosage)

    return drug_model,adr_model,dosage_model

drug_model,adr_model,dosage_model=train_models()

@st.cache_resource
def get_drug_explainer():
    return shap.TreeExplainer(drug_model)

@st.cache_resource
def get_adr_explainer():
    return shap.TreeExplainer(adr_model)

drug_explainer=get_drug_explainer()
adr_explainer=get_adr_explainer()

# ================= SIDEBAR =================
st.sidebar.title("⚕️ Patient Profile")

with st.sidebar.expander("👤 Demographics",expanded=True):
    age=st.number_input("Age",0,100,value=45)
    gender=st.selectbox("Gender",le_gender.classes_)
    weight=st.number_input("Weight (kg)",3,200,value=70)

with st.sidebar.expander("🩺 Vitals & Labs"):
    bp=st.number_input("Blood Pressure",80,200,value=120)
    glucose=st.number_input("Blood Glucose",70,300,value=100)
    cholesterol=st.number_input("Cholesterol",100,400,value=180)
    crp=st.number_input("CRP",0.0,50.0,value=2.5)
    egfr=st.number_input("Kidney Function",10.0,150.0,value=90.0)
    alt=st.number_input("Liver Function",5.0,200.0,value=25.0)

with st.sidebar.expander("📋 Clinical History"):
    severity=st.selectbox("Disease Severity",le_severity.classes_)
    frequency=st.selectbox("Frequency",["Once daily","Twice daily","Thrice daily"])
    comorbidities=st.multiselect("Comorbidities",["Diabetes","Hypertension","CKD","Liver Disease"])
    symptoms=st.multiselect("Symptoms",["Fever","Pain","Inflammation","Fatigue"])

with st.sidebar.expander("💊 Medication History"):
    previous_drugs=st.multiselect("Previous Drugs",list(le_drug.classes_))
    allergies=st.multiselect("Allergy History",list(le_drug.classes_))

col_pred1,col_pred2=st.sidebar.columns(2)
with col_pred1:
    predict=st.button("Predict Treatment",use_container_width=True)
with col_pred2:
    reset=st.button("Reset",use_container_width=True)

if reset:
    st.rerun()

# ================= SHAP FUNCTION =================
def get_shap_summary(input_data,explainer):
    try:
        shap_values=explainer.shap_values(input_data)
        if isinstance(shap_values,list):
            shap_val=shap_values[0][0]
        else:
            shap_val=shap_values[0]
        shap_val=np.array(shap_val).flatten()
        expl_df=pd.DataFrame({"Feature":input_data.columns,"Impact":shap_val})
        expl_df["Absolute Impact"]=expl_df["Impact"].abs()
        return expl_df.sort_values(by="Absolute Impact",ascending=False).head(5)
    except:
        return pd.DataFrame(columns=["Feature","Impact","Absolute Impact"])

# ================= ADR GRAPH =================
def plot_adr_bar(input_data_adr,model,title):
    probs=model.predict_proba(input_data_adr)[0]
    classes=le_adr.inverse_transform(model.classes_)
    plt.figure(figsize=(8,5))
    colors=['#4CAF50' if 'Low' in c else '#FF9800' if 'Medium' in c else '#F44336' for c in classes]
    bars=plt.bar(classes,probs,color=colors)
    plt.title(title,fontweight="bold")
    plt.ylabel("Probability")
    plt.xlabel("ADR Risk Level")
    plt.ylim(0,1)
    for bar in bars:
        yval=bar.get_height()
        plt.text(bar.get_x()+bar.get_width()/2,yval+0.03,f"{yval:.2f}",ha='center')
    buf=BytesIO()
    plt.tight_layout()
    plt.savefig(buf,format='png')
    plt.close()
    buf.seek(0)
    return buf

# ================= PREDICTION =================
if predict:
    diabetes=1 if "Diabetes" in comorbidities else 0
    hypertension=1 if "Hypertension" in comorbidities else 0
    ckd=1 if "CKD" in comorbidities else 0
    liver_disease=1 if "Liver Disease" in comorbidities else 0
    fever=1 if "Fever" in symptoms else 0
    pain=1 if "Pain" in symptoms else 0
    inflammation=1 if "Inflammation" in symptoms else 0
    fatigue=1 if "Fatigue" in symptoms else 0

    input_dict={
        "Age":age,
        "Gender":le_gender.transform([gender])[0],
        "Weight":weight,
        "Blood_Pressure":bp,
        "Glucose":glucose,
        "Cholesterol":cholesterol,
        "CRP":crp,
        "eGFR":egfr,
        "ALT":alt,
        "Severity":le_severity.transform([severity])[0],
        "Diabetes":diabetes,
        "Hypertension":hypertension,
        "CKD":ckd,
        "Liver_Disease":liver_disease,
        "Fever":fever,
        "Pain":pain,
        "Inflammation":inflammation,
        "Fatigue":fatigue
    }

    input_data=pd.DataFrame([input_dict])
    input_data=input_data.reindex(columns=X.columns,fill_value=0).astype(float)

    drug_pred=drug_model.predict(input_data)[0]
    dosage_pred=dosage_model.predict(input_data)[0]
    drug_result=le_drug.inverse_transform([drug_pred])[0]
    dosage_result=round(dosage_pred,2)
    tablet_result=get_tablet_name(drug_result,round(dosage_result))

    probs=drug_model.predict_proba(input_data)[0]
    alt_index=np.argsort(probs)[-2]
    alternate_drug=le_drug.inverse_transform([alt_index])[0]
    alt_dosage=round(dosage_model.predict(input_data)[0]*0.9,2)
    alt_tablet=get_tablet_name(alternate_drug,round(alt_dosage))

    input_data_primary_adr=input_data.copy()
    input_data_primary_adr["Drug"]=drug_pred
    input_data_alt_adr=input_data.copy()
    input_data_alt_adr["Drug"]=alt_index
    adr_pred=adr_model.predict(input_data_primary_adr)[0]
    adr_result=le_adr.inverse_transform([adr_pred])[0]

    shap_drug_primary=get_shap_summary(input_data,drug_explainer)
    if not shap_drug_primary.empty and "Feature" in shap_drug_primary.columns:
        top_features=shap_drug_primary.head(3)["Feature"].tolist()
    else:
        top_features=["clinical indicators","lab results","patient symptoms"]

    reason_text=f"{drug_result} recommended due to factors: {', '.join(top_features)}."
    alt_text=f"{alternate_drug} suggested as alternate."

    # ================= DISPLAY =================
    st.subheader("Recommended Treatment📝 :")
    col1,col2=st.columns(2)
    with col1:
        st.metric("Primary Drug",drug_result)
        st.metric("Dosage",f"{dosage_result} mg")
        st.metric("ADR Risk",adr_result)
        st.success(f"Dispense Label: {tablet_result}")
    with col2:
        st.metric("Alternate Drug",alternate_drug)
        st.metric("Dosage",f"{alt_dosage} mg")
        st.success(f"Dispense Label: {alt_tablet}")

    st.subheader("Explainable AI Decision🧠 :")
    st.write(reason_text)
    st.write(alt_text)

    # ================= ADR VISUAL =================
    st.subheader("ADR Risk Visualization📊 :")
    col3,col4=st.columns(2)
    with col3:
        buf_primary=plot_adr_bar(input_data_primary_adr,adr_model,f"{drug_result} ADR Risk")
        st.image(buf_primary,use_container_width=True)
    with col4:
        buf_alt=plot_adr_bar(input_data_alt_adr,adr_model,f"{alternate_drug} ADR Risk")
        st.image(buf_alt,use_container_width=True)

    # ================= PDF REPORT =================
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1E3A8A'),
        spaceAfter=14
    )

    heading_style = ParagraphStyle(
        'CustomH2',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2563EB'),
        spaceBefore=12,
        spaceAfter=8
    )

    normal_style = styles["Normal"]

    elements = []

    # ================= HEADER =================
    elements.append(Paragraph("<b>AURAMED Clinical Treatment Report</b>", title_style))
    elements.append(Paragraph("AI-generated Personalized Medication Plan", normal_style))
    elements.append(Spacer(1, 0.2 * inch))

    # ================= PATIENT DETAILS =================
    patient_data = [
        ['Patient Age', str(age), 'Gender', str(gender)],
        ['Weight', f"{weight} kg", 'Severity', str(severity)],
        ['Comorbidities', ', '.join(comorbidities) if comorbidities else 'None',
         'Symptoms', ', '.join(symptoms) if symptoms else 'Not specified'],
        ['Allergies', ', '.join(allergies) if allergies else 'None',
         'History', ', '.join(previous_drugs) if previous_drugs else 'None']
    ]

    t = Table(patient_data, colWidths=[1.2*inch,2.3*inch,1.2*inch,2.3*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(0,-1),colors.HexColor('#f1f5f9')),
        ('BACKGROUND',(2,0),(2,-1),colors.HexColor('#f1f5f9')),
        ('TEXTCOLOR',(0,0),(-1,-1),colors.black),
        ('ALIGN',(0,0),(-1,-1),'LEFT'),
        ('FONTNAME',(0,0),(0,-1),'Helvetica-Bold'),
        ('FONTNAME',(2,0),(2,-1),'Helvetica-Bold'),
        ('BOTTOMPADDING',(0,0),(-1,-1),8),
        ('TOPPADDING',(0,0),(-1,-1),8),
        ('GRID',(0,0),(-1,-1),0.5,colors.lightgrey)
    ]))
    elements.append(t)
    elements.append(Spacer(1,0.3*inch))

    # ================= TREATMENT TABLE =================
    elements.append(Paragraph("<b>Recommended Treatments</b>", heading_style))

    treatment_data = [
        ['Parameter','Primary Recommendation','Alternate Option'],
        ['Drug Name', drug_result, alternate_drug],
        ['Dosage', f"{dosage_result} mg ({frequency})", f"{alt_dosage} mg ({frequency})"],
        ['Dispense Label', tablet_result, alt_tablet],
        ['ADR Risk Level', adr_result, "Refer to chart below"]
    ]

    t2 = Table(treatment_data, colWidths=[1.5*inch,2.75*inch,2.75*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1E3A8A')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTNAME',(0,1),(0,-1),'Helvetica-Bold'),
        ('BOTTOMPADDING',(0,0),(-1,-1),8),
        ('TOPPADDING',(0,0),(-1,-1),8),
        ('GRID',(0,0),(-1,-1),0.5,colors.grey)
    ]))
    elements.append(t2)
    elements.append(Spacer(1,0.3*inch))

    # ================= EXPLAINABLE AI =================
    elements.append(Paragraph("<b>Explainable AI Decision</b>", heading_style))
    elements.append(Paragraph(reason_text, normal_style))
    elements.append(Spacer(1,0.15*inch))
    elements.append(Paragraph("<b>Alternate Recommendation Reason</b>", heading_style))
    elements.append(Paragraph(alt_text, normal_style))
    elements.append(Spacer(1,0.3*inch))

    # ================= ADR CHARTS =================
    elements.append(Paragraph("<b>ADR Risk Profiles</b>", heading_style))

    chart_data = [[
        Image(buf_primary,width=3.2*inch,height=2.1*inch),
        Image(buf_alt,width=3.2*inch,height=2.1*inch)
    ]]

    t3 = Table(chart_data)
    t3.setStyle(TableStyle([
        ('ALIGN',(0,0),(-1,-1),'CENTER')
    ]))
    elements.append(t3)
    elements.append(Spacer(1,0.3*inch))

    # ================= DISCLAIMER =================
    elements.append(Spacer(1,0.4*inch))
    elements.append(
        Paragraph(
            "⚠ This report is generated by an AI Clinical Decision Support Tool. "
            "Final medical decisions must be taken by a licensed healthcare professional.",
            styles["Italic"]
        )
    )

    # ================= BUILD PDF =================
    try:
        doc.build(elements)
        buffer.seek(0)
        st.download_button(
            label="Download Treatment Report (PDF)",
            data=buffer,
            file_name="AURAMED_Treatment_Report.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")