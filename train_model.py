"""
Train a Random Forest fraud detection model on synthetic EMSCAD-like data.
In production, replace with the actual EMSCAD dataset from Kaggle.
"""
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- Synthetic training data (representative of EMSCAD patterns) ---
# In production: load actual EMSCAD CSV from Kaggle
legitimate_jobs = [
    "We are looking for a qualified software engineer with 3 years experience in Python and Django. Competitive salary offered. Please apply with your CV and cover letter to hr@techcompany.co.za. Background check required.",
    "Marketing Manager position available at established Cape Town firm. Minimum 5 years experience required. Salary: R35,000 - R45,000 per month. Send CV to careers@marketingfirm.co.za. Office based role.",
    "Accountant needed for well-established Johannesburg company. CA(SA) qualification required. Salary negotiable based on experience. Benefits include medical aid and pension. Apply through our website.",
    "Junior Data Analyst position. BCom or BSc degree required. Excel and SQL proficiency essential. Will be responsible for reporting and data management. Apply with full CV and references to jobs@analytics.co.za.",
    "Administrative Assistant required for Pretoria law firm. Minimum matric, 2 years admin experience. Duties include filing, correspondence and client liaison. R12,000 per month. Email applications to admin@lawfirm.co.za.",
    "Registered Nurse required for private hospital in Durban. SANC registration essential. 3 years clinical experience. Competitive package including medical aid. Human resources department will contact shortlisted candidates.",
    "Civil Engineer needed for infrastructure projects in Gauteng. BEng Civil required, ECSA registration preferred. 5 years experience. Project management skills advantageous. Salary: R60,000 - R80,000 CTC.",
    "Customer Service Representative for major retail chain. Matric required. Friendly, professional demeanor. R8,500 per month plus benefits. Apply in person at any store or email cv@retailchain.co.za.",
    "IT Support Technician - CompTIA A+ certification preferred. 2 years desktop support experience. Johannesburg based. R15,000 per month. Medical aid and retirement fund included. Full job description on company website.",
    "Financial Advisor position at registered FSP. RE5 and FAIS qualifications required. Commission plus basic salary structure. Must have own transport. Background checks and FICA verification will be conducted.",
    "Primary School Teacher required for reputable Bloemfontein school. SACE registered. Foundation phase experience preferred. Apply with certified copies of qualifications to principal@school.co.za.",
    "Logistics Coordinator position in Port Elizabeth. Matric plus relevant diploma. 3 years supply chain experience. Driver's license required. R18,000 per month. Email applications with references.",
    "Graphic Designer wanted for Johannesburg creative agency. Portfolio required. Adobe Creative Suite proficiency essential. 2 years commercial design experience. Salary based on experience. Office based position.",
    "Quantity Surveyor needed for construction company. BSc QS degree and 4 years site experience required. ASAQS membership advantageous. Competitive salary with benefits package offered.",
    "Call Centre Agent for established telecommunications company. Matric required. Previous call centre experience beneficial. R7,500 basic plus commission. Training provided. Apply online at careers portal.",
]

fraudulent_jobs = [
    "URGENT HIRING! Earn R5000 per week working from home! No experience needed! Just send R200 registration fee to confirm your placement. WhatsApp us immediately for instant job offer! Limited positions available!!!",
    "Work from home and earn R15,000 weekly! No qualifications required! Anyone can apply! Just deposit R500 joining fee via EFT and start earning immediately! Reply NOW to secure your position!",
    "Government job opportunity! We are recruiting on behalf of SASSA. Earn R8000 monthly. Send your ID copy, bank details and R350 processing fee to qualify. Positions filling fast! Contact us on WhatsApp.",
    "International company hiring South Africans! No experience required! Earn USD 500 daily! Send your personal details including ID number, banking details and passport copy to apply. Relocation package included!",
    "MAKE MONEY FAST! Data capture job from home. Earn R3000 per day! No experience needed! Just send your bank account details and R150 registration to get your login credentials. Start immediately!!!",
    "Mystery shopper opportunity! Earn R2500 per assignment! We will send you a cheque, deposit it and transfer the difference. No experience needed. Anyone qualify. Send personal and banking details to agent.",
    "Nanny and domestic worker positions available in USA and UK! Salary USD 3000 per month! We will process your visa! Just pay R1500 administration fee to our agent. Send ID and passport copy immediately.",
    "Earn money online liking Facebook posts and watching videos! No skills needed! Make R500 per hour! Pay R99 registration fee to join our exclusive programme. Thousands already earning! Join today!",
    "Job offer from mining company in Australia! High salary! Free accommodation! All expenses paid! Just need your personal details and passport copy for visa processing. Small admin fee of R2000 required.",
    "URGENT: Secretary needed immediately! R45000 monthly salary! No interview required! You are selected based on your profile! Send ID copy, banking details and R500 placement fee to confirm appointment.",
    "Earn R10,000 weekly stuffing envelopes at home! No experience required! Materials supplied! Just pay R300 for your starter kit and begin immediately! Flexible hours! Work whenever you want!",
    "Our company is recruiting agents in South Africa! Earn commission selling our products online! No stock needed! Pay R800 joining fee to access our system and start earning thousands monthly!",
    "CASHIER JOB AVAILABLE IMMEDIATELY! Salary R25000! No experience necessary! Send your bank account number, ID number and mother's maiden name to apply. Interview by WhatsApp only. Start Monday!",
    "Work for Amazon from home! Earn R20,000 per week reviewing products! No qualifications needed! Anyone can do this! Send R250 activation fee and your banking details to get started today!!!",
    "Government grant processing job! Help people apply for government money and earn R5000 per day! Send your SARS tax number and bank details to register. Free training provided after R400 deposit.",
]

# Build training data
texts = legitimate_jobs + fraudulent_jobs
labels = [0] * len(legitimate_jobs) + [1] * len(fraudulent_jobs)  # 0=legitimate, 1=fraudulent

# Add more variation
extra_legit = [
    "Experienced plumber required for residential and commercial work. Trade tested essential. Must have own tools. R200-R250 per hour. Email CVs to maintenance@propertygroup.co.za.",
    "Sales Representative for FMCG company. Matric plus valid driver's license. Own vehicle required. Basic salary plus commission. Fuel allowance provided. Territory: Gauteng South.",
    "Project Manager for IT company. PMP certification advantageous. 5 years PM experience required. Agile methodology knowledge essential. Competitive CTC package offered.",
] * 5

extra_fraud = [
    "EARN BIG FROM HOME! Retype documents and earn R1000 per page! No experience! Just pay R200 starter fee! WhatsApp us now for details! Limited spots remaining!",
    "Job guarantee! We place you in any company! Just pay R1000 placement fee! Positions in all sectors! Send ID and bank details! 100% success rate guaranteed!",
    "Online job for students! Earn while studying! Just need your student number and bank account! R2000 per week guaranteed! Pay R150 to register today!",
] * 5

texts += extra_legit + extra_fraud
labels += [0] * len(extra_legit) + [1] * len(extra_fraud)

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=1
    )),
    ('clf', RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        class_weight='balanced'
    ))
])

pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print("Model trained successfully.")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraudulent']))

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("Model saved to model.pkl")
