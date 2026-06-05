============================================================
  FraudGuard SA — Setup and Run Instructions
  CUT PG Dip IT — Assignment 4 Prototype
============================================================

REQUIREMENTS
------------
Python 3.8 or higher must be installed on your computer.
Download Python from: https://www.python.org/downloads/

STEP 1 — Install required packages (run once only)
---------------------------------------------------
Open Command Prompt (Windows) or Terminal (Mac/Linux) and run:

    pip install flask scikit-learn pandas numpy

STEP 2 — Set up the folder structure
--------------------------------------
Create a folder called:  fraudguard_sa

Inside it, create a subfolder called:  templates

Place the files as follows:

    fraudguard_sa/
    ├── app.py
    ├── train_model.py
    ├── model.pkl          (already trained — skip Step 3)
    └── templates/
        └── index.html

STEP 3 — Train the model (optional — model.pkl already included)
-----------------------------------------------------------------
Only do this if model.pkl is missing. Run:

    python train_model.py

STEP 4 — Run the application
------------------------------
Navigate to the fraudguard_sa folder in your terminal and run:

    python app.py

You will see:
    FraudGuard SA running on http://localhost:5000

STEP 5 — Open the browser
---------------------------
Open any web browser and go to:

    http://localhost:5000

The FraudGuard SA interface will load. Paste a job posting
and click "Analyse Posting" to see the fraud risk result.

TAKING SCREENSHOTS FOR ASSIGNMENT
-----------------------------------
Take screenshots showing:
  1. The home screen with a job posting pasted in the text area
  2. A FRAUDULENT result (use the "Work-from-home scam" example)
  3. A SUSPICIOUS result
  4. A LEGITIMATE result (use the "Software developer post" example)
  5. The risk indicators panel

On Windows: Press Windows + Shift + S to take a screenshot.
On Mac: Press Command + Shift + 4.

============================================================
