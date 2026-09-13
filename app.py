"""
Job Applicant Classification API
Flask backend using Orange3 pretrained models (.pkcls)
"""
import os
import pickle
import traceback
import numpy as np

from flask import Flask, request, jsonify, render_template

# ── Orange3 imports ───────────────────────────────────────────────────────────
try:
    import Orange
    from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable, StringVariable
    from Orange.data import Instance
    ORANGE_AVAILABLE = True
    print(f"[OK] Orange3 {Orange.__version__} loaded successfully.")
except ImportError as e:
    ORANGE_AVAILABLE = False
    print(f"[WARNING] Orange3 not available: {e}")

app = Flask(__name__)

# ── Model configuration ───────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "pretrained_models")

MODEL_FILES = {
    "Artificial Neural Network": "ANN.pkcls",
    "K-Nearest Neighbor":        "KNN.pkcls",
    "Support Vector Machine":    "SVM.pkcls",
    "Logistic Regression":       "LogisticRegression.pkcls",
}

# ── Feature Domain Definition ─────────────────────────────────────────────────
# Raw input domain — matches training data BEFORE preprocessing.
COUNTRIES = [
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan", 
    "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bhutan", "Bolivia", "Bosnia and Herzegovina", 
    "Botswana", "Brazil", "Bulgaria", "Burkina Faso", "Burundi", "Cambodia", "Cameroon", "Canada", "Cape Verde", "Chile", 
    "China", "Colombia", "Congo, Republic of the...", "Costa Rica", "Croatia", "Cuba", "Cyprus", "Czech Republic", "Côte d'Ivoire", 
    "Democratic Republic of the Congo", "Denmark", "Djibouti", "Dominica", "Dominican Republic", "Ecuador", "Egypt", "El Salvador", 
    "Estonia", "Ethiopia", "Fiji", "Finland", "France", "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Guatemala", 
    "Guinea", "Guyana", "Haiti", "Honduras", "Hong Kong (S.A.R.)", "Hungary", "Iceland", "India", "Indonesia", "Iran, Islamic Republic of...", 
    "Iraq", "Ireland", "Isle of Man", "Israel", "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kosovo", 
    "Kuwait", "Kyrgyzstan", "Lao People's Democratic Republic", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libyan Arab Jamahiriya", 
    "Lithuania", "Luxembourg", "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Mauritania", "Mauritius", 
    "Mexico", "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar", "Namibia", "Nepal", "Netherlands", 
    "New Zealand", "Nicaragua", "Niger", "Nigeria", "Nomadic", "Norway", "Oman", "Pakistan", "Palestine", "Panama", 
    "Paraguay", "Peru", "Philippines", "Poland", "Portugal", "Qatar", "Republic of Korea", "Republic of Moldova", "Romania", 
    "Russian Federation", "Rwanda", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Saudi Arabia", 
    "Senegal", "Serbia", "Seychelles", "Singapore", "Slovakia", "Slovenia", "Somalia", "South Africa", "South Korea", "Spain", 
    "Sri Lanka", "Sudan", "Suriname", "Swaziland", "Sweden", "Switzerland", "Syrian Arab Republic", "Taiwan", "Tajikistan", 
    "Thailand", "The former Yugoslav Republic of Macedonia", "Timor-Leste", "Togo", "Trinidad and Tobago", "Tunisia", "Turkey", 
    "Turkmenistan", "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom of Great Britain and Northern Ireland", 
    "United Republic of Tanzania", "United States of America", "Uruguay", "Uzbekistan", "Venezuela, Bolivarian Republic of...", 
    "Viet Nam", "Yemen", "Zambia", "Zimbabwe"
]

def build_domain():
    attrs = [
        DiscreteVariable("Age", values=["<35", ">35"]),
        DiscreteVariable("Accessibility", values=["No", "Yes"]),
        DiscreteVariable("EdLevel", values=["Master", "NoHigherEd", "Other", "PhD", "Undergraduate"]),
        DiscreteVariable("Employment", values=["0", "1"]),
        DiscreteVariable("Gender", values=["Man", "NonBinary", "Woman"]),
        DiscreteVariable("MentalHealth", values=["No", "Yes"]),
        DiscreteVariable("MainBranch", values=["Dev", "NotDev"]),
        ContinuousVariable("YearsCode"),
        ContinuousVariable("YearsCodePro"),
        ContinuousVariable("PreviousSalary"),
        ContinuousVariable("ComputerSkills"),
        DiscreteVariable("Country", values=COUNTRIES),
    ]
    metas = [StringVariable("HaveWorkedWith")]
    class_var = DiscreteVariable("Employed", values=["0", "1"])
    return Domain(attrs, class_vars=class_var, metas=metas)

DOMAIN = build_domain() if ORANGE_AVAILABLE else None

# ── Load models ───────────────────────────────────────────────────────────────
models = {}

def load_models():
    """Load all .pkcls model files from the pretrained_models directory."""
    for display_name, filename in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(path):
            print(f"[WARN] Model file not found: {path}")
            continue
        try:
            with open(path, "rb") as f:
                clf = pickle.load(f)
            models[display_name] = clf
            print(f"[OK] Loaded: {display_name} ({filename})")
        except Exception as exc:
            print(f"[ERROR] Failed to load '{display_name}': {exc}")
            traceback.print_exc()

load_models()

# ── Helper: convert raw dict → Orange Table ───────────────────────────────────
def dict_to_orange_table(raw: dict) -> "Table":
    """
    Convert a raw input dict to an Orange.data.Table row.
    Categorical (DiscreteVariable) values are stored as integer indices
    into var.values. Continuous values are stored as floats.
    String variable is stored as string.
    """
    row = []
    
    # Process regular attributes
    for var in DOMAIN.attributes:
        if isinstance(var, StringVariable):
            continue
            
        key = var.name
        val = raw.get(key)
        if val is None:
            raise ValueError(f"Missing required feature: '{key}'")

        if isinstance(var, ContinuousVariable):
            try:
                row.append(float(val))
            except (TypeError, ValueError):
                raise ValueError(f"'{key}' must be a number, got: {val!r}")
        else:
            str_val = str(val)
            if str_val not in var.values:
                raise ValueError(
                    f"Invalid value '{str_val}' for '{key}'. "
                    f"Expected one of: {list(var.values)}"
                )
            row.append(str_val)

    # Append a placeholder class value. 
    # Use the first valid class value to avoid Orange RuntimeWarnings about casting NaN/Unknown.
    row.append(DOMAIN.class_var.values[0])
    
    table = Table.from_list(DOMAIN, [row])
    
    # Process string variables (HaveWorkedWith)
    for var in DOMAIN.metas:
        if isinstance(var, StringVariable):
            key = var.name
            val = str(raw.get(key, "")).strip()
            # Prevent "?" from being parsed as Unknown by Orange.
            if val == "?":
                val = ""
            table[0, var] = val

    return table


# ── Helper: run prediction on one model ──────────────────────────────────────
def predict_with_model(clf, table: "Table") -> dict:
    """
    Run Orange classifier and return a result dict.
    """
    predictions, probs = clf(table, clf.ValueProbs)

    raw_pred = float(predictions[0])
    raw_prob0 = float(probs[0][0])
    raw_prob1 = float(probs[0][1])

    # Guard against NaN from models (e.g. when inputs produce unsupported feature combinations)
    import math
    pred_idx = int(raw_pred) if not math.isnan(raw_pred) else 0
    prob_not_hired = (raw_prob0 * 100) if not math.isnan(raw_prob0) else 0.0
    prob_hired     = (raw_prob1 * 100) if not math.isnan(raw_prob1) else 0.0

    return {
        "prediction": pred_idx,
        "label": "Employed" if pred_idx == 1 else "Not Employed",
        "prob_hired": round(prob_hired, 2),
        "prob_not_hired": round(prob_not_hired, 2),
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", model_names=list(MODEL_FILES.keys()), countries=COUNTRIES)


@app.route("/predict", methods=["POST"])
def predict():
    if not ORANGE_AVAILABLE:
        return jsonify({"error": "Orange3 library is not available on the server."}), 500
    if not models:
        return jsonify({"error": "No models loaded. Check server logs."}), 500

    payload = request.get_json(force=True, silent=True) or {}
    selected_model = payload.get("model", "").strip()
    features_raw = payload.get("features", {})

    if not selected_model:
        return jsonify({"error": "No model selected."}), 400
    if selected_model not in models:
        return jsonify({
            "error": f"Model '{selected_model}' not available.",
            "available_models": list(models.keys()),
        }), 400

    try:
        table = dict_to_orange_table(features_raw)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Data preparation failed: {exc}"}), 500

    try:
        result = predict_with_model(models[selected_model], table)
        result["model"] = selected_model
        return jsonify(result)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


@app.route("/models", methods=["GET"])
def list_models():
    status = {}
    for name in MODEL_FILES:
        status[name] = "loaded" if name in models else "failed to load"
    return jsonify({
        "orange_version": Orange.__version__ if ORANGE_AVAILABLE else "not installed",
        "models": status,
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "orange_available": ORANGE_AVAILABLE,
        "models_loaded": list(models.keys()),
    })


if __name__ == "__main__":
    print("=" * 60)
    print("  Job Applicant Classification API")
    print("=" * 60)
    print(f"  Models loaded  : {list(models.keys())}")
    print(f"  Starting at    : http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)
