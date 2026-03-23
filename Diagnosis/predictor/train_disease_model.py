

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
DEFAULT_SYMPTOMS_CSV  = ROOT / "data" / "symptoms_data.csv"
DEFAULT_AYURGENIX_CSV = ROOT / "data" / "AyurGenixAI_Dataset.csv"
DEFAULT_MODEL_PATH    = Path(__file__).parent / "disease_model.pkl"

# Ayurvedic fallbacks for diseases not in AyurGenix KB
FALLBACKS = {
    "(vertigo) paroymsal  positional vertigo": {
        "disease":"Vertigo (BPPV)","hindi":"चक्कर आना","doshas":"Vata, Kapha",
        "herbs":"Ashwagandha, Brahmi, Shankhapushpi",
        "formulation":"Brahmi ghee (1 tsp daily); Ashwagandha powder (3g) with warm milk",
        "diet":"Avoid cold, heavy foods; reduce salt; eat warm light meals.",
        "yoga":"Epley manoeuvre; gentle neck stretches; Shavasana; Nadi Shodhana pranayama",
        "prevention":"Avoid sudden head movements; stay hydrated; manage stress",
        "severity":"Moderate","prognosis":"Often resolves with repositioning manoeuvres",
        "complications":"Falls, anxiety, chronic dizziness","symptoms_text":"Dizziness, spinning sensation, nausea, vomiting",
    },
    "aids": {
        "disease":"AIDS / HIV","hindi":"एड्स","doshas":"Vata, Pitta",
        "herbs":"Ashwagandha, Guduchi (Giloy), Amalaki, Tulsi",
        "formulation":"Guduchi sattva (500mg twice daily); Chyawanprash (1 tsp daily)",
        "diet":"High-protein, nutrient-dense diet; avoid raw/uncooked food; stay hydrated.",
        "yoga":"Gentle yoga, pranayama, meditation to strengthen immunity and reduce stress",
        "prevention":"Safe sex practices, sterile needles, antiretroviral therapy",
        "severity":"Severe","prognosis":"Managed with antiretroviral therapy",
        "complications":"Opportunistic infections, wasting syndrome","symptoms_text":"Fatigue, weight loss, frequent infections, fever",
    },
    "alcoholic hepatitis": {
        "disease":"Alcoholic Hepatitis","hindi":"मद्यज यकृतशोथ","doshas":"Pitta, Kapha",
        "herbs":"Bhumyamalaki, Kutki, Punarnava, Kalmegh",
        "formulation":"Arogyavardhini vati (500mg twice daily); Kutki powder (1g) with honey",
        "diet":"No alcohol; high-protein diet; avoid fatty, fried, spicy foods.",
        "yoga":"Gentle forward folds, Shavasana; avoid strenuous exercise",
        "prevention":"Abstain from alcohol; hepatitis vaccination; healthy diet",
        "severity":"Severe","prognosis":"Depends on alcohol cessation",
        "complications":"Liver cirrhosis, liver failure","symptoms_text":"Jaundice, abdominal pain, nausea, fever, fatigue",
    },
    "allergy": {
        "disease":"Allergy","hindi":"एलर्जी","doshas":"Vata, Kapha",
        "herbs":"Haridra (Turmeric), Tulsi, Shirish, Neem",
        "formulation":"Haridra khand (3g) with warm milk twice daily; Tulsi kadha daily",
        "diet":"Avoid known allergens; eat anti-inflammatory foods; avoid cold, damp foods.",
        "yoga":"Pranayama (Anulom Vilom), Kapalbhati to clear airways",
        "prevention":"Identify and avoid triggers; keep environment clean; improve immunity",
        "severity":"Mild to Moderate","prognosis":"Manageable with avoidance and treatment",
        "complications":"Anaphylaxis, asthma","symptoms_text":"Sneezing, runny nose, itching, rashes, watery eyes",
    },
    "bronchial asthma": {
        "disease":"Bronchial Asthma","hindi":"दमा","doshas":"Vata, Kapha",
        "herbs":"Vasaka (Adhatoda), Pushkarmool, Pippali, Kantakari",
        "formulation":"Sitopaladi churna (3g) with honey twice daily; Vasaka swaras (10ml)",
        "diet":"Avoid cold, heavy, dairy-rich foods; eat warm, light meals; ginger tea.",
        "yoga":"Pranayama (Anulom Vilom, Bhramari), Setu Bandhasana; avoid Kapalabhati",
        "prevention":"Avoid triggers (dust, smoke, allergens); maintain warm environment",
        "severity":"Moderate to Severe","prognosis":"Manageable with lifestyle and medication",
        "complications":"Status asthmaticus, respiratory failure","symptoms_text":"Wheezing, breathlessness, chest tightness, cough",
    },
    "cervical spondylosis": {
        "disease":"Cervical Spondylosis","hindi":"ग्रीवा संधिवात","doshas":"Vata",
        "herbs":"Guggulu, Shallaki (Boswellia), Dashmool, Rasna",
        "formulation":"Yograj guggulu (500mg twice daily); Dashmool oil for local massage",
        "diet":"Warm, easily digestible foods; avoid cold, dry foods; sesame seeds and ghee.",
        "yoga":"Gentle neck stretches, Matsyasana, Bhujangasana; avoid forward neck bending",
        "prevention":"Correct posture, ergonomic workspace, regular neck exercises",
        "severity":"Moderate","prognosis":"Manageable with physiotherapy and lifestyle",
        "complications":"Myelopathy, radiculopathy","symptoms_text":"Neck pain, stiffness, headache, arm numbness",
    },
    "chicken pox": {
        "disease":"Chickenpox","hindi":"छोटी माता","doshas":"Pitta, Kapha",
        "herbs":"Neem, Haridra, Chandana (Sandalwood), Manjishtha",
        "formulation":"Neem paste externally; Haridra milk (1 tsp turmeric in warm milk) twice daily",
        "diet":"Light, easily digestible foods; avoid spicy, fried foods; plenty of fluids.",
        "yoga":"Complete rest; gentle breathing exercises post-recovery",
        "prevention":"Varicella vaccination; avoid contact with infected persons",
        "severity":"Mild to Moderate","prognosis":"Self-limiting, resolves in 1-2 weeks",
        "complications":"Secondary bacterial infection, pneumonia, encephalitis","symptoms_text":"Itchy blisters, fever, fatigue, headache",
    },
    "chronic cholestasis": {
        "disease":"Chronic Cholestasis","hindi":"पित्त अवरोध","doshas":"Pitta, Kapha",
        "herbs":"Kutki, Bhumyamalaki, Punarnava, Triphala",
        "formulation":"Arogyavardhini vati (250mg twice daily); Triphala churna (3g) at bedtime",
        "diet":"Low-fat diet; avoid alcohol, fried foods; increase fibre intake.",
        "yoga":"Gentle twisting poses, Pawanmuktasana; avoid strenuous exercise",
        "prevention":"Avoid hepatotoxic drugs; limit alcohol; healthy diet",
        "severity":"Moderate to Severe","prognosis":"Depends on underlying cause",
        "complications":"Cirrhosis, liver failure","symptoms_text":"Jaundice, itching, fatigue, pale stools, dark urine",
    },
    "dengue": {
        "disease":"Dengue Fever","hindi":"डेंगू बुखार","doshas":"Pitta, Vata",
        "herbs":"Papaya leaf extract, Guduchi, Tulsi, Ginger",
        "formulation":"Papaya leaf juice (30ml twice daily); Guduchi sattva (500mg twice daily)",
        "diet":"Light, easily digestible foods; plenty of fluids; coconut water; avoid spicy food.",
        "yoga":"Complete bed rest; gentle breathing post-recovery",
        "prevention":"Mosquito control, repellents, eliminate stagnant water",
        "severity":"Moderate to Severe","prognosis":"Most recover in 1-2 weeks with supportive care",
        "complications":"Dengue hemorrhagic fever, shock syndrome","symptoms_text":"High fever, severe headache, joint pain, rash, bleeding",
    },
    "dimorphic hemmorhoids(piles)": {
        "disease":"Piles (Haemorrhoids)","hindi":"बवासीर","doshas":"Vata, Pitta",
        "herbs":"Haritaki, Nagkesar, Kutaja, Triphala",
        "formulation":"Arshkuthar ras (250mg twice daily); Triphala churna (5g) at bedtime",
        "diet":"High fibre diet; plenty of water; avoid spicy, fried foods; prunes, figs.",
        "yoga":"Ashwini mudra, Moolabandha, gentle inversions; avoid straining",
        "prevention":"High fibre diet, adequate hydration, avoid prolonged sitting",
        "severity":"Mild to Moderate","prognosis":"Good with dietary and lifestyle changes",
        "complications":"Anaemia, thrombosis, prolapse","symptoms_text":"Rectal bleeding, pain, itching, swelling around anus",
    },
    "drug reaction": {
        "disease":"Drug Reaction","hindi":"दवा प्रतिक्रिया","doshas":"Pitta, Vata",
        "herbs":"Neem, Haridra, Manjishtha, Guduchi",
        "formulation":"Guduchi sattva (500mg); Neem capsules (500mg twice daily)",
        "diet":"Light detox diet; plenty of water; avoid processed, heavy foods.",
        "yoga":"Gentle pranayama; Shavasana for relaxation",
        "prevention":"Always inform doctors of allergies; avoid self-medication",
        "severity":"Mild to Severe","prognosis":"Usually resolves once causative drug is stopped",
        "complications":"Anaphylaxis, Stevens-Johnson syndrome","symptoms_text":"Skin rash, itching, fever, swelling, difficulty breathing",
    },
    "fungal infection": {
        "disease":"Fungal Infection","hindi":"फंगल संक्रमण","doshas":"Kapha, Pitta",
        "herbs":"Neem, Haridra (Turmeric), Daruhaldi, Karanja",
        "formulation":"Neem oil topically; Haridra paste externally; Khadirarishta (15ml twice daily)",
        "diet":"Avoid sugar, refined carbohydrates, fermented foods; eat anti-fungal foods like garlic.",
        "yoga":"Keep affected areas dry; maintain hygiene; gentle yoga for immunity",
        "prevention":"Keep skin dry and clean; avoid sharing personal items; breathable clothing",
        "severity":"Mild to Moderate","prognosis":"Responds well to antifungal treatment",
        "complications":"Systemic infection in immunocompromised patients","symptoms_text":"Itching, rash, redness, scaling, ring-shaped lesions",
    },
    "gerd": {
        "disease":"GERD","hindi":"अम्ल प्रतिवाह","doshas":"Pitta",
        "herbs":"Shatavari, Licorice (Mulethi), Amalaki, Fennel",
        "formulation":"Amalaki churna (3g) with honey; Licorice root tea before meals",
        "diet":"Avoid spicy, acidic, fried foods; eat smaller meals; avoid eating 3hrs before bed.",
        "yoga":"Vajrasana after meals; elevate head while sleeping",
        "prevention":"Maintain healthy weight; avoid trigger foods; quit smoking",
        "severity":"Mild to Moderate","prognosis":"Manageable with lifestyle modifications",
        "complications":"Oesophagitis, Barrett's oesophagus","symptoms_text":"Heartburn, acid reflux, chest pain, regurgitation",
    },
    "heart attack": {
        "disease":"Heart Attack","hindi":"दिल का दौरा","doshas":"Vata, Pitta",
        "herbs":"Arjuna, Pushkarmool, Guggulu, Ashwagandha",
        "formulation":"Arjuna ksheerpaka (Arjuna bark 10g boiled in 400ml milk/water); Hridayarnava rasa (250mg)",
        "diet":"Low fat, low sodium diet; avoid red meat, fried foods; eat heart-healthy foods.",
        "yoga":"Cardiac rehab yoga; gentle pranayama; only under medical supervision",
        "prevention":"Healthy diet, exercise, no smoking, manage blood pressure and cholesterol",
        "severity":"Severe","prognosis":"Depends on prompt medical treatment",
        "complications":"Heart failure, arrhythmia, death","symptoms_text":"Chest pain, shortness of breath, sweating, arm pain, nausea",
    },
    "hepatitis d": {
        "disease":"Hepatitis D","hindi":"हेपेटाइटिस डी","doshas":"Pitta",
        "herbs":"Bhumyamalaki, Kutki, Kalmegh, Punarnava",
        "formulation":"Kutki powder (1g) with honey twice daily; Liv-52 tablet",
        "diet":"No alcohol; low-fat, high-protein diet; avoid processed and spicy foods.",
        "yoga":"Gentle liver-supporting poses: Ardha Matsyendrasana, Paschimottanasana",
        "prevention":"Hepatitis B vaccination (prevents D); safe practices",
        "severity":"Severe","prognosis":"Worse than Hepatitis B alone; requires specialist care",
        "complications":"Cirrhosis, liver failure","symptoms_text":"Jaundice, fatigue, abdominal pain, nausea, dark urine",
    },
    "hepatitis e": {
        "disease":"Hepatitis E","hindi":"हेपेटाइटिस ई","doshas":"Pitta",
        "herbs":"Bhumyamalaki, Punarnava, Amalaki, Guduchi",
        "formulation":"Bhumyamalaki powder (3g) twice daily; Punarnava swaras (10ml) morning",
        "diet":"No alcohol; light, easily digestible food; plenty of fluids.",
        "yoga":"Gentle restorative yoga; adequate rest",
        "prevention":"Safe drinking water; good sanitation; avoid undercooked meat",
        "severity":"Moderate","prognosis":"Usually self-limiting in 4-6 weeks",
        "complications":"Fulminant hepatitis in pregnancy","symptoms_text":"Jaundice, fatigue, nausea, abdominal pain, fever",
    },
    "impetigo": {
        "disease":"Impetigo","hindi":"इम्पेटिगो","doshas":"Pitta, Kapha",
        "herbs":"Neem, Haridra, Manjishtha, Lodhra",
        "formulation":"Neem paste externally; Haridra + Coconut oil topically",
        "diet":"Avoid sugar and processed foods; eat immunity-boosting foods; plenty of fluids.",
        "yoga":"Maintain hygiene; gentle yoga for immunity",
        "prevention":"Good hygiene; avoid touching sores; wash hands frequently",
        "severity":"Mild","prognosis":"Resolves within 2-3 weeks with treatment",
        "complications":"Cellulitis, post-streptococcal glomerulonephritis","symptoms_text":"Red sores, blisters, honey-coloured crusts, itching",
    },
    "jaundice": {
        "disease":"Jaundice","hindi":"पीलिया","doshas":"Pitta",
        "herbs":"Bhumyamalaki, Punarnava, Kutki, Kalmegh",
        "formulation":"Bhumyamalaki powder (3g) twice daily with water; Punarnava mandura (250mg)",
        "diet":"Light, easily digestible foods; plenty of fluids; sugarcane juice; avoid alcohol and fat.",
        "yoga":"Gentle liver poses: Jathara Parivartanasana; complete rest in acute phase",
        "prevention":"Safe drinking water, hepatitis vaccination, avoid alcohol",
        "severity":"Moderate","prognosis":"Depends on underlying cause",
        "complications":"Liver damage, anaemia","symptoms_text":"Yellow skin/eyes, dark urine, pale stools, fatigue, abdominal pain",
    },
    "osteoarthristis": {
        "disease":"Osteoarthritis","hindi":"अस्थिसंधिशोथ","doshas":"Vata",
        "herbs":"Shallaki (Boswellia), Guggulu, Ashwagandha, Nirgundi",
        "formulation":"Yograj guggulu (500mg twice daily); Mahanarayan oil for local massage",
        "diet":"Anti-inflammatory diet; turmeric with warm milk; avoid cold, dry foods; ghee.",
        "yoga":"Gentle joint exercises, Tadasana, Virabhadrasana; warm oil massage before yoga",
        "prevention":"Healthy weight, regular low-impact exercise, joint protection",
        "severity":"Moderate","prognosis":"Progressive but manageable",
        "complications":"Joint deformity, disability","symptoms_text":"Joint pain, stiffness, swelling, reduced range of motion",
    },
    "paralysis (brain hemorrhage)": {
        "disease":"Paralysis (Brain Hemorrhage)","hindi":"लकवा","doshas":"Vata",
        "herbs":"Ashwagandha, Bala, Dashmool, Brahmi",
        "formulation":"Mahayogaraj guggulu (500mg); Bala oil massage; Brahmi ghee (1 tsp daily)",
        "diet":"Easy-to-swallow, nourishing foods; warm soups; avoid cold and dry foods.",
        "yoga":"Passive joint movements; under physiotherapist guidance; pranayama for brain health",
        "prevention":"Control blood pressure; healthy lifestyle; avoid smoking and alcohol",
        "severity":"Severe","prognosis":"Rehabilitation-dependent; variable recovery",
        "complications":"Permanent disability, aspiration pneumonia","symptoms_text":"Sudden weakness, speech difficulty, facial drooping, headache",
    },
    "peptic ulcer diseae": {
        "disease":"Peptic Ulcer","hindi":"पेप्टिक अल्सर","doshas":"Pitta",
        "herbs":"Shatavari, Licorice (Mulethi), Amalaki, Yashtimadhu",
        "formulation":"Shatavari churna (3g) with milk twice daily; Yashtimadhu powder (2g) before meals",
        "diet":"Avoid spicy, acidic, fried foods; eat small frequent meals; cold milk; avoid NSAIDs.",
        "yoga":"Vajrasana, Shavasana; avoid Kapalbhati",
        "prevention":"Avoid NSAIDs, smoking, alcohol; manage stress; treat H. pylori",
        "severity":"Moderate","prognosis":"Good with treatment and dietary changes",
        "complications":"Bleeding, perforation, obstruction","symptoms_text":"Burning stomach pain, nausea, bloating, dark stools",
    },
    "typhoid": {
        "disease":"Typhoid","hindi":"टाइफाइड","doshas":"Pitta, Kapha",
        "herbs":"Guduchi, Nimba (Neem), Kutki, Sudarshan churna",
        "formulation":"Sudarshan churna (3g) twice daily; Guduchi sattva (500mg) twice daily",
        "diet":"Light, easily digestible foods; khichdi, soups; plenty of fluids.",
        "yoga":"Complete bed rest in acute phase; gentle recovery yoga post-fever",
        "prevention":"Safe food and water; typhoid vaccination; good hygiene",
        "severity":"Moderate to Severe","prognosis":"Good with antibiotic treatment",
        "complications":"Intestinal perforation, encephalitis","symptoms_text":"Sustained fever, headache, abdominal pain, weakness, rash",
    },
    "urinary tract infection": {
        "disease":"Urinary Tract Infection (UTI)","hindi":"मूत्र मार्ग संक्रमण","doshas":"Pitta",
        "herbs":"Gokshura, Varuna, Punarnava, Chandana (Sandalwood)",
        "formulation":"Chandraprabha vati (500mg twice daily); Gokshuradi guggulu (500mg twice daily)",
        "diet":"Plenty of water; cranberry juice; avoid spicy, sour, alcohol; eat cooling foods.",
        "yoga":"Baddha Konasana, Viparita Karani; pelvic floor exercises",
        "prevention":"Stay hydrated; urinate after intercourse; maintain hygiene",
        "severity":"Mild to Moderate","prognosis":"Responds well to treatment",
        "complications":"Kidney infection (pyelonephritis)","symptoms_text":"Burning urination, frequent urination, cloudy urine, pelvic pain",
    },
}


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def pivot_to_binary(rows: list[dict]) -> tuple[list[str], np.ndarray, list[str]]:
    """Convert wide-format symptom rows to binary feature matrix."""
    sym_cols = [c for c in rows[0].keys() if c != "Disease"]
    all_syms: set[str] = set()
    for r in rows:
        for c in sym_cols:
            v = r[c].strip()
            if v:
                all_syms.add(v)
    vocab = sorted(all_syms)
    vidx  = {s: i for i, s in enumerate(vocab)}

    X, y = [], []
    for r in rows:
        vec = [0] * len(vocab)
        for c in sym_cols:
            v = r[c].strip()
            if v and v in vidx:
                vec[vidx[v]] = 1
        X.append(vec)
        y.append(r["Disease"].strip())
    return vocab, np.array(X), y


def build_ayur_kb(rows: list[dict]) -> dict:
    kb = {}
    for r in rows:
        name = r["Disease"].strip()
        key  = name.lower()
        if key not in kb:
            kb[key] = {
                "disease":       name,
                "hindi":         r["Hindi Name"].strip(),
                "symptoms_text": r["Symptoms"].strip(),
                "doshas":        r["Doshas"].strip(),
                "herbs":         r["Ayurvedic Herbs"].strip(),
                "formulation":   r["Formulation"].strip(),
                "diet":          r["Diet and Lifestyle Recommendations"].strip(),
                "yoga":          r["Yoga & Physical Therapy"].strip(),
                "prevention":    r["Prevention"].strip(),
                "severity":      r["Symptom Severity"].strip(),
                "prognosis":     r["Prognosis"].strip(),
                "complications": r["Complications"].strip(),
            }
    return kb


def train(
    symptoms_path : Path = DEFAULT_SYMPTOMS_CSV,
    ayurgenix_path: Path = DEFAULT_AYURGENIX_CSV,
    model_path    : Path = DEFAULT_MODEL_PATH,
    n_estimators  : int  = 200,
    test_size     : float = 0.2,
    random_state  : int  = 42,
) -> None:

    # ── AyurGenix KB ─────────────────────────────────────────────────────────
    print(f"Loading AyurGenix KB from: {ayurgenix_path}")
    ayur_kb = build_ayur_kb(load_csv(ayurgenix_path))
    for key, entry in FALLBACKS.items():
        if key not in ayur_kb:
            ayur_kb[key] = entry
    print(f"  {len(ayur_kb)} diseases in knowledge base")

    # ── Symptom dataset ───────────────────────────────────────────────────────
    print(f"\nLoading symptoms data from: {symptoms_path}")
    sym_rows = load_csv(symptoms_path)
    vocab, X, y_raw = pivot_to_binary(sym_rows)
    diseases = sorted(set(y_raw))
    print(f"  {len(sym_rows)} rows | {len(vocab)} symptoms | {len(diseases)} diseases")

    le = LabelEncoder()
    y  = le.fit_transform(y_raw)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\nTraining RandomForest (n_estimators={n_estimators})…")
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    print(f"Test Accuracy: {accuracy_score(y_te, y_pred)*100:.2f}%\n")
    print(classification_report(y_te, y_pred, target_names=le.classes_))

    print("Top 10 most important symptoms:")
    for sym, imp in sorted(zip(vocab, clf.feature_importances_), key=lambda x: -x[1])[:10]:
        print(f"  {imp:.4f}  {sym}")

    print("\nKB coverage for classifier diseases:")
    for d in diseases:
        found = d.lower() in ayur_kb
        print(f"  {'✓' if found else '✗'}  {d}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({
            "clf":          clf,
            "label_enc":    le,
            "symptom_cols": vocab,
            "ayur_kb":      ayur_kb,
        }, f)
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Ayurvedic disease detection model.")
    parser.add_argument("--symptoms",  type=Path, default=DEFAULT_SYMPTOMS_CSV)
    parser.add_argument("--ayurgenix", type=Path, default=DEFAULT_AYURGENIX_CSV)
    parser.add_argument("--model",     type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--trees",     type=int,  default=200)
    parser.add_argument("--test-size", type=float,default=0.2)
    parser.add_argument("--seed",      type=int,  default=42)
    args = parser.parse_args()
    train(
        symptoms_path  = args.symptoms,
        ayurgenix_path = args.ayurgenix,
        model_path     = args.model,
        n_estimators   = args.trees,
        test_size      = args.test_size,
        random_state   = args.seed,
    )
