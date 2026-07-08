from typing import Dict, List, Any

# Medical Disclaimer showing on all predictions
MEDICAL_DISCLAIMER = (
    "Disclaimer: This AI-powered tool is built for educational and research purposes only. "
    "It does NOT constitute medical advice, diagnosis, or treatment. The findings generated "
    "by this model should not be used as a substitute for professional clinical judgment or "
    "consultation with a qualified healthcare provider. If you suspect you have a medical condition, "
    "please seek immediate medical attention from a licensed physician or dermatologist."
)

MEDICAL_INFO: Dict[str, Dict[str, Any]] = {
    "Oral_Cancer": {
        "title": "Oral Squamous Cell Carcinoma / Oral Malignancy",
        "description": (
            "Oral cancer refers to malignant growths occurring in the oral cavity, which includes the lips, "
            "tongue, inner lining of the cheeks (buccal mucosa), gums, floor of the mouth, and hard palate. "
            "The most common type is Oral Squamous Cell Carcinoma (OSCC), which accounts for over 90% of cases. "
            "It begins in the flat, thin squamous cells that line the mouth and lips."
        ),
        "symptoms": [
            "A sore, lump, or ulcer on the lip or in the mouth that does not heal within two weeks.",
            "White patches (leukoplakia) or red patches (erythroplakia) on the inner lining of the mouth or tongue.",
            "Unexplained bleeding, pain, or numbness in the mouth or lip.",
            "Difficulty chewing, swallowing, speaking, or moving the tongue or jaw.",
            "A lump or thickening in the neck or cheek.",
            "Loose teeth without an obvious dental cause."
        ],
        "risk_factors": [
            "Tobacco Use: Smoking cigarettes, cigars, pipes, or chewing smokeless tobacco/gutkha is the primary risk factor.",
            "Alcohol Consumption: Heavy alcohol intake combined with tobacco use synergistically multiplies the risk.",
            "Human Papillomavirus (HPV): Infection with high-risk HPV strains (particularly HPV-16) is associated with oropharyngeal cancers.",
            "Sun Exposure: Prolonged exposure to ultraviolet radiation increases the risk of lip cancers.",
            "Poor Oral Hygiene: Chronic irritation from sharp teeth, ill-fitting dentures, or general poor oral health.",
            "Dietary Deficiencies: A diet low in fruits, vegetables, and essential micronutrients."
        ],
        "preventive_measures": [
            "Avoid all forms of tobacco (cigarettes, vapes, chewing tobacco, gutkha, paan).",
            "Limit alcohol intake to moderate levels or avoid it entirely.",
            "Practice good oral hygiene: brush and floss daily, and visit a dentist regularly (at least twice a year).",
            "Protect your lips from UV rays using lip balms containing SPF 30 or higher and wide-brimmed hats.",
            "Eat a balanced diet rich in antioxidant vitamins (A, C, E) from fresh fruits and vegetables.",
            "Perform monthly self-examinations of your mouth (lips, tongue, gums, cheeks) to spot any changes early."
        ],
        "treatment_suggestions": [
            "Surgery: Primary choice to excise the tumor and, if necessary, remove affected lymph nodes (neck dissection).",
            "Radiation Therapy: High-energy X-rays used to destroy cancer cells, often post-surgery or combined with chemotherapy.",
            "Chemotherapy: Powerful drugs given systemically or locally to kill cancer cells, often for advanced or metastatic disease.",
            "Targeted Therapy: Drugs (e.g., Cetuximab) that target specific proteins or receptors on cancer cells to halt their growth.",
            "Immunotherapy: Helps the patient's own immune system recognize and fight cancer cells (e.g., Pembrolizumab)."
        ],
        "consultation_advice": (
            "You should immediately schedule an appointment with an Oral Pathologist, Oral and Maxillofacial Surgeon, "
            "or an Ear, Nose, and Throat (ENT) specialist. A tissue biopsy is the definitive method to confirm or "
            "rule out oral cancer. Do not delay professional evaluation."
        )
    },
    "Oral_Normal": {
        "title": "Normal / Healthy Oral Cavity",
        "description": (
            "The image analysis indicates a healthy, normal oral cavity. The mucosa appears pink, moist, and intact, "
            "without any suspicious leukoplakia, erythroplakia, persistent ulcers, or neoplastic lesions."
        ),
        "symptoms": [
            "No persistent pain, swelling, or bleeding.",
            "Smooth and uniform texture of the oral mucosa, tongue, and gums.",
            "Sores or minor bite injuries heal quickly (within a few days)."
        ],
        "risk_factors": [
            "Risk is low, but can increase due to sudden lifestyle changes (starting tobacco/alcohol use) or poor dental care."
        ],
        "preventive_measures": [
            "Maintain your current oral health routine (brushing twice daily and flossing).",
            "Schedule regular preventative dental cleanings and checkups.",
            "Continue avoiding tobacco products and heavy alcohol use.",
            "Keep up a nutrient-dense diet to support mucosal health."
        ],
        "treatment_suggestions": [
            "No active medical treatment required.",
            "Maintain routine dental checkups."
        ],
        "consultation_advice": (
            "No urgent action is needed. However, if you develop any sores, lumps, or red/white patches in the future "
            "that persist for more than 14 days, consult your dentist or primary physician immediately."
        )
    },
    "Skin_Cancer": {
        "title": "Malignant Skin Lesion (Melanoma or Carcinoma)",
        "description": (
            "Skin cancer is the abnormal, uncontrolled growth of skin cells. The most aggressive type is Melanoma, "
            "which originates in the pigment-producing melanocytes. Other common types include Basal Cell Carcinoma (BCC) "
            "and Squamous Cell Carcinoma (SCC), which are typically slower-growing but still require prompt treatment."
        ),
        "symptoms": [
            "ABCDE of Melanoma: Asymmetry, Border irregularity, Color variation, Diameter > 6mm, and Evolving/changing size or shape.",
            "A new mole or a change in an existing mole (e.g., bleeding, itching, oozing).",
            "A pearly, waxy bump, or a flat, flesh-colored scar-like lesion (common in Basal Cell Carcinoma).",
            "A firm, red nodule, or a flat lesion with a scaly, crusted surface (common in Squamous Cell Carcinoma).",
            "A sore that bleeds, scabs, heals, and then bleeds again."
        ],
        "risk_factors": [
            "Ultraviolet (UV) Exposure: Intense sun exposure, sunburn history, and tanning bed use.",
            "Fair Skin: Having less melanin provides less protection against UV radiation (though anyone can get skin cancer).",
            "Large Number of Moles: Having multiple atypical moles (dysplastic nevi) increases risk.",
            "Family or Personal History: Prior skin cancer increases susceptibility.",
            "Weakened Immune System: Immunosuppression due to medical conditions or medications.",
            "Age and Gender: Incidence increases with age, though melanoma is also common in young adults."
        ],
        "preventive_measures": [
            "Apply broad-spectrum sunscreen with an SPF of 30 or higher daily, even on cloudy days, and reapply every 2 hours.",
            "Seek shade during peak UV hours (10:00 AM to 4:00 PM).",
            "Wear protective clothing, including long-sleeved shirts, pants, wide-brimmed hats, and UV-blocking sunglasses.",
            "Avoid tanning beds and sun lamps completely.",
            "Perform a full-body skin self-exam monthly, checking hard-to-reach places (scalp, back, soles) with a mirror.",
            "Get a professional skin exam annually by a certified dermatologist."
        ],
        "treatment_suggestions": [
            "Surgical Excision: Removing the lesion along with a safe margin of healthy skin.",
            "Mohs Micrographic Surgery: Precise removal layer by layer, common for face/sensitive areas to preserve tissue.",
            "Cryosurgery: Freezing the lesion with liquid nitrogen (suitable for early, superficial lesions).",
            "Chemotherapy: Topical creams (e.g., 5-Fluorouracil) for superficial cancers, or systemic chemo for advanced stages.",
            "Targeted & Immunotherapy: Advanced therapies (e.g., Pembrolizumab, Nivolumab) that target genetic mutations or boost immune response."
        ],
        "consultation_advice": (
            "You should consult a board-certified Dermatologist immediately. A skin biopsy is required to diagnose "
            "the lesion accurately and guide the appropriate treatment path. Do not squeeze, scratch, or try to self-treat the lesion."
        )
    },
    "Skin_Normal": {
        "title": "Normal / Benign Skin Lesion or Healthy Skin",
        "description": (
            "The image analysis indicates healthy skin or a benign dermatological lesion (such as a normal nevus/mole, "
            "seborrheic keratosis, or freckle). There are no immediate signs of malignancy (irregular borders, "
            "asymmetrical coloration, or rapid evolution)."
        ),
        "symptoms": [
            "Symmetrical mole or spot with uniform color (typically tan, brown, or black).",
            "Distinct, smooth borders.",
            "No rapid changes in size, shape, color, or symptoms (itching, bleeding, pain)."
        ],
        "risk_factors": [
            "Risk is currently low, but general UV radiation and aging remain long-term risk factors for developing new lesions."
        ],
        "preventive_measures": [
            "Continue applying SPF 30+ sunscreen daily when outdoors.",
            "Wear sun-protective clothing and hats.",
            "Avoid tanning beds.",
            "Maintain monthly self-checks to ensure no existing spots start changing or new suspicious ones appear."
        ],
        "treatment_suggestions": [
            "No active medical treatment or intervention is necessary.",
            "Cosmetic removal can be discussed with a dermatologist if the spot is physically irritating."
        ],
        "consultation_advice": (
            "No urgent action is needed. Keep up with annual professional skin examinations. If you ever notice "
            "any changes in a mole (asymmetry, bleeding, uneven colors, size growth), contact a dermatologist for an evaluation."
        )
    },
    "Oral_Ulcer": {
        "title": "Oral Ulcer / Canker Sore (Benign / Non-Cancerous)",
        "description": (
            "An oral ulcer, commonly known as a mouth ulcer or canker sore, is a painful sore on the delicate tissue inside "
            "the mouth (lips, cheeks, tongue, or gums). Unlike oral cancer, oral ulcers are benign, non-infectious, and "
            "typically heal on their own within one to two weeks."
        ),
        "symptoms": [
            "Painful round or oval sore with a red border and a white, gray, or yellow center.",
            "Usually small (less than 1cm) and located on the inner cheek, lips, tongue, or gums.",
            "Difficulty eating, drinking, or speaking due to local irritation."
        ],
        "risk_factors": [
            "Minor mouth injury from accidental biting, rough brushing, or dental appliances.",
            "Stress, lack of sleep, or hormonal changes.",
            "Nutritional deficiencies, especially Vitamin B12, Iron, or Folate.",
            "Sensitivity to acidic or spicy foods."
        ],
        "preventive_measures": [
            "Avoid acidic, spicy, or very hot foods that irritate the mucosa.",
            "Use a soft-bristled toothbrush to prevent physical abrasion.",
            "Manage stress and maintain a balanced diet."
        ],
        "treatment_suggestions": [
            "Over-the-counter topical pain-relieving gels or creams.",
            "Warm salt-water rinses to reduce inflammation and promote healing.",
            "Avoiding irritating substances."
        ],
        "consultation_advice": (
            "Oral ulcers are benign and usually resolve within 10-14 days. If the ulcer does not heal after 2 weeks, "
            "is extremely large, or is accompanied by a high fever, consult a dentist or doctor to rule out other conditions."
        )
    }
}
