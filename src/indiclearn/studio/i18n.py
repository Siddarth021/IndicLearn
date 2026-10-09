"""Tutor-side strings in English, Telugu, Hindi and Sanskrit + multilingual intent keywords.

The Telugu / Hindi strings were written for this project; the Sanskrit strings are deliberately
simple.  All of them are plain data so a language expert can review and correct them without
touching code (see ``docs/MULTILINGUAL.md``).
"""

from __future__ import annotations

from indiclearn.studio.lang import LANG_NAMES, SUPPORTED_LANGS, match_key

T: dict[str, dict[str, str]] = {
    "en": {
        "q_cloze": "Fill in the blank",
        "q_def": "Which description fits “{term}”?",
        "q_tf": "True or false?",
        "true": "True",
        "false": "False",
        "correct": "Correct!",
        "incorrect": "Not quite.",
        "answer_was": "Answer",
        "explained": "From your notes",
        "quiz_done": "Quiz complete: {k} of {n} correct.",
        "greeting": "Hi! I'm your study tutor. Ask me anything about “{title}” — in English, తెలుగు, हिन्दी or संस्कृतम्. I can explain concepts, give examples, summarise a topic and quiz you.",
        "definition": "Definition", "from_notes": "From your notes", "example": "Example", "formula": "Formula", "key_points": "Key points",
        "related": "Related ideas", "review_first": "Review first", "next_up": "Next in the flow", "page": "p.", "covers": "This part covers",
        "not_found": "I couldn't find that in your uploaded notes, so I won't guess. I've added it to your doubts on the map — try rephrasing, or upload the chapter that covers it.",
        "closest": "Closest passage I found", "doubt_saved": "Added to your map as a doubt", "map_updated": "Your map was updated",
        "quiz_intro": "Here's a {n}-question quiz on {topic}. Every answer updates your knowledge map.",
        "quiz_none": "I couldn't build good quiz questions from this part yet — it needs clearer definitions. Try another topic.",
        "summary_title": "Summary", "study_next": "I'd study these next", "simple": "In simple words",
        "src_lang_note": "Your notes are in {lang}; the passage is shown as written.", "translated_note": "machine-translated from {lang}",
        "no_progress": "You haven't started yet — ask a question or take a quiz and I'll build your personal map.",
        "reason_weak_quiz": "you missed questions on it", "reason_asked_doubt": "you asked about it", "reason_not_started": "not started yet",
        "reason_keep_practising": "keep practising", "reason_prereq_for": "needed before {for_label}", "chapters": "Chapters",
        "low_grounding": "Check this against your notes — the answer may go beyond them.",
        "unit_Page": "page", "unit_Slide": "slide", "unit_Part": "part",
    },
    "te": {
        "q_cloze": "ఖాళీని పూరించండి",
        "q_def": "“{term}”కి ఏ వివరణ సరిపోతుంది?",
        "q_tf": "నిజమా కాదా?",
        "true": "నిజం",
        "false": "కాదు",
        "correct": "సరైనది!",
        "incorrect": "కాదు.",
        "answer_was": "సరైన సమాధానం",
        "explained": "మీ నోట్స్ నుండి",
        "quiz_done": "క్విజ్ పూర్తయింది: {n}లో {k} సరైనవి.",
        "greeting": "నమస్కారం! నేను మీ స్టడీ ట్యూటర్‌ని. “{title}” గురించి ఏదైనా అడగండి — తెలుగు, English, हिन्दी లేదా संस्कृतम्‌లో. భావనలను వివరించగలను, ఉదాహరణలు ఇవ్వగలను, సారాంశం చెప్పగలను, క్విజ్ పెట్టగలను.",
        "definition": "నిర్వచనం", "from_notes": "మీ నోట్స్ నుండి", "example": "ఉదాహరణ", "formula": "సూత్రం", "key_points": "ముఖ్యాంశాలు",
        "related": "సంబంధిత అంశాలు", "review_first": "ముందు ఇవి చదవండి", "next_up": "తర్వాతి అంశం", "page": "పేజీ", "covers": "ఈ భాగంలో ఉన్నవి",
        "not_found": "మీరు అప్‌లోడ్ చేసిన నోట్స్‌లో ఇది కనిపించలేదు, కాబట్టి ఊహించి చెప్పను. దీన్ని మ్యాప్‌లో మీ సందేహాల్లో చేర్చాను — వేరే విధంగా అడగండి లేదా సంబంధిత అధ్యాయాన్ని అప్‌లోడ్ చేయండి.",
        "closest": "నాకు దొరికిన అత్యంత దగ్గరి భాగం", "doubt_saved": "మీ మ్యాప్‌లో సందేహంగా చేర్చాను", "map_updated": "మీ మ్యాప్ అప్‌డేట్ అయింది",
        "quiz_intro": "{topic} పై {n} ప్రశ్నల క్విజ్ ఇదిగో. ప్రతి సమాధానం మీ నాలెడ్జ్ మ్యాప్‌ను అప్‌డేట్ చేస్తుంది.",
        "quiz_none": "ఈ భాగం నుండి మంచి క్విజ్ ప్రశ్నలు ఇంకా తయారు చేయలేకపోయాను — స్పష్టమైన నిర్వచనాలు కావాలి. మరొక అంశం ప్రయత్నించండి.",
        "summary_title": "సారాంశం", "study_next": "తర్వాత వీటిని చదవమని సూచిస్తాను", "simple": "సరళంగా చెప్పాలంటే",
        "src_lang_note": "మీ నోట్స్ {lang}లో ఉన్నాయి; భాగాన్ని అలాగే చూపిస్తున్నాను.", "translated_note": "{lang} నుండి యంత్ర అనువాదం",
        "no_progress": "మీరు ఇంకా ప్రారంభించలేదు — ప్రశ్న అడగండి లేదా క్విజ్ రాయండి, మీ వ్యక్తిగత మ్యాప్ తయారవుతుంది.",
        "reason_weak_quiz": "ఇందులో ప్రశ్నలు తప్పాయి", "reason_asked_doubt": "దీని గురించి అడిగారు", "reason_not_started": "ఇంకా మొదలుపెట్టలేదు",
        "reason_keep_practising": "ఇంకా సాధన చేయండి", "reason_prereq_for": "{for_label} కంటే ముందు అవసరం", "chapters": "అధ్యాయాలు",
        "low_grounding": "ఈ సమాధానాన్ని మీ నోట్స్‌తో సరిచూడండి — ఇది నోట్స్‌కు మించి ఉండవచ్చు.",
        "unit_Page": "పేజీ", "unit_Slide": "స్లైడ్", "unit_Part": "భాగం",
    },
    "hi": {
        "q_cloze": "रिक्त स्थान भरें",
        "q_def": "“{term}” के लिए कौन-सा वर्णन सही है?",
        "q_tf": "सही या गलत?",
        "true": "सही",
        "false": "गलत",
        "correct": "सही!",
        "incorrect": "सही नहीं।",
        "answer_was": "सही उत्तर",
        "explained": "आपके नोट्स से",
        "quiz_done": "क्विज़ पूरी: {n} में से {k} सही।",
        "greeting": "नमस्ते! मैं आपका स्टडी ट्यूटर हूँ। “{title}” के बारे में कुछ भी पूछें — हिन्दी, English, తెలుగు या संस्कृतम् में। मैं अवधारणाएँ समझा सकता हूँ, उदाहरण दे सकता हूँ, सारांश बता सकता हूँ और क्विज़ ले सकता हूँ।",
        "definition": "परिभाषा", "from_notes": "आपके नोट्स से", "example": "उदाहरण", "formula": "सूत्र", "key_points": "मुख्य बिंदु",
        "related": "संबंधित विषय", "review_first": "पहले इन्हें दोहराएँ", "next_up": "अगला विषय", "page": "पृष्ठ", "covers": "इस भाग में",
        "not_found": "यह आपके अपलोड किए नोट्स में नहीं मिला, इसलिए मैं अंदाज़ा नहीं लगाऊँगा। इसे मैंने मैप में आपकी शंकाओं में जोड़ दिया है — दूसरे शब्दों में पूछें या संबंधित अध्याय अपलोड करें।",
        "closest": "मुझे मिला सबसे निकट अंश", "doubt_saved": "आपके मैप में शंका के रूप में जोड़ा गया", "map_updated": "आपका मैप अपडेट हुआ",
        "quiz_intro": "{topic} पर {n} प्रश्नों की क्विज़ यह रही। हर उत्तर आपके नॉलेज मैप को अपडेट करता है।",
        "quiz_none": "इस भाग से अभी अच्छे क्विज़ प्रश्न नहीं बन सके — इसमें स्पष्ट परिभाषाएँ चाहिए। कोई और विषय आज़माएँ।",
        "summary_title": "सारांश", "study_next": "मैं आगे इन्हें पढ़ने का सुझाव दूँगा", "simple": "सरल शब्दों में",
        "src_lang_note": "आपके नोट्स {lang} में हैं; अंश जैसा लिखा है वैसा दिखा रहा हूँ।", "translated_note": "{lang} से मशीन अनुवाद",
        "no_progress": "आपने अभी शुरू नहीं किया — कोई प्रश्न पूछें या क्विज़ दें, मैं आपका निजी मैप बना दूँगा।",
        "reason_weak_quiz": "इसमें प्रश्न गलत हुए", "reason_asked_doubt": "आपने इसके बारे में पूछा", "reason_not_started": "अभी शुरू नहीं हुआ",
        "reason_keep_practising": "अभ्यास जारी रखें", "reason_prereq_for": "{for_label} से पहले ज़रूरी", "chapters": "अध्याय",
        "low_grounding": "इसे अपने नोट्स से मिलाकर देखें — उत्तर नोट्स से आगे जा सकता है।",
        "unit_Page": "पृष्ठ", "unit_Slide": "स्लाइड", "unit_Part": "भाग",
    },
    "sa": {
        "q_cloze": "रिक्तस्थानं पूरयत",
        "q_def": "“{term}” इत्यस्य कतमं वर्णनं युक्तम्?",
        "q_tf": "सत्यम् असत्यं वा?",
        "true": "सत्यम्",
        "false": "असत्यम्",
        "correct": "सम्यक्!",
        "incorrect": "न सम्यक्।",
        "answer_was": "सम्यक् उत्तरम्",
        "explained": "भवतः टिप्पणीभ्यः",
        "quiz_done": "प्रश्नोत्तरी समाप्ता: {n} मध्ये {k} सम्यक्।",
        "greeting": "नमस्ते! अहं भवतः अध्ययनसहायकः अस्मि। “{title}” विषये किमपि पृच्छतु — संस्कृते, हिन्दीभाषायाम्, तेलुगुभाषायाम् आङ्ग्लभाषायां वा। अहं संकल्पनाः व्याख्यातुम्, उदाहरणानि दातुम्, सारं वक्तुं, प्रश्नोत्तरीं च कर्तुं शक्नोमि।",
        "definition": "परिभाषा", "from_notes": "भवतः टिप्पणीभ्यः", "example": "उदाहरणम्", "formula": "सूत्रम्", "key_points": "मुख्यांशाः",
        "related": "सम्बद्धविषयाः", "review_first": "प्रथमम् एतानि पुनः पठतु", "next_up": "अग्रिमः विषयः", "page": "पृष्ठम्", "covers": "अस्मिन् भागे",
        "not_found": "भवता उपारोपितासु टिप्पणीषु एतत् न प्राप्तम्, अतः अनुमानं न करोमि। मानचित्रे भवतः संशयेषु एतत् योजितम् — अन्यथा पृच्छतु अथवा सम्बद्धम् अध्यायम् उपारोपयतु।",
        "closest": "प्राप्तः निकटतमः अंशः", "doubt_saved": "मानचित्रे संशयरूपेण योजितम्", "map_updated": "भवतः मानचित्रम् नवीकृतम्",
        "quiz_intro": "{topic} विषये {n} प्रश्नानां प्रश्नोत्तरी इयम्। प्रत्येकम् उत्तरं भवतः ज्ञानमानचित्रं नवीकरोति।",
        "quiz_none": "अस्मात् भागात् उत्तमाः प्रश्नाः अद्य न निर्मितुं शक्ताः — स्पष्टाः परिभाषाः अपेक्ष्यन्ते। अन्यं विषयं प्रयतताम्।",
        "summary_title": "सारः", "study_next": "इतः परम् एतानि पठितुं सूचयामि", "simple": "सरलभाषया",
        "src_lang_note": "भवतः टिप्पण्यः {lang}भाषायाम् सन्ति; अंशः यथालिखितं दर्श्यते।", "translated_note": "{lang}भाषातः यन्त्रानुवादः",
        "no_progress": "भवान् अद्यापि न आरब्धवान् — प्रश्नं पृच्छतु अथवा प्रश्नोत्तरीं करोतु, वैयक्तिकं मानचित्रं निर्मास्यामि।",
        "reason_weak_quiz": "अस्मिन् प्रश्नाः दुष्टाः जाताः", "reason_asked_doubt": "एतत् भवता पृष्टम्", "reason_not_started": "अद्यापि न आरब्धम्",
        "reason_keep_practising": "अभ्यासं करोतु", "reason_prereq_for": "{for_label} इत्यस्मात् पूर्वम् आवश्यकम्", "chapters": "अध्यायाः",
        "low_grounding": "एतत् स्वटिप्पणीभिः सह परीक्षताम् — उत्तरं टिप्पणीभ्यः अधिकं स्यात्।",
        "unit_Page": "पृष्ठम्", "unit_Slide": "पटलम्", "unit_Part": "भागः",
    },
}


def tr(code: str, key: str, /, **kw: str | int) -> str:
    table = T.get(code) or T["en"]
    s = table.get(key) or T["en"].get(key, key)
    try:
        return s.format(**kw) if kw else s
    except (KeyError, IndexError):
        return s


# ---------------------------------------------------------------------------
# Intent keywords (substring match on the lower-cased, ZW-stripped question)
# ---------------------------------------------------------------------------
INTENTS: dict[str, tuple[str, ...]] = {
    "quiz": (
        "quiz", "test me", "mcq", "practice question", "practice me", "examine me", "ask me question", "ask me some question",
        "క్విజ్", "ప్రశ్నలు అడుగు", "ప్రశ్నలు ఇవ్వు", "ప్రశ్నలు పెట్టు", "పరీక్ష పెట్టు", "టెస్ట్", "టెస్ట్ చేయి",
        "क्विज़", "क्विज", "प्रश्नोत्तरी", "प्रश्न पूछ", "टेस्ट लो", "परीक्षा लो", "मुझसे सवाल",
        "प्रश्नान् पृच्छ", "परीक्षां कुरु", "प्रश्नोत्तरीं",
        "quiz cheyyi", "quiz pettu", "prashnalu adugu", "sawal pucho", "quiz lo",
    ),
    "summary": (
        "summar", "overview", "recap", "tl;dr", "in short", "gist", "main points", "key points of",
        "సారాంశం", "క్లుప్తంగా", "సంక్షిప్తంగా", "సమ్మరీ", "ముఖ్యాంశాలు",
        "सारांश", "संक्षेप", "सार बताओ", "समरी", "मुख्य बिंदु",
        "सारं वद", "सारः", "संक्षेपेण",
        "saramsam", "sankshiptanga", "short ga", "saransh",
    ),
    "example": ("example", "for instance", "illustrate", "ఉదాహరణ", "उदाहरण", "udaharana", "udaharan", "udaharanam"),
    "simple": (
        "simple", "simply", "easy way", "easily", "eli5", "layman", "in easy", "beginner",
        "సులభంగా", "సరళంగా", "తేలిగ్గా", "आसान", "सरल", "आसानी से", "सरलभाषया", "saralanga", "easy ga", "aasan",
    ),
    "recommend": (
        "what should i study", "what next", "what to study", "what to revise", "revise", "weak areas", "weak topics", "study plan", "what to read", "where am i",
        "ఏం చదవాలి", "తర్వాత ఏమి", "రివిజన్", "బలహీన", "क्या पढ़", "आगे क्या", "दोहरा", "कमज़ोर", "रिवीजन", "किं पठा", "अग्रे किम्", "kya padhu", "em chadavali",
    ),
    "greet": ("hi", "hello", "hey", "namaste", "namaskaram", "నమస్తే", "నమస్కారం", "హలో", "నమస్కారం", "नमस्ते", "हेलो", "नमो नमः", "नमस्कार"),
}


def detect_intent(text: str) -> list[str]:
    key = match_key(text)
    found: list[str] = []
    words = set(key.split())
    for intent, kws in INTENTS.items():
        for kw in kws:
            if intent == "greet":
                if kw in words and len(words) <= 4:
                    found.append(intent)
                    break
            elif kw in key:
                found.append(intent)
                break
    return found


def lang_options() -> list[dict[str, str]]:
    return [{"code": c, "name": LANG_NAMES[c]} for c in SUPPORTED_LANGS]
