"""AI Study Companion - Track D: Personalized Tutoring & Adaptive Learning"""
import json
import random
import re

import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Study Companion", page_icon="📚", layout="centered")
st.markdown(
    "<style>.block-container{padding-top:1.2rem;padding-bottom:3rem;}</style>",
    unsafe_allow_html=True,
)

# ------------------------------ settings ------------------------------------
# The app tries these models in order until one works.
DEFAULT_MODELS = ["gemini-3.5-flash", "gemini-2.5-flash", "gemini-flash-latest"]
LEVELS = ["Easy", "Medium", "Hard"]

SAMPLE_NOTES = """Photosynthesis is the process by which plants, algae and some bacteria convert light energy into chemical energy stored in glucose. It takes place mainly in the chloroplasts of leaf cells, which contain a green pigment called chlorophyll that absorbs red and blue light.

The overall equation is: 6CO2 + 6H2O + light energy -> C6H12O6 + 6O2. Carbon dioxide enters the leaf through tiny pores called stomata, and water is absorbed by the roots and carried up through the xylem.

Photosynthesis has two stages. In the light-dependent reactions, which occur in the thylakoid membranes, light energy splits water molecules, releases oxygen and produces the energy carriers ATP and NADPH. In the Calvin cycle, which occurs in the stroma, ATP and NADPH are used to fix carbon dioxide into glucose.

Factors that affect the rate of photosynthesis include light intensity, carbon dioxide concentration and temperature. At very high temperatures enzymes denature and the rate drops sharply. A factor in short supply that limits the rate is called a limiting factor."""

# ------------------------------ session state -------------------------------
DEFAULTS = {
    "kb": None, "chat": [], "quiz": None, "quiz_id": 0,
    "quiz_result": None, "level": "Medium", "weak": [],
}
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


# ------------------------------ Gemini helper -------------------------------
def get_secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return None


def get_api_key():
    return get_secret("GEMINI_API_KEY") or st.session_state.get("manual_key", "")


def ask_gemini(prompt, as_json=False):
    key = get_api_key()
    if not key:
        raise RuntimeError(
            "No API key found. Add GEMINI_API_KEY in the app's Secrets, "
            "or paste it into the box at the top of the page."
        )
    client = genai.Client(api_key=key)
    config = types.GenerateContentConfig(
        temperature=0.3,
        response_mime_type="application/json" if as_json else None,
    )
    last_error = None
    for model in dict.fromkeys(m for m in [get_secret("GEMINI_MODEL")] + DEFAULT_MODELS if m):
        try:
            response = client.models.generate_content(model=model, contents=prompt, config=config)
            if response.text:
                return response.text
        except Exception as e:  # try the next model
            last_error = e
    raise RuntimeError(f"Gemini request failed: {last_error}")


# ------------------------------ knowledge base (RAG) ------------------------
def split_sections(text, size=3000):
    text = text.strip()
    return [(f"section {i // size + 1}", text[i:i + size]) for i in range(0, len(text), size)]


def read_upload(f):
    """Return a list of (source, location, text)."""
    if f.name.lower().endswith(".pdf"):
        out = []
        for i, page in enumerate(PdfReader(f).pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                out.append((f.name, f"p.{i}", text))
        return out
    text = f.read().decode("utf-8", errors="ignore")
    return [(f.name, loc, t) for loc, t in split_sections(text)]


def chunk_text(text, size=900, overlap=150):
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def build_kb(docs):
    chunks = [
        {"source": s, "loc": loc, "text": piece}
        for s, loc, text in docs
        for piece in chunk_text(text)
    ]
    if not chunks:
        raise ValueError("No readable text found.")
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    matrix = vec.fit_transform([c["text"] for c in chunks])
    return {"chunks": chunks, "vec": vec, "matrix": matrix}


def load_kb(docs):
    try:
        st.session_state.kb = build_kb(docs)
    except Exception as e:
        st.error(f"Could not build the knowledge base: {e}")
        return
    st.session_state.update(chat=[], quiz=None, quiz_result=None, weak=[], level="Medium")


def retrieve(query, k=5):
    kb = st.session_state.kb
    sims = cosine_similarity(kb["vec"].transform([query]), kb["matrix"])[0]
    order = sims.argsort()[::-1][:k]
    return [kb["chunks"][i] for i in order if sims[i] > 0]


def format_excerpts(chunks):
    return "\n\n".join(f"[{c['source']}, {c['loc']}]\n{c['text']}" for c in chunks)


# ------------------------------ tutoring chat -------------------------------
def answer_question(question):
    history = st.session_state.chat
    last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    chunks = retrieve(f"{last_user} {question}")
    if not chunks:
        return ("I couldn't find anything about that in your uploaded materials. "
                "Try rephrasing, or upload notes that cover this topic.", [])
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    prompt = f"""You are a friendly, patient tutor. Answer the student's question using ONLY the excerpts below.
Rules:
- Cite every fact in the form (filename, p.3) using the exact labels shown on the excerpts.
- If the excerpts do not contain the answer, say so honestly. Never invent facts.
- Explain simply, with a short example if useful.
- End with one short question that checks the student's understanding.

EXCERPTS:
{format_excerpts(chunks)}

CONVERSATION SO FAR:
{convo}

STUDENT QUESTION: {question}"""
    return ask_gemini(prompt), chunks


def show_sources(chunks):
    with st.expander("📖 Sources used"):
        for c in chunks:
            st.markdown(f"**{c['source']} · {c['loc']}**")
            st.caption(c["text"][:220] + "...")


# ------------------------------ adaptive quiz -------------------------------
def quiz_context():
    chunks = st.session_state.kb["chunks"]
    picked = retrieve(" ".join(st.session_state.weak), k=4) if st.session_state.weak else []
    pool = [c for c in chunks if c not in picked]
    picked += random.sample(pool, min(6, len(pool)))
    return picked


def parse_quiz(raw):
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    data = json.loads(raw)
    if isinstance(data, dict):
        data = data.get("questions", [])
    quiz = []
    for q in data[:3]:
        options = [str(o) for o in q["options"]][:4]
        answer = options[int(q["answer_index"])]
        random.shuffle(options)
        quiz.append({
            "question": q["question"], "options": options, "answer": answer,
            "topic": q.get("topic", "General"),
            "explanation": q.get("explanation", ""), "source": q.get("source", ""),
        })
    if len(quiz) != 3:
        raise ValueError("Quiz must have 3 questions")
    return quiz


def generate_quiz():
    weak = st.session_state.weak
    focus = (f"The student struggled with: {', '.join(weak)}. Include at least one question on these."
             if weak else "")
    prompt = f"""Create a 3-question multiple-choice quiz from the study excerpts below.
Difficulty: {st.session_state.level}. {focus}
Use ONLY information from the excerpts.
Return ONLY a JSON array of exactly 3 objects with these keys:
"question" (string), "options" (array of exactly 4 strings, no A/B/C/D prefixes),
"answer_index" (integer 0-3), "topic" (2-4 words), "explanation" (1-2 sentences),
"source" (like "notes.pdf, p.3", copied from the excerpt label).

EXCERPTS:
{format_excerpts(quiz_context())}"""
    for _ in range(2):
        try:
            return parse_quiz(ask_gemini(prompt, as_json=True))
        except (ValueError, KeyError, IndexError, TypeError):
            continue
    raise RuntimeError("Could not build a valid quiz. Please tap the button again.")


def new_quiz():
    with st.spinner("Writing your quiz..."):
        try:
            st.session_state.quiz = generate_quiz()
            st.session_state.quiz_id += 1
            st.session_state.quiz_result = None
            st.rerun()
        except RuntimeError as e:
            st.error(str(e))


def grade_quiz(answers):
    quiz = st.session_state.quiz
    score = sum(a == q["answer"] for a, q in zip(answers, quiz))
    weak = set(st.session_state.weak)
    for a, q in zip(answers, quiz):
        (weak.discard if a == q["answer"] else weak.add)(q["topic"])
    i = LEVELS.index(st.session_state.level)
    if score == 3:
        i, advice = min(i + 1, 2), "🌟 Perfect! Your next quiz will be **harder**."
    elif score <= 1:
        i, advice = max(i - 1, 0), "💪 Let's slow down. Your next quiz will be **easier** and revisit weak spots."
    else:
        advice = "👍 Good progress! Same level next time, with focus on your weak spots."
    st.session_state.level = LEVELS[i]
    st.session_state.weak = sorted(weak)
    st.session_state.quiz_result = {"score": score, "answers": answers, "advice": advice}


# ------------------------------ UI ------------------------------------------
st.title("📚 AI Study Companion")
st.caption("Upload your notes → ask questions with cited answers → take adaptive quizzes.")

if not get_api_key():
    st.warning("Paste your free Gemini API key to start (or add it to the app Secrets).")
    st.text_input("Gemini API key", type="password", key="manual_key")

tab_mat, tab_chat, tab_quiz = st.tabs(["📄 Materials", "💬 Tutor Chat", "📝 Quiz"])

with tab_mat:
    files = st.file_uploader("Upload PDFs or .txt files", type=["pdf", "txt"], accept_multiple_files=True)
    pasted = st.text_area("...or paste notes / a lecture transcript", height=150)
    c1, c2 = st.columns(2)
    if c1.button("⚙️ Build knowledge base", type="primary"):
        docs = []
        for f in files or []:
            try:
                docs += read_upload(f)
            except Exception:
                st.warning(f"Couldn't read {f.name} (password-protected or damaged?).")
        if pasted.strip():
            docs += [("Pasted notes", loc, t) for loc, t in split_sections(pasted)]
        if not docs:
            st.error("No readable text found. Scanned (image-only) PDFs aren't supported - "
                     "use a text-based PDF or paste the text.")
        else:
            load_kb(docs)
    if c2.button("✨ Load sample notes"):
        load_kb([("sample_biology_notes.txt", loc, t) for loc, t in split_sections(SAMPLE_NOTES)])
    if st.session_state.kb:
        kb = st.session_state.kb
        n_src = len({c["source"] for c in kb["chunks"]})
        st.success(f"✅ Knowledge base ready: {n_src} source(s), {len(kb['chunks'])} chunks. "
                   "Go to the Tutor Chat or Quiz tab.")

with tab_chat:
    if not st.session_state.kb:
        st.info("Add study materials in the Materials tab first.")
    else:
        for m in st.session_state.chat:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m.get("sources"):
                    show_sources(m["sources"])
        question = st.chat_input("Ask a question about your notes...")
        if question:
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        reply, sources = answer_question(question)
                    except RuntimeError as e:
                        reply, sources = f"⚠️ {e}", []
                st.markdown(reply)
                if sources:
                    show_sources(sources)
            st.session_state.chat += [
                {"role": "user", "content": question},
                {"role": "assistant", "content": reply, "sources": sources},
            ]

with tab_quiz:
    if not st.session_state.kb:
        st.info("Add study materials in the Materials tab first.")
    else:
        weak_txt = f" · Review: {', '.join(st.session_state.weak)}" if st.session_state.weak else ""
        st.caption(f"Difficulty: **{st.session_state.level}**{weak_txt}")
        quiz, result, qid = st.session_state.quiz, st.session_state.quiz_result, st.session_state.quiz_id
        if quiz is None:
            if st.button("🎯 Generate quiz", type="primary"):
                new_quiz()
        elif result is None:
            with st.form("quiz_form"):
                for i, q in enumerate(quiz):
                    st.markdown(f"**Q{i + 1}. {q['question']}**")
                    st.radio("Choose one", q["options"], index=None,
                             key=f"q{qid}_{i}", label_visibility="collapsed")
                submitted = st.form_submit_button("Submit answers")
            if submitted:
                answers = [st.session_state.get(f"q{qid}_{i}") for i in range(3)]
                if None in answers:
                    st.warning("Please answer all 3 questions.")
                else:
                    grade_quiz(answers)
                    st.rerun()
        else:
            st.subheader(f"Score: {result['score']}/3")
            for i, q in enumerate(quiz):
                chosen = result["answers"][i]
                st.markdown(f"**Q{i + 1}. {q['question']}**")
                if chosen == q["answer"]:
                    st.success(f"✅ Correct: {q['answer']}")
                else:
                    st.error(f"❌ You chose: {chosen}")
                    st.info(f"Correct answer: {q['answer']}")
                st.caption(f"{q['explanation']}  📖 {q['source']}")
            st.markdown(result["advice"])
            if st.button("➡️ Next adaptive quiz", type="primary"):
                new_quiz()
