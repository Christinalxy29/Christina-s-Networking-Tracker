# networking_navigator.py
"""
Christina Networking Navigator - Streamlit single-file app

Features:
1) Interaction Entry (who, relationship type, anxiety 1-5, notes, tags, persona, replay text)
2) Mood Board (mosaic of colored tiles with emoji + tag)
3) Comfort Zone Radar Chart (plotly polar)
4) Anxiety Tracker line chart and simple "Play Animation"
5) Auto-Reflection Writer (rule-based; optional OpenAI key support if you want richer text)
6) Personas Map insights & counts
7) Gamification: XP, badges, progress bar
8) Conversation Replay area
9) Recommendations engine (simple rule logic)
10) Theme presets & switching

No external database: stored in st.session_state (persist while the app runs).
You can export / import JSON or CSV.
"""

import streamlit as st
from streamlit_extras.colored_header import colored_header
from streamlit_extras.stylable_container import stylable_container
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import uuid
import os
import time

DATA_FILE = "data.json"

# ----------------------------
# Load data from disk
# ----------------------------
def load_data():
    if not os.path.exists(DATA_FILE):
        return {"interactions": []}
    try:
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    except:
        return {"interactions": []}



# ----------------------------
# Save data to disk
# ----------------------------
def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)
# ---------- Utilities & Data structures ----------
DEFAULT_THEMES = {
    "Calming Pastel": {"bg": "#F5C8DE", "card": "#FFFFFF", "accent": "#A8D5E2"},
    "Productivity Blue": {"bg": "#ADD3F3", "card": "#FFFFFF", "accent": "#2B6CB0"},
    "High-Energy": {"bg": "#F3D09A", "card": "#FFFFFF", "accent": "#F6B042"},
}
EMOJI_BY_TONE = {
    "positive": "😊",
    "neutral": "😐",
    "negative": "😬",
    "excited": "🤩",
    "calm": "😌",
}
TAGS = {
    "coffee": "☕ Coffee Chat",
    "social": "💼 Networking Event",
    "casual": "💬 Casual conversation",
}
PERSONAS = {
    "supportive": "🌟 Supportive mentor",
    "cold": "🧊 Cold/brief connector",
    "formal": "🎭 Formal professional",
    "genuine": "🤝 Genuine collaborator",
    "energetic": "⚡ Energetic peer",
}
BADGES = {
    "first_cold_email": {"label": "First cold email", "xp": 50},
    "talked_to_stranger": {"label": "Talked to a stranger", "xp": 50},
    "handled_awkward_pause": {"label": "Handled an awkward pause", "xp": 50},
    "good_followup": {"label": "Asked a good follow-up", "xp": 50},
}

def compute_xp(anxiety: int, tone: str) -> int:
    """
    XP Rules:
    - High anxiety = more XP (reward doing hard conversations)
    - Tone 'positive' or 'excited' = XP bonus
    - Tone 'negative' = no bonus
    """

    # --- Anxiety XP ---
    # anxiety 5 → +30 XP  
    # anxiety 4 → +20 XP  
    # anxiety 3 → +10 XP  
    # anxiety 2 → +5 XP  
    # anxiety 1 → +3 XP  
    anxiety_bonus = {
        5: 30,
        4: 20,
        3: 10,
        2: 5,
        1: 3
    }.get(anxiety, 5)  # default 5

    # --- Tone XP ---
    tone = tone.lower()
    if tone in ["positive", "excited", "enthusiastic", "warm"]:
        tone_bonus = 15
    elif tone in ["neutral"]:
        tone_bonus = 5
    else:
        # negative / anxious / cold → NO bonus
        tone_bonus = 0

    return anxiety_bonus + tone_bonus

def recalc_all_xp():
    total_xp = 0
    new_interactions = []

    # Reset badge records but DO NOT add badge XP here
    st.session_state.badges = set()

    for it in st.session_state.interactions:
        anxiety = int(it.get("anxiety", 3))
        tone = it.get("tone", "neutral")

        # recompute only BASE interaction xp
        xp_value = compute_xp(anxiety, tone)
        it["xp"] = xp_value
        total_xp += xp_value

        # restore badges WITHOUT adding XP
        notes = it.get("notes", "").lower()
        rel = it.get("relationship", "")

        if "cold email" in notes:
            st.session_state.badges.add("first_cold_email")

        if rel == "stranger":
            st.session_state.badges.add("talked_to_stranger")

        if "awkward" in notes:
            st.session_state.badges.add("handled_awkward_pause")

        if "follow-up" in notes or "follow up" in notes:
            st.session_state.badges.add("good_followup")

        new_interactions.append(it)

    st.session_state.interactions = new_interactions

    # Add badge XP ONCE (after loop)
    for b in st.session_state.badges:
        total_xp += BADGES[b]["xp"]

    st.session_state.xp = total_xp
    save_data({"interactions": st.session_state.interactions})

# ----------------------------
# Initialize persistent state
# ----------------------------
if "data" not in st.session_state:
    st.session_state.data = load_data()

if "interactions" not in st.session_state:
    loaded = load_data()
    st.session_state.interactions = loaded.get("interactions", [])

recalc_all_xp()

if "xp" not in st.session_state:
    st.session_state.xp = 0

if "badges" not in st.session_state:
    st.session_state.badges = set()



# Optional: use OpenAI if you set OPENAI_API_KEY in env (not required)
USE_OPENAI = False
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_KEY:
    USE_OPENAI = True
    try:
        import openai
        openai.api_key = OPENAI_KEY
    except Exception:
        USE_OPENAI = False

st.set_page_config(page_title="Christina's Networking Tracker", layout="wide")

@dataclass
class Interaction:
    id: str
    date: str
    who: str
    relationship: str
    anxiety: int
    tone: str
    tags: List[str]
    persona: str
    notes: str
    replay_you: str
    replay_other: str
    xp: int = 0

# initialize session state containers
if "interactions" not in st.session_state:
    st.session_state.interactions: List[Dict[str, Any]] = []

if "xp" not in st.session_state:
    st.session_state.xp = 0

if "badges" not in st.session_state:
    st.session_state.badges = set()

if "theme" not in st.session_state:
    st.session_state.theme = "Calming Pastel"

# ---------- Helper functions ----------
def add_interaction(who, date, relationship, anxiety, tone, tags, persona, notes, replay_you, replay_other):
    entry = Interaction(
        id=str(uuid.uuid4()),
        date=str(date),
        who=who,
        relationship=relationship,
        anxiety=int(anxiety),
        tone=tone,
        tags=tags,
        persona=persona,
        notes=notes,
        replay_you=replay_you,
        replay_other=replay_other,
        xp=compute_xp(anxiety, tone)
    )
    st.session_state.interactions.append(asdict(entry))
    st.session_state.xp += entry.xp
    check_and_award_badges(entry)
    save_data({"interactions": st.session_state.interactions})
    st.success("Interaction saved! 🎉")


def anxiety_to_color(score):
    if score is None:
        return "#cccccc"
    try:
        score = float(score)
    except:
        return "#cccccc"
    if score <= 3:
        return "#7ED957"
    elif score <= 6:
        return "#F7C948"
    else:
        return "#EF5350"


def check_and_award_badges(entry: Interaction):
    notes = entry.notes.lower()
    if "cold email" in notes or "cold email" in entry.tags:
        st.session_state.badges.add("first_cold_email")
        st.session_state.xp += BADGES["first_cold_email"]["xp"]
    if entry.relationship == "stranger":
        st.session_state.badges.add("talked_to_stranger")
        st.session_state.xp += BADGES["talked_to_stranger"]["xp"]
    if "awkward" in notes:
        st.session_state.badges.add("handled_awkward_pause")
        st.session_state.xp += BADGES["handled_awkward_pause"]["xp"]
    if "follow-up" in notes or "follow up" in notes:
        st.session_state.badges.add("good_followup")
        st.session_state.xp += BADGES["good_followup"]["xp"]

def interactions_df():
    return pd.DataFrame(st.session_state.interactions)

def weekly_anxiety_average():
    df = interactions_df()
    if df.empty:
        return None
    return df['anxiety'].mean()

def theme_style():
    t = DEFAULT_THEMES.get(
        st.session_state.get("theme", "Calming Pastel"),
        DEFAULT_THEMES["Calming Pastel"]
    )
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {t['bg']} !important;
        }}
        .theme-card {{
            background-color: {t['card']} !important;
            padding: 20px;
            border-radius: 16px;
            margin-bottom: 20px;
        }}
        .theme-accent {{
            color: {t['accent']} !important;
            font-weight: 600;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    return t

def generate_reflection(notes: str, anxiety: int, tone: str):
    takeaway = (f"You noticed {('high' if anxiety>=4 else 'moderate' if anxiety==3 else 'low')} anxiety; "
                f"tone felt {tone}.")
    suggestion = ("Prepare 2-3 talking points when anxiety is high; try a 30s small-win warm-up before the chat.")
    authenticity = ("Name one value you wanted to express; if you didn't, practice saying it out loud once.")
    return {"takeaway": takeaway, "suggestion": suggestion, "authenticity": authenticity}

def recommend_next():
    df = interactions_df()
    if df.empty:
        return "Log a few interactions and the recommender will suggest next steps."
    avg_by_rel = df.groupby('relationship')['anxiety'].mean().to_dict()
    lowest = min(avg_by_rel.items(), key=lambda x: x[1])[0]
    highest = max(avg_by_rel.items(), key=lambda x: x[1])[0]
    return (f"If you're feeling confident, reach out to someone in '{lowest}' this week. "
            f"To reduce stress with '{highest}', prepare 3 prompts and a 1-sentence intro.")

def compute_comfort_scores():
    df = interactions_df()
    categories = ["professor", "alumni", "classmate", "stranger", "executive"]
    if df.empty:
        return {cat: 2.5 for cat in categories}
    out = {}
    for cat in categories:
        subset = df[df['relationship'] == cat]
        out[cat] = 2.5 if subset.empty else max(1, 5 - subset['anxiety'].mean())
    return out

def badges_display():
    if not st.session_state.badges:
        st.info("No badges yet — try a cold outreach or handle an awkward pause to earn badges!")
        return
    cols = st.columns(len(st.session_state.badges))
    for (c, b) in zip(cols, st.session_state.badges):
        meta = BADGES.get(b, {"label": b, "xp": 0})
        c.metric(meta["label"], f"{meta['xp']} XP")

# ---------- Layout ----------
with st.container():
    col1, col2, col3 = st.columns([3,2,2])
    with col1:
        st.title("💫 Christina's Networking Tracker")
        st.write("A safe space to understand your social energy")
    with col2:
        theme_choice = st.selectbox("Theme preset", list(DEFAULT_THEMES.keys()),
                                    index=list(DEFAULT_THEMES.keys()).index(st.session_state.theme))
        st.session_state.theme = theme_choice
        t = theme_style()
    with col3:
        avg_anx = weekly_anxiety_average()
        if avg_anx is None:
            st.metric("Weekly Anxiety Avg", "—")
        else:
            st.metric("Weekly Anxiety Avg", f"{avg_anx:.2f} / 5")
        st.metric("Total XP", st.session_state.xp)

st.markdown("---")
 # Sidebar background
st.markdown("""
<style>
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #FAF1C8;   
    }
</style>
""", unsafe_allow_html=True)
# Sidebar
with st.sidebar:
    st.header("Pages")
    page = st.selectbox(
        "Select a page",
        ["Home", "Log Interaction", "Analytics", "Reflection Hub", "Export/Import", "Settings"])
    st.markdown("---")
    st.header("Badges")
    badges_display()
    st.markdown("---")
    st.write("Quick tips:")
    st.write("• Add each interaction in the Log page.")
    st.write("• Use tags to categorize chats.")

# ---------- Helper: delete an interaction ----------
def delete_interaction(interaction_id: str):
    """Remove an interaction by its unique id."""
    st.session_state.interactions = [
        i for i in st.session_state.interactions if i.get('id') != interaction_id
    ]
    save_data({"interactions": st.session_state.interactions})
    st.success("Deleted interaction ✅")
def edit_interaction(entry_id, updated_data):
    for i, it in enumerate(st.session_state.interactions):
        if it["id"] == entry_id:
            st.session_state.interactions[i].update(updated_data)
            save_data({"interactions": st.session_state.interactions})
            return True
    return False

# ---------- Home ----------
if page == "Home":
    left, right = st.columns([2, 1])

    # ---------------- Left Column ----------------
    with left:
        st.subheader("Recent Interactions")
        df = interactions_df()

        if df.empty:
            st.info("No interactions yet. Go to 'Log Interaction' to add one.")
        else:
            # Display recent 10 interactions in a table
            recent = df.sort_values('date', ascending=False).head(15)
            st.dataframe(
                recent[['date', 'who', 'relationship', 'anxiety', 'tone', 'persona', 'tags', 'notes']]
            )

        st.markdown("---")

        st.subheader("Mood Board Preview")
        tiles = st.session_state.interactions[-24:] if st.session_state.interactions else []
        cols = st.columns(6)

        # Display tiles in a grid
        for i, it in enumerate(tiles[::-1]):
            col = cols[i % 6]
            color = anxiety_to_color(it['anxiety'])
            emoji = EMOJI_BY_TONE.get(it['tone'], "🙂")
            tag_labels = ", ".join(it['tags']) if it['tags'] else "untagged"

            card_html = f"""
            <div style="
                background:{color}; 
                border-radius:12px; 
                padding:10px; 
                min-height:90px; 
                display:flex; 
                flex-direction:column; 
                justify-content:center;">
                <div style="font-size:28px;">
                    {emoji} <strong style='font-size:12px'>{tag_labels}</strong>
                </div>
                <div style="font-size:11px; opacity:0.9; margin-top:6px;">
                    {it['who']} · {it['relationship']}
                </div>
            </div>
            """
            col.markdown(card_html, unsafe_allow_html=True)


    # ---------------- Right Column ----------------
    with right:
        st.subheader("Quick Analytics")
        st.metric("Total Interactions", len(st.session_state.interactions))
        st.metric("Unique People", len(set([i['who'] for i in st.session_state.interactions])))

        st.markdown("**Recommendation**")
        st.write(recommend_next())



# ---------- Interaction Entry ----------
if page == "Log Interaction":
    st.header("Log a New Interaction")
    with st.form("interaction_form", clear_on_submit=False):
        rcol1, rcol2 = st.columns(2)
        who = rcol1.text_input("Who did you talk to? (name or descriptor)", "")
        relationship = rcol1.selectbox("Relationship type", ["professor","alumni","classmate","stranger","executive"])
        interaction_date = st.date_input("Date of interaction")
        anxiety = rcol1.slider("Anxiety level (1 = calm, 5 = very anxious)", 1, 5, 3)
        tone = rcol2.selectbox("Reflection tone", list(EMOJI_BY_TONE.keys()), index=1)
        tags_selected = rcol2.multiselect("Tags (interaction type)", list(TAGS.values()))
        persona = st.radio("Persona of the other person", list(PERSONAS.values()))
        notes = st.text_area("Short notes / what happened", height=120)
        replay_you = st.text_area("What you said (short)", placeholder="Optional - what you said", height=70)
        replay_other = st.text_area("What they said (short)", placeholder="Optional - what they said", height=70)
        submitted = st.form_submit_button("Save interaction")
        if submitted:
            add_interaction(who or "Unknown", interaction_date, relationship, anxiety, tone, tags_selected, persona, notes, replay_you, replay_other)

# ---------- Analytics ----------
if page == "Analytics":
    st.header("Analytics")
    df = interactions_df()
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Comfort Zone Radar")
        scores = compute_comfort_scores()
        cats = list(scores.keys())
        vals = list(scores.values())
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=cats + [cats[0]],
            fill='toself',
            name='Comfort level (1-5)'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,5])),
            showlegend=False,
            height=420
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        st.subheader("Anxiety Over Time")
        if df.empty:
            st.info("No data yet")
        else:
            df_sorted = df.sort_values('date')
            fig2 = px.line(df_sorted, x='date', y='anxiety', markers=True,
                           title="Anxiety Timeline (Based on user-entered dates)")
            fig2.update_traces(marker=dict(size=8))
            st.plotly_chart(fig2, use_container_width=True)
            if st.button("Play Anxiety Animation"):
                placeholder = st.empty()
                for i in range(len(df_sorted)):
                    part = df_sorted.iloc[:i+1]
                    fig_tmp = px.line(part, x='date', y='anxiety', markers=True)
                    placeholder.plotly_chart(fig_tmp, use_container_width=True)
                    time.sleep(0.3)
    with col2:
        st.subheader("Persona Distribution")
        if df.empty:
            st.info("No data yet")
        else:
            persona_counts = df['persona'].value_counts().reset_index()
            persona_counts.columns = ['persona', 'count']
            st.bar_chart(data=persona_counts.set_index('persona'))
        st.markdown("---")
        st.subheader("Mood Board (full)")
        tiles = st.session_state.interactions[::-1]
        if not tiles:
            st.info("No tiles yet")
        else:
            grid_cols = st.columns(3)
            for i, it in enumerate(tiles):
                col = grid_cols[i % 3]
                color = anxiety_to_color(it['anxiety'])
                emoji = EMOJI_BY_TONE.get(it['tone'], "🙂")
                tag_labels = ", ".join(it['tags']) if it['tags'] else "untagged"
                small = f"""
                <div style="background:{color}; border-radius:10px; padding:10px; margin-bottom:8px;">
                    <div style="font-size:20px;">{emoji} <strong style='font-size:12px'>{tag_labels}</strong></div>
                    <div style="font-size:11px; opacity:0.85;'>{it['who']} — {it['relationship']}</div>
                </div>
                """
                col.markdown(small, unsafe_allow_html=True)

# ---------- Reflection Hub ----------
if page == "Reflection Hub":
    st.header("Reflection Hub")
    st.write("Auto-generated reflections from your latest note.")
    df = interactions_df()
    if df.empty:
        st.info("Add an interaction to generate reflections.")
    else:
        latest = df.sort_values('date').iloc[-1]
        st.subheader("Latest interaction")
        st.write(f"**Who:** {latest['who']} · **Relationship:** {latest['relationship']} · **Anxiety:** {latest['anxiety']}")
        st.write("**Notes:**", latest['notes'])
        if st.button("Generate Reflection"):
            with st.spinner("Generating..."):
                out = generate_reflection(latest['notes'], latest['anxiety'], latest['tone'])
                st.markdown(f"**Takeaway:** {out['takeaway']}")
                st.markdown(f"**Suggestion:** {out['suggestion']}")
                st.markdown(f"**Authenticity prompt:** {out['authenticity']}")


# ---------- Export / Import ----------
if page == "Export/Import":
    st.header("Export / Import Data")
    if st.button("Export JSON"):
        json_str = json.dumps(st.session_state.interactions, indent=4)
        st.download_button("Download interactions.json", data=json_str, file_name="interactions.json", mime="application/json")
    st.markdown("---")
    st.subheader("Import JSON")
    uploaded_file = st.file_uploader("Upload interactions.json", type=["json"])
    if uploaded_file:
        try:
            imported = json.load(uploaded_file)
            st.session_state.interactions.extend(imported)
            st.success(f"Imported {len(imported)} interactions!")
        except Exception as e:
            st.error("Failed to import JSON: "+str(e))

#------ SETTINGS PAGE ------
if page == "Settings":
    st.header("Edit / Delete Interactions")

    df = interactions_df()

    if df.empty:
        st.info("No interactions yet.")
    else:
        # Show latest 15, newest first
        recent = df.sort_values("date", ascending=False).head(15)

        for _, row in recent.iterrows():
            with st.expander(f"{row['date']} — {row['who']} ({row['relationship']})"):

                st.write(f"**Notes:** {row['notes']}")
                st.write(f"**Anxiety:** {row['anxiety']} | **Tone:** {row['tone']}")
                st.write(f"**Persona:** {row['persona']} | **Tags:** {', '.join(row['tags'])}")

                col1, col2 = st.columns([1, 1])

                # -------- EDIT BUTTON --------
                with col1:
                    if st.button("✏️ Edit", key=f"edit_{row['id']}"):
                        # Save to session state for Log Interaction page
                        st.session_state.edit_id = row["id"]
                        st.session_state.edit_data = row.to_dict()

                        # Navigate to Log Interaction
                        st.session_state.page = "Log Interaction"
                        st.rerun()

                # -------- DELETE BUTTON --------
                with col2:
                    if st.button("🗑️ Delete", key=f"del_{row['id']}"):
                        delete_interaction(row["id"])
                        st.success("Deleted!")
                        st.rerun()



# ---------- Save periodically ----------
save_data({"interactions": st.session_state.interactions})
