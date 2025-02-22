import gradio as gr
import pandas as pd
from collections import Counter
import plotly.express as px
import os
import time
from functools import lru_cache

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams

# -------------------------
# 1) Connect to Qdrant
# -------------------------
api_key = os.environ.get("qdrant_api_key")
qdrant_client = QdrantClient(
    url="https://91a11fe3-9d9a-4344-a59c-caa311da78ec.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key=api_key,
    timeout=5
)
try:
    print("Existing Collections:", qdrant_client.get_collections())
except Exception as e:
    print("Error retrieving collections:", e)

# Define collection parameters.
collection_name = "game_log"
vector_dim = 1  # Using a dummy vector

# Create collection if it does not exist
if not qdrant_client.collection_exists(collection_name="game_log"):
    qdrant_client.create_collection(
        collection_name="game_log",
        vector_size=5,  # Adjust if needed
        distance="Cosine"
    )


# -------------------------
# 2) Load Words from CSV
# -------------------------
def load_valid_words():
    """
    Expects a CSV file named 'wordle.csv' with a column named 'word'.
    Loads all 5-letter alphabetic words into a list.
    """
    df = pd.read_csv('wordle.csv')
    valid = []
    for w in df['word']:
        w = str(w).strip().lower()
        if len(w) == 5 and w.isalpha():
            valid.append(w)
    return list(set(valid))


# -------------------------
# 3) Qdrant Logging Functions
# -------------------------
def log_game_data_qdrant(guess, feedback, candidate_count, attempt):
    """
    Inserts a log entry into Qdrant with a dummy vector.
    """
    dummy_vector = [0.0]
    point_id = int(time.time() * 1000)
    payload = {
        "attempt": attempt,
        "guess": guess,
        "feedback": feedback,
        "candidates_remaining": candidate_count
    }
    point = PointStruct(id=point_id, vector=dummy_vector, payload=payload)
    qdrant_client.upsert(collection_name=collection_name, points=[point])


def get_historical_stats_qdrant():
    """
    Retrieves historical stats from Qdrant by scanning logged entries.
    For each letter position, counts letters from winning guesses (feedback "ggggg")
    with extra weight for green (correct) feedback.
    """
    print("Attempting to retrieve historical stats from Qdrant...")
    try:
        result = qdrant_client.scroll(
            collection_name=collection_name,
            with_payload=True,
            limit=1000
        )
    except Exception as e:
        print("Error during Qdrant scroll:", e)
        return None

    points = result[0]
    if not points:
        print("No points found in Qdrant collection.")
        return None

    positions_stats = [Counter() for _ in range(5)]
    for point in points:
        feedback = point.payload.get("feedback", "")
        guess = point.payload.get("guess", "")
        if len(feedback) == 5 and len(guess) == 5:
            for i, letter in enumerate(guess):
                if feedback[i] == "g":
                    positions_stats[i][letter] += 2  # Green feedback weighted higher
                elif feedback[i] == "y":
                    positions_stats[i][letter] += 1  # Yellow feedback lower weight
    return positions_stats


# -------------------------
# 4) Helper Functions with Caching
# -------------------------
@lru_cache(maxsize=None)
def get_feedback(guess, secret):
    """
    Returns a 5-character string (each character 'b','y','g') comparing guess and secret.
    'b' = Gray, 'y' = Yellow, 'g' = Green.
    """
    feedback = ['b'] * 5
    secret_chars = list(secret)
    for i in range(5):
        if guess[i] == secret[i]:
            feedback[i] = 'g'
            secret_chars[i] = None
    for i in range(5):
        if feedback[i] != 'g' and guess[i] in secret_chars:
            feedback[i] = 'y'
            secret_chars[secret_chars.index(guess[i])] = None
    return ''.join(feedback)


def prune_candidates(guess, feedback, candidates):
    """
    Returns only candidates that would produce the same feedback as provided.
    """
    new_candidates = []
    for c in candidates:
        if get_feedback(guess, c) == feedback:
            new_candidates.append(c)
    return new_candidates


# -------------------------
# 5) Advanced Guess Strategy (Realtime Updated)
# -------------------------
def choose_next_guess_advanced(candidates, attempt):
    """
    Combines a base frequency score with a bonus from historical stats to choose the next guess.
    """
    if attempt == 1:
        starters = ["crate", "about", "valid", "louse", "poise"]
        for word in starters:
            if word in candidates:
                return word
        return candidates[0]
    if len(candidates) <= 2:
        return candidates[0]
    candidate_counters = {word: Counter(word) for word in candidates}
    historical_stats = get_historical_stats_qdrant()

    def score_word(word):
        score = 0
        word_counter = Counter(word)
        for c in candidates:
            candidate_counter = candidate_counters[c]
            for letter in set(word):
                score += min(word_counter[letter], candidate_counter[letter])
        score /= len(candidates)
        historical_bonus = 0
        if historical_stats:
            for i in range(5):
                historical_bonus += historical_stats[i].get(word[i], 0)
        weight = 0.1
        score += weight * historical_bonus
        return score

    best = max(candidates, key=score_word)
    return best


# -------------------------
# 6) Data Visualization Functions
# -------------------------
def create_position_frequency_heatmap(candidates):
    letters = [chr(i) for i in range(97, 123)]
    matrix = []
    for pos in range(5):
        counter = Counter([word[pos] for word in candidates])
        row = [counter.get(letter, 0) for letter in letters]
        matrix.append(row)
    fig = px.imshow(
        matrix,
        labels=dict(x="Letter", y="Position", color="Frequency"),
        x=[letter.upper() for letter in letters],
        y=[f"Pos {i + 1}" for i in range(5)],
        title="Letter Frequency Heatmap by Position"
    )
    return fig


def create_confidence_chart(candidates):
    total = len(candidates) if candidates else 1
    positions = [f"Pos {i + 1}" for i in range(5)]
    best_letters = []
    confidences = []
    for i in range(5):
        counter = Counter(word[i] for word in candidates)
        if counter:
            letter, count = max(counter.items(), key=lambda x: x[1])
            best_letters.append(letter.upper())
            confidences.append(count / total)
        else:
            best_letters.append("?")
            confidences.append(0)
    df = pd.DataFrame({
        "Position": positions,
        "Best Letter": best_letters,
        "Confidence": confidences
    })
    fig = px.bar(df, x="Position", y="Confidence", text="Best Letter",
                 title="Confidence per Position (Best Candidate Letter)")
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    return fig


# -------------------------
# 7) Functions for Rendering Guess Boxes
# -------------------------
def get_button_style(state):
    """
    Returns inline CSS style based on feedback state using Wordle theme colors.
    """
    if state == 'b':
        return "background-color: #787c7e; color: white;"  # Wordle gray
    elif state == 'y':
        return "background-color: #c9b458; color: white;"  # Wordle yellow
    elif state == 'g':
        return "background-color: #6aaa64; color: white;"  # Wordle green
    else:
        return ""


def render_letter_box(letter, state):
    """
    Returns HTML for a single letter box with styling based on its feedback state.
    """
    style = get_button_style(
        state) + "display:inline-block; width:80px; height:80px; margin:5px; border:2px solid #3a3a3c; text-align:center; line-height:80px; font-size:48px; border-radius:10px;"
    return f"<div class='letter-box' data-letter='{letter}' data-state='{state}' style='{style}'>{letter.upper()}</div>"


def render_guess_display(current_guess, current_feedback):
    """
    Renders the full guess (5 letters) as a row of styled letter boxes.
    """
    boxes = [render_letter_box(current_guess[i], current_feedback[i]) for i in range(5)]
    return "<div style='text-align:center;'>" + "".join(boxes) + "</div>"


# -------------------------
# 8) WordleSolver Class
# -------------------------
class WordleSolver:
    def __init__(self, valid_words):
        self.valid_words = valid_words
        self.reset()

    def reset(self):
        self.candidates = self.valid_words[:]
        self.attempt = 1
        self.history = []
        self.current_feedback = ['b'] * 5
        self.current_guess = choose_next_guess_advanced(self.candidates, self.attempt)
        return self.get_current_state()

    def next_guess(self):
        self.attempt += 1
        self.current_guess = choose_next_guess_advanced(self.candidates, self.attempt)
        self.current_feedback = ['b'] * 5
        return self.get_current_state()

    def submit_current_feedback(self):
        feedback = ''.join(self.current_feedback)
        self.history.append((self.current_guess, feedback))
        log_game_data_qdrant(self.current_guess, feedback, len(self.candidates), self.attempt)
        if feedback == "ggggg":
            msg = (
                f"🎉 Solved in {len(self.history)} attempts! The word was '{self.current_guess.upper()}'.\nHistory: {self.history}\nStarting new game...")
            self.reset()
            state = self.get_current_state()
            state["message"] = msg
            return state
        self.candidates = prune_candidates(self.current_guess, feedback, self.candidates)
        if not self.candidates:
            msg = "No valid words remain with that feedback. Starting new game..."
            self.reset()
            state = self.get_current_state()
            state["message"] = msg
            return state
        return self.next_guess()

    def get_current_state(self):
        state = {
            "guess": self.current_guess,
            "feedback": self.current_feedback,
            "message": f"Attempt {self.attempt}: Try '{self.current_guess.upper()}' (Candidates remaining: {len(self.candidates)})"
        }
        return state


# -------------------------
# 9) Gradio UI with Wordle-like Clickable Letter Boxes and Small Buttons Underneath
# -------------------------
def cycle_letter(fb, gs, idx):
    """
    Cycles the state of the letter at index idx:
    Gray ('b') -> Yellow ('y') -> Green ('g') -> Gray ('b').
    Returns updated feedback list and updated guess display HTML.
    """
    if isinstance(fb, str):
        fb = list(fb)
    new_feedback = fb.copy()
    if new_feedback[idx] == 'b':
        new_feedback[idx] = 'y'
    elif new_feedback[idx] == 'y':
        new_feedback[idx] = 'g'
    else:
        new_feedback[idx] = 'b'
    html_out = render_guess_display(gs, new_feedback)
    return new_feedback, html_out


def submit_feedback(solver, current_feedback):
    """
    Submits the current feedback to update the solver's state.
    Returns updated feedback, new guess display HTML, new guess state,
    status message, updated charts, and updated letter button values.
    """
    solver.current_feedback = current_feedback
    new_state = solver.submit_current_feedback()
    new_html = render_guess_display(new_state["guess"], new_state["feedback"])
    new_letters = list(new_state["guess"].upper())
    return (new_state["feedback"], new_html, new_state["guess"], new_state["message"],
            create_position_frequency_heatmap(solver.candidates),
            create_confidence_chart(solver.candidates),
            new_letters[0], new_letters[1], new_letters[2], new_letters[3], new_letters[4])


def reset_game(solver):
    new_state = solver.reset()
    new_html = render_guess_display(new_state["guess"], new_state["feedback"])
    new_letters = list(new_state["guess"].upper())
    return (new_state["feedback"], new_html, new_state["guess"], new_state["message"],
            create_position_frequency_heatmap(solver.candidates),
            create_confidence_chart(solver.candidates),
            new_letters[0], new_letters[1], new_letters[2], new_letters[3], new_letters[4])


def create_interface():
    solver = WordleSolver(load_valid_words())
    init_state = solver.get_current_state()
    init_guess_html = render_guess_display(init_state["guess"], init_state["feedback"])
    init_letters = list(init_state["guess"].upper())
    with gr.Blocks(css="""
        <style>
            .title {
                text-align: center;
                font-family: 'Helvetica', sans-serif;
                color: #e74c3c;
            }
            .status-box {
                font-size: 18px;
                color: #34495e;
                text-align: center;
            }
            /* Small button styling for the letter buttons underneath */
            .small-button {
                width: 40px !important;
                height: 40px !important;
                font-size: 16px !important;
                margin: 2px !important;
                border: 2px solid #3a3a3c !important;
                border-radius: 5px !important;
            }
        </style>
    """) as demo:
        gr.Markdown("<h1 class='title'>Advanced Wordle Solver</h1>")
        gr.Markdown(
            "Click on the small button below each letter to cycle its color (Gray → Yellow → Green). When you're ready, click **Submit Feedback** to process the guess.")

        # Hidden states to store current feedback and guess
        feedback_state = gr.State(value=init_state["feedback"])
        guess_state = gr.State(value=init_state["guess"])

        # Main guess display (large, non-clickable)
        guess_display = gr.HTML(value=init_guess_html, label="Current Guess", elem_id="guess_display")

        # Row of small buttons underneath the main display
        letter_buttons = []
        with gr.Row():
            for i in range(5):
                btn = gr.Button(value=init_letters[i], elem_id=f"letter_btn_{i}", variant="secondary",
                                elem_classes="small-button")
                letter_buttons.append(btn)

        with gr.Row():
            submit_btn = gr.Button("Submit Feedback", variant="primary")
            reset_btn = gr.Button("Start New Game")

        status_message = gr.Textbox(value=init_state["message"], label="Status", lines=2, elem_classes="status-box")
        with gr.Row():
            chart_heatmap = gr.Plot(value=create_position_frequency_heatmap(solver.candidates), label="Heatmap")
            chart_confidence = gr.Plot(value=create_confidence_chart(solver.candidates), label="Confidence Chart")

        # Set up callbacks for each small letter button.
        for idx in range(5):
            def make_cycle(idx):
                return lambda fb, gs: cycle_letter(fb, gs, idx)

            letter_buttons[idx].click(
                fn=make_cycle(idx),
                inputs=[feedback_state, guess_state],
                outputs=[feedback_state, guess_display]
            )

        # Callback for Submit Feedback button.
        submit_btn.click(
            fn=lambda fb, s=solver: submit_feedback(s, fb),
            inputs=[feedback_state],
            outputs=[feedback_state, guess_display, guess_state, status_message, chart_heatmap,
                     chart_confidence] + letter_buttons
        )

        # Callback for Reset button.
        reset_btn.click(
            fn=lambda s=solver: reset_game(s),
            inputs=[],
            outputs=[feedback_state, guess_display, guess_state, status_message, chart_heatmap,
                     chart_confidence] + letter_buttons
        )
    return demo


demo = create_interface()
demo.launch()
