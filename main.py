import os
import sys
import time
import logging
import tempfile
import subprocess
import urllib.parse

import json
import requests
import speech_recognition as sr
import static_ffmpeg

# הוסף ffmpeg ל-PATH אוטומטית (עובד ב-Railway ללא nixpacks)
static_ffmpeg.add_paths()

# ===== Logging =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# ===== Environment Variables =====
YEMOT_PHONE    = os.getenv("YEMOT_PHONE")
YEMOT_PASSWORD = os.getenv("YEMOT_PASSWORD")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")

if not all([YEMOT_PHONE, YEMOT_PASSWORD, GROQ_API_KEY]):
    log.error("חסרים משתני סביבה: YEMOT_PHONE, YEMOT_PASSWORD, GROQ_API_KEY")
    sys.exit(1)

# ===== הגדרות =====
EXTENSION      = "3"
INTERVAL       = 60
PROCESSED_FILE = os.getenv("PROCESSED_FILE", "processed.txt")
MAX_RETRIES    = 3
HEALTH_EVERY   = 10   # הדפסת health check כל X מחזורים
TIMEOUT_SHORT  = 10   # login, get_files
TIMEOUT_LONG   = 30   # download, upload, TTS, AI

SYSTEM_PROMPT = """אתה עוזר קולי חכם של מערכת ימות המשיח. ענה תמיד בעברית בלבד.
תן תשובה קצרה וברורה — משפט או שניים, מתאים לשמיעה בטלפון.

כללים:
- הגדרות, מדע, היסטוריה, תרבות, הלכה — ענה בביטחון מהידע שלך.
- אם קיבלת מידע מהאינטרנט בתחילת ההודעה (מסומן ב-[מידע עדכני]) — השתמש בו בתשובה.
- אם אינך יודע — אמור זאת ישירות בלי להמציא."""

NEEDS_SEARCH_PROMPT = """האם השאלה הבאה דורשת מידע עדכני מהאינטרנט?
שאלות שדורשות חיפוש: מי מכהן בתפקיד כלשהו, מה השעה/תאריך/כניסת שבת, חדשות, מזג אוויר, שערי מטבע, מחירים.
שאלות שלא דורשות חיפוש: הגדרות, היסטוריה, מדע, הלכה, מתמטיקה, שאלות כלליות.

ענה במילה אחת בלבד: כן או לא.
שאלה: """

session = requests.Session()
session.headers.update({"User-Agent": "YemotAI/1.0"})


# ===== Retry Logic =====
def with_retry(func, *args, retries=MAX_RETRIES, **kwargs):
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            wait = 2 ** attempt
            log.warning(f"ניסיון {attempt+1}/{retries} נכשל: {e}. ממתין {wait} שניות...")
            time.sleep(wait)
    raise Exception(f"כשל אחרי {retries} ניסיונות")


# ===== ימות המשיח =====
def _login():
    resp = session.post(
        "https://private.call2all.co.il/ym/api/Login",
        data={"username": YEMOT_PHONE, "password": YEMOT_PASSWORD},
        timeout=TIMEOUT_SHORT
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("responseStatus") == "OK":
        return data["token"]
    raise Exception(f"כשל התחברות: {data.get('message')}")

def _get_files(token):
    resp = session.post(
        "https://private.call2all.co.il/ym/api/GetFiles",
        data={"token": token, "path": f"/{EXTENSION}/"},
        timeout=TIMEOUT_SHORT
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("responseStatus") == "OK":
        return data.get("files", [])
    raise Exception(f"שגיאת GetFiles: {data.get('message')}")

def _download(token, filename):
    resp = session.post(
        "https://private.call2all.co.il/ym/api/DownloadFile",
        data={"token": token, "path": f"ivr2:/{EXTENSION}/{filename}"},
        timeout=TIMEOUT_LONG
    )
    resp.raise_for_status()
    if resp.content[:4] == b'RIFF':
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        tmp.write(resp.content)
        tmp.close()
        return tmp.name
    raise Exception(f"שגיאת הורדה: תוכן לא תקין (status {resp.status_code})")

def _upload(token, wav_path):
    with open(wav_path, 'rb') as f:
        resp = session.post(
            "https://private.call2all.co.il/ym/api/UploadFile",
            data={"token": token, "path": f"ivr2:/{EXTENSION}/M0000.wav"},
            files={"file": ("M0000.wav", f, "audio/wav")},
            timeout=TIMEOUT_LONG
        )
    resp.raise_for_status()
    data = resp.json()
    if data.get("responseStatus") == "OK":
        return True
    raise Exception(f"שגיאת העלאה: {data.get('message')}")


# ===== AI =====
def _needs_search(user_text: str) -> bool:
    """שואל את הבינה אם השאלה דורשת חיפוש — קריאה קלה עם מודל קטן"""
    try:
        resp = session.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.1-8b-instant",   # מודל קטן וזול לשאלת כן/לא
                "messages": [{"role": "user", "content": NEEDS_SEARCH_PROMPT + user_text}],
                "max_tokens": 5
            },
            timeout=TIMEOUT_SHORT
        )
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"].strip().lower()
        return "כן" in answer or "yes" in answer
    except Exception as e:
        log.warning(f"בדיקת חיפוש נכשלה: {e} — ממשיך ללא חיפוש")
        return False

def _tavily_search(query: str) -> str:
    """מחזיר תקציר תוצאות חיפוש מ-Tavily"""
    try:
        resp = session.post(
            "https://api.tavily.com/search",
            json={
                "api_key": os.getenv("TAVILY_API_KEY"),
                "query": query,
                "search_depth": "basic",
                "max_results": 3
            },
            timeout=TIMEOUT_LONG
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return ""
        # מחזיר את התוכן הרלוונטי בלבד
        snippets = [r.get("content", "")[:200] for r in results[:2]]
        return " | ".join(snippets)
    except Exception as e:
        log.warning(f"חיפוש Tavily נכשל: {e}")
        return ""

def _ask_groq(user_text: str) -> str:
    # שלב 1: האם צריך חיפוש?
    tavily_key = os.getenv("TAVILY_API_KEY")
    context = ""

    if tavily_key and _needs_search(user_text):
        log.info("🔍 מחפש מידע עדכני...")
        context = _tavily_search(user_text)
        if context:
            log.info(f"נמצא מידע: {context[:80]}...")

    # שלב 2: בונה הודעה עם הקשר אם יש
    user_message = user_text
    if context:
        user_message = f"[מידע עדכני מהאינטרנט]: {context}\n\nשאלה: {user_text}"

    resp = session.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": 200
        },
        timeout=TIMEOUT_LONG
    )
    resp.raise_for_status()
    data = resp.json()
    if "choices" in data:
        return data["choices"][0]["message"]["content"]
    raise Exception(f"שגיאת Groq: {data}")


# ===== שמע =====
def _speech_to_text(audio_path) -> str:
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data, language='he-IL')
        # פחות מ-2 מילים = רעש/שקט — לא שולחים ל-Groq
        if len(text.split()) < 2:
            log.warning(f"STT: טקסט קצר מדי ('{text}'), מתייחס כשקט")
            return ""
        return text
    except sr.UnknownValueError:
        return ""

def _text_to_speech(text):
    """המרת טקסט לדיבור בעברית עם קול גברי דרך edge-tts"""
    mp3_path = "response_tmp.mp3"
    wav_path = "ai_response.wav"

    # edge-tts: קול גברי בעברית
    result = subprocess.run(
        ["edge-tts", "--voice", "he-IL-AvriNeural", "--text", text, "--write-media", mp3_path],
        capture_output=True,
        timeout=TIMEOUT_LONG
    )
    if result.returncode != 0:
        log.warning(f"edge-tts נכשל (stderr: {result.stderr.decode().strip()}), נסיון עם Google TTS...")
        encoded = urllib.parse.quote(text)
        url = f"https://translate.google.com/translate_tts?ie=UTF-8&q={encoded}&tl=iw&client=tw-ob"
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=TIMEOUT_LONG)
        resp.raise_for_status()
        with open(mp3_path, 'wb') as f:
            f.write(resp.content)

    result = subprocess.run([
        "ffmpeg", "-y", "-i", mp3_path,
        "-ar", "8000", "-ac", "1", "-acodec", "pcm_s16le",
        wav_path
    ], capture_output=True)

    if result.returncode != 0:
        raise Exception(f"ffmpeg נכשל: {result.stderr.decode()}")

    if os.path.exists(mp3_path):
        os.unlink(mp3_path)

    return wav_path


# ===== Processed Files (נטען לזיכרון פעם אחת) =====
def _init_processed() -> set:
    try:
        with open(PROCESSED_FILE, 'r') as f:
            return set(f.read().splitlines())
    except FileNotFoundError:
        return set()

processed_set: set = _init_processed()   # זיכרון גלובלי — לא קוראים מהדיסק בכל מחזור

def save_processed(filename: str):
    processed_set.add(filename)
    with open(PROCESSED_FILE, 'a') as f:
        f.write(filename + "\n")


ERROR_MESSAGE = "מצטערים, חלה שגיאה בעיבוד הנתונים. אנא נסו שוב מאוחר יותר."

def _upload_error_message(token):
    """מעלה הודעת שגיאה קולית כדי שהמשתמש לא ישמע שקט"""
    try:
        log.info("מעלה הודעת שגיאה קולית...")
        wav_path = _text_to_speech(ERROR_MESSAGE)
        try:
            with_retry(_upload, token, wav_path)
            log.info("הודעת שגיאה הועלתה בהצלחה")
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)
    except Exception as e:
        log.error(f"נכשל גם להעלות הודעת שגיאה: {e}")


# ===== Pipeline =====
def run_pipeline():
    token = with_retry(_login)
    files = with_retry(_get_files, token)

    recordings = []
    for f in files:
        name = f.get('name', f) if isinstance(f, dict) else str(f)
        if (name.lower().endswith(('.wav', '.mp3'))
                and name not in ('M0000.wav', 'ai_response.wav')
                and not name.endswith('.ymgr')
                and name not in processed_set):
            recordings.append(name)

    if not recordings:
        log.info("אין הקלטות חדשות")
        return

    target = recordings[0]
    log.info(f"עובד על: {target}")

    audio_path = with_retry(_download, token, target)
    try:
        user_text = _speech_to_text(audio_path)
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)

    if not user_text:
        log.warning("לא זוהה טקסט — מדלג")
        save_processed(target)
        return

    log.info(f"שאלה: {user_text}")

    try:
        ai_response = with_retry(_ask_groq, user_text)
    except Exception as e:
        log.error(f"Groq נכשל: {e} — מעלה הודעת שגיאה קולית")
        _upload_error_message(token)
        save_processed(target)
        return

    log.info(f"תשובה: {ai_response}")

    wav_path = _text_to_speech(ai_response)
    try:
        with_retry(_upload, token, wav_path)
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)

    save_processed(target)
    log.info("✅ סיום מוצלח")


# ===== Main Loop =====
if __name__ == "__main__":
    log.info("🚀 המערכת מתחילה...")
    log.info("⚠️  שים לב: processed.txt נשמר מקומית.")
    log.info("    להתמדה בין restarts, הגדר Railway Volume ו-PROCESSED_FILE=/data/processed.txt")

    cycle = 0
    while True:
        cycle += 1

        if cycle % HEALTH_EVERY == 0:
            log.info(f"💚 Health check — מחזור {cycle} רץ תקין")

        log.info(f"--- מחזור {cycle} ---")
        try:
            run_pipeline()
        except Exception as e:
            log.error(f"שגיאה במחזור: {e}", exc_info=True)
            time.sleep(10)   # מניעת tight crash loop

        time.sleep(INTERVAL)
