"""Voice layer: Gnani STT in, Gnani TTS out. Nothing else is used for speech.

The agent never sees audio. `Voice.listen` gives it text and `Voice.speak` turns its reply into
a WAV file, so the Step 3 brain is unchanged.
"""
import os
import re
from pathlib import Path

from .config import STT_LANGUAGE, TTS_MODEL, TTS_VOICE, TTS_VOICE_HINGLISH
from .textutil import script_of, speakable

RECOVERY = {
    "en-IN": "Sorry, I didn't catch that. Could you say it again?",
    "hi-IN": "माफ़ कीजिए, मैं सुन नहीं पाई। क्या आप दोबारा बोल सकते हैं?",
}
HINGLISH_WORDS = {"hai", "hain", "kya", "aap", "aapka", "aapke", "ka", "ki", "ke", "nahi", "nahin", "aur", "kar",
                  "karein", "kariye", "mein", "main", "se", "ko", "raha", "sakte", "sakti", "sakta", "kab",
                  "mera", "meri", "ji", "dhanyavaad", "kripya", "tak", "hoga", "hogi", "ho"}


class VoiceError(RuntimeError):
    pass


def hinglish_score(text: str) -> int:
    return sum(1 for w in re.findall(r"[a-z]+", text.lower()) if w in HINGLISH_WORDS)


class Voice:
    def __init__(self, stt_language: str = STT_LANGUAGE, model: str = TTS_MODEL, voice: str = TTS_VOICE):
        if not os.environ.get("GNANI_API_KEY"):
            raise VoiceError("GNANI_API_KEY is not set. Copy .env.example to .env and add your key.")
        from gnani.stt import GnaniSTTClient
        from gnani.tts import GnaniTTSClient
        self.stt, self.tts = GnaniSTTClient(), GnaniTTSClient()
        self.stt_language, self.model, self.voice = stt_language, model, voice

    # ---- ears ------------------------------------------------------------------------
    def listen(self, audio_path, language: str = None) -> dict:
        """Transcribe a clip (REST accepts up to 60 s). Falls back to Hindi if English comes back empty.

        Returns {"transcript", "language", "error"}; an empty transcript means "ask again".
        """
        first = language or self.stt_language
        order = [first] + [c for c in ("hi-IN", "en-IN") if c != first][:1]
        last_error = None
        for code in order:
            try:
                kwargs = {"format": "transcribe"} if code in ("en-IN", "hi-IN") else {}
                res = self.stt.transcribe(str(audio_path), language_code=code, **kwargs)
                text = (res.get("transcript") or "").strip()
                if text:
                    return {"transcript": text, "language": code, "error": None}
            except Exception as exc:  # bad audio, too long, network: recover by asking again
                last_error = f"{type(exc).__name__}: {exc}"
        return {"transcript": "", "language": first, "error": last_error}

    # ---- mouth -----------------------------------------------------------------------
    def speak(self, text: str, out_path, language_hint: str = None) -> dict:
        """Synthesise `text` to a WAV file. Picks language/voice from the reply's script."""
        spoken = speakable(text)
        if script_of(spoken) == "devanagari":
            lang, voice = "hi-IN", self.voice
        elif hinglish_score(spoken) >= 2 or language_hint in ("hi-en", "hi"):
            lang, voice = "auto", TTS_VOICE_HINGLISH
        else:
            lang, voice = "en-IN", self.voice
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            self.tts.synthesize(spoken, voice=voice, model=self.model, language=lang, output_file=str(out_path))
            return {"spoken_text": spoken, "audio": str(out_path), "language": lang, "voice": voice, "error": None}
        except Exception as exc:
            return {"spoken_text": spoken, "audio": None, "language": lang, "voice": voice,
                    "error": f"{type(exc).__name__}: {exc}"}

    @staticmethod
    def recovery_text(language: str) -> str:
        return RECOVERY["hi-IN"] if language == "hi-IN" else RECOVERY["en-IN"]
