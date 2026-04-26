import os
import time
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    def __init__(self, provider: Optional[str] = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "groq")).lower().strip()
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")

    def is_configured(self) -> bool:
        if self.provider == "groq":
            return bool(self.groq_api_key)
        if self.provider == "gemini":
            return bool(self.gemini_api_key)
        return False

    def call_llm(self, prompt: str, system_prompt: str = "") -> str:
        for attempt in range(3):
            try:
                if self.provider == "groq":
                    return self._call_groq(prompt, system_prompt)
                if self.provider == "gemini":
                    return self._call_gemini(prompt, system_prompt)
                raise ValueError(f"Unsupported provider: {self.provider}")
            except Exception:
                if attempt == 2:
                    break
                time.sleep(0.8 * (attempt + 1))
        return ""

    def _call_groq(self, prompt: str, system_prompt: str = "") -> str:
        from groq import Groq

        if not self.groq_api_key:
            return ""
        client = Groq(api_key=self.groq_api_key)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            temperature=0.5,
        )
        return response.choices[0].message.content or ""

    def _call_gemini(self, prompt: str, system_prompt: str = "") -> str:
        import google.generativeai as genai

        if not self.gemini_api_key:
            return ""
        genai.configure(api_key=self.gemini_api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        final_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        response = model.generate_content(final_prompt)
        return getattr(response, "text", "") or ""
