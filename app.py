import streamlit as st
from google import genai
from google.genai.errors import APIError
from google.genai import types 
from dotenv import load_dotenv
import os

# ==============================================================================
# 1. APPLICATION SETUP AND INITIALIZATION (Using Caching for stability)
# ==============================================================================

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Use st.cache_resource to create the client only once and keep it stable
@st.cache_resource
def get_gemini_client():
    """Initializes and caches the Gemini client for persistence."""
    if not API_KEY:
        st.error("API Key not found. Please ensure GEMINI_API_KEY is set in your .env file.")
        st.stop()
    try:
        return genai.Client(api_key=API_KEY)
    except Exception:
        st.error("Error connecting to Gemini. Please check the key in .env.")
        st.stop()

client = get_gemini_client()

# Set project constants
MODEL = 'gemini-2.5-flash' 
# UPDATED SYSTEM PROMPT: Now requests multi-language capability for chat.
SYSTEM_PROMPT = "You are a world-class, extremely helpful expert explainer and tutor. For general conversation (Mode 1), reply in the same language as the user's last message. For structured analysis (Mode 2), strictly use English."

# ==============================================================================
# 2. CHAT SESSION MANAGEMENT (Hybrid Chat) - Ensures the chat session is persistent
# ==============================================================================

# Use st.cache_resource to create the chat session only once and keep it stable
# We need to manually tell Streamlit not to hash the client instance
@st.cache_resource(hash_funcs={genai.Client: lambda _: None})
def initialize_gemini_chat(client_instance):
    """Creates and caches the persistent chat session."""
    return client_instance.chats.create(model=MODEL)

# Initialize chat object in session state
if "gemini_chat" not in st.session_state:
    st.session_state.gemini_chat = initialize_gemini_chat(client)

# Initialize chat messages list with a welcome message
if "chat_messages" not in st.session_state:
    # Changed welcome message to Arabic
    st.session_state.chat_messages = [{"role": "assistant", "content": "مرحباً! أنا خبيرك التعليمي. يمكنك سؤالي بأي لغة وسأرد عليك بها، وسأتذكر حوارنا."}]

# ==============================================================================
# 3. STRUCTURED SEARCH AND EXPLAIN FUNCTION (For specific structured output)
# ==============================================================================

def structured_search_and_explain(topic_name):
    """
    Uses Gemini with Google Search tool to find real-time information
    and structure it into a professional, simplified explanation.
    """
    # This prompt forces the 3-part structured English output
    prompt = f"""
    You are an expert content creator. Use the Google Search tool to find the most accurate and up-to-date information for the topic: {topic_name}.
    
    You MUST structure your output using three dedicated sections:
    
    ## 1. Core Concept
    (Provide a single, short paragraph defining the concept clearly and concisely.)
    
    ## 2. Analogy & Example
    (Provide one compelling, real-world analogy to make the concept tangible.)
    
    ## 3. Significance & Impact
    (Explain, in one short paragraph, the impact or significance of this concept.)
    
    **Instructions:**
    * The entire output must be in **English**.
    * Use **Markdown formatting** (like headings) strictly.
    * Base your answer primarily on the search results.
    * Topic: {topic_name}
    """
    
    try:
        # Generate content using the defined model, prompt, and Google Search tool
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            # Pass tools inside the config object
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}] 
            )
        )
        return response.text

    except APIError as e:
        return f"AI API Error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# ==============================================================================
# 4. STREAMLIT USER INTERFACE (UI)
# ==============================================================================

st.title("🧠 خبير محادثة وبحث شامل (مشروع أحمد)")
st.subheader("اختر وضع التشغيل للبدء في التحليل أو المحادثة.")

# --- Mode Selection ---
mode = st.selectbox(
    "اختر وضع تشغيل التطبيق:",
    ("1. وضع المحادثة الهجينة (ذاكرة + بحث فوري)", "2. وضع التحليل المنظم (Structured Search & Explain)")
)

st.markdown("---")

# ==============================================================================
# 4.1. HYBRID CHAT MODE LOGIC (Memory + Search)
# ==============================================================================
if mode == "1. وضع المحادثة الهجينة (ذاكرة + بحث فوري)":
    
    st.info("في هذا الوضع، يحتفظ الروبوت بذاكرة الحوار **ويستخدم بحث جوجل مع كل رسالة** للحصول على معلومات دقيقة ومحدثة، والردود ستكون **بنفس اللغة التي تسأل بها**.")
    
    # Display all previous messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle New User Input
    if prompt := st.chat_input("اسأل سؤالاً أو تابع موضوعاً سابقاً..."):
        # The chat object is now guaranteed to exist due to caching
            
        # 1. Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 2. Add user message to history
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        # ===============================================================
        #  🔥 NEW: Identity Check (فحص الهوية)
        # ===============================================================
        
        # Check for identity questions using lower case for flexibility
        identity_keywords = ["من أنت", "من أنشأك", "من طورك", "who are you", "who made you"]
        
        # Determine if the prompt contains any of the identity keywords
        is_identity_question = any(keyword in prompt.lower() for keyword in identity_keywords)

        if is_identity_question:
            # Customized response
            custom_response = "أنا نموذج لغوي كبير، وقد تم تطويري وإنشائي بواسطة **أحمد الزاوجال** كجزء من مشروعه التعليمي المتميز! 💻"
            
            # Display custom response
            with st.chat_message("assistant"):
                st.markdown(custom_response)
            
            # Add custom response to history and stop processing for this input
            st.session_state.chat_messages.append({"role": "assistant", "content": custom_response})
            # To ensure the chat window updates immediately
            st.rerun() 
            # Note: The code below the 'else' block will handle the standard Gemini call.

        # ===============================================================
        #  END OF NEW: Identity Check
        # ===============================================================

        else: # Proceed with standard Gemini Chat call if it's not an identity question
            # 3. Get AI response and display it
            with st.chat_message("assistant"):
                with st.spinner("جاري التفكير والبحث الفوري..."):
                    try:
                        # Send tools and system_instruction inside the 'config' object
                        # SYSTEM_PROMPT now guides the model to reply in the user's language
                        response = st.session_state.gemini_chat.send_message(
                            prompt, 
                            config=types.GenerateContentConfig(
                                tools=[{"google_search": {}}],
                                system_instruction=SYSTEM_PROMPT
                            )
                        )
                        st.markdown(response.text)
                        st.session_state.chat_messages.append({"role": "assistant", "content": response.text})

                    except Exception as e:
                        st.error(f"حدث خطأ: {e}")

# ==============================================================================
# 4.2. STRUCTURED SEARCH MODE LOGIC (No Memory, Structured Output)
# ==============================================================================
elif mode == "2. وضع التحليل المنظم (Structured Search & Explain)":

    st.info("هذا الوضع ممتاز لإنشاء محتوى احترافي. يُدخل المصطلح ويتم تحليل الإجابة وتنسيقها في 3 أقسام **باللغة الإنجليزية** (بدون ذاكرة حوار).")

    search_topic = st.text_input(
        "أدخل المفهوم أو الكلمة المفتاحية (باللغة الإنجليزية للحصول على أفضل النتائج):",
        placeholder="Example: Recent discoveries about Jupiter's moon Europa"
    )

    if st.button("ابحث وحلّل المفهوم!"):
        if search_topic:
            with st.spinner('1. جاري البحث عن المعلومات وتحليلها بواسطة Gemini...'):
                # Note: structured_search_and_explain function still forces English output for professional content
                simplified_explanation = structured_search_and_explain(search_topic)

            # Display Final Result
            st.markdown("---")
            st.markdown(f"## 🧠 Simplified Explanation for {search_topic}:")
            st.write(simplified_explanation)



