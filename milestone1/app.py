import streamlit as st
import cv2
import sqlite3
import pytesseract
import json
import os
import pandas as pd
import plotly.express as px
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime
import time
import re
import hashlib

from google.api_core.exceptions import ResourceExhausted, NotFound

# ✅ PDF Support
import fitz  # PyMuPDF

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(page_title="Receipt & Invoice Digitizer", layout="wide")

# =========================
# LOAD ENV + GEMINI CONFIG
# =========================
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not API_KEY:
    st.error("❌ GEMINI_API_KEY not found! Add it in .env file.")
    st.stop()

genai.configure(api_key=API_KEY)

# =========================
# TESSERACT PATH
# =========================
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# =========================
# DATABASE: vault.db
# =========================
conn = sqlite3.connect("vault.db", check_same_thread=False)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS invoices (
    invoice_id TEXT PRIMARY KEY,
    vendor_name TEXT,
    date TEXT,
    total_amount REAL,
    raw_text TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_id TEXT,
    item_name TEXT,
    quantity INTEGER,
    price REAL,
    FOREIGN KEY(invoice_id) REFERENCES invoices(invoice_id)
)
""")
conn.commit()


# =========================
# DB MIGRATION
# =========================
def ensure_column_exists(table_name: str, column_name: str, column_type: str):
    cur.execute(f"PRAGMA table_info({table_name})")
    cols = [row[1] for row in cur.fetchall()]
    if column_name not in cols:
        cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
        conn.commit()


ensure_column_exists("invoices", "created_at", "TEXT")
ensure_column_exists("invoices", "category", "TEXT")
ensure_column_exists("invoices", "currency", "TEXT")
ensure_column_exists("invoices", "tax_amount", "REAL")
ensure_column_exists("invoices", "items_sum", "REAL")
ensure_column_exists("invoices", "validation_status", "TEXT")

# =========================
# GEMINI MODEL AUTO DETECT
# =========================
@st.cache_data(show_spinner=False)
def get_best_gemini_model() -> str:
    try:
        models = genai.list_models()
        candidates = []

        for m in models:
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                name = (m.name or "").lower()
                if "exp" in name or "preview" in name:
                    continue
                candidates.append(m.name)

        preference = [
            "gemini-1.5-flash",
            "gemini-flash-latest",
            "gemini-1.5-pro",
            "gemini-pro-latest",
        ]
        for pref in preference:
            for c in candidates:
                if pref in c.lower():
                    return c

        return candidates[0] if candidates else ""
    except Exception:
        return ""


MODEL_NAME = get_best_gemini_model()
if not MODEL_NAME:
    st.error("❌ No supported Gemini model found for your API key.")
    st.stop()


# =========================
# PDF → IMAGES
# =========================
def pdf_to_images(uploaded_pdf, zoom: float = 2.0):
    pdf_bytes = uploaded_pdf.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    images = []
    mat = fitz.Matrix(zoom, zoom)

    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    doc.close()
    return images


# =========================
# CATEGORY DETECTION
# =========================
CATEGORY_LIST = ["Dining", "Shopping", "Travel", "Health", "Bills", "Education", "Others"]


def classify_category(vendor: str, items: list[str]) -> str:
    text = " ".join([vendor] + items).lower()

    rules = {
        "Dining": ["restaurant", "cafe", "coffee", "biryani", "pizza", "burger", "zomato", "swiggy", "food", "hotel"],
        "Shopping": ["mart", "store", "amazon", "flipkart", "dmart", "d-mart", "supermarket", "mall", "shopping"],
        "Travel": ["uber", "ola", "irctc", "rail", "flight", "petrol", "diesel", "fuel", "metro", "bus"],
        "Health": ["pharmacy", "medical", "clinic", "hospital", "medicine", "apollo", "medplus"],
        "Bills": ["electric", "current", "water", "gas", "recharge", "bill", "broadband", "wifi", "internet"],
        "Education": ["books", "stationery", "college", "course", "tuition", "institute", "school"]
    }

    for cat, keywords in rules.items():
        if any(k in text for k in keywords):
            return cat

    return "Others"


def backfill_categories():
    rows = cur.execute("""
        SELECT invoice_id, vendor_name
        FROM invoices
        WHERE category IS NULL OR category=''
    """).fetchall()

    for invoice_id, vendor_name in rows:
        item_rows = cur.execute(
            "SELECT item_name FROM items WHERE invoice_id=?",
            (invoice_id,)
        ).fetchall()
        item_list = [r[0] for r in item_rows] if item_rows else []
        cat = classify_category(vendor_name or "", item_list)
        cur.execute("UPDATE invoices SET category=? WHERE invoice_id=?", (cat, invoice_id))

    conn.commit()


backfill_categories()


# =========================
# IMAGE PREPROCESSING
# =========================
def preprocess_image(img_bgr: np.ndarray,
                     denoise_strength: int = 5,
                     block_size: int = 21,
                     c_value: int = 7) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if w < 1200:
        scale = 1200 / w
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    denoised = cv2.fastNlMeansDenoising(gray, None, denoise_strength, 7, 21)

    if block_size % 2 == 0:
        block_size += 1
    if block_size < 3:
        block_size = 3

    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, c_value
    )

    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    return cleaned


def ocr_extract_text(processed_img: np.ndarray) -> str:
    config = r'--oem 3 --psm 6'
    return (pytesseract.image_to_string(processed_img, config=config) or "").strip()


# =========================
# CURRENCY + TOTAL + TAX
# =========================
def detect_currency(raw_text: str) -> str:
    text = (raw_text or "").lower()
    if "₹" in text or "rs" in text or "inr" in text:
        return "INR"
    if "$" in text or "usd" in text:
        return "USD"
    if "€" in text or "eur" in text:
        return "EUR"
    if "£" in text or "gbp" in text:
        return "GBP"
    if "¥" in text or "jpy" in text:
        return "JPY"
    return "INR"


def extract_total_from_text(raw_text: str) -> float:
    lines = [ln.strip() for ln in (raw_text or "").splitlines() if ln.strip()]
    for ln in lines[::-1]:
        low = ln.lower()
        if any(k in low for k in ["grand total", "total amount", "net total", "total", "amount due"]):
            nums = re.findall(r"\d+(?:\.\d+)?", ln.replace(",", ""))
            if nums:
                try:
                    return float(nums[-1])
                except:
                    pass

    nums = re.findall(r"\d+(?:\.\d+)?", (raw_text or "").replace(",", ""))
    floats = []
    for n in nums:
        try:
            floats.append(float(n))
        except:
            pass
    return max(floats) if floats else 0.0


def extract_tax_from_text(raw_text: str) -> float:
    text = (raw_text or "").replace(",", "")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines[::-1]:
        low = ln.lower()
        if any(k in low for k in ["gst", "cgst", "sgst", "igst", "vat", "tax"]):
            nums = re.findall(r"\d+(?:\.\d+)?", ln)
            if nums:
                try:
                    return float(nums[-1])
                except:
                    pass
    return 0.0


# =========================
# ITEMS CLEANING
# =========================
def clean_items(items: list[dict]) -> list[dict]:
    cleaned = []
    bad_words = ["subtotal", "sub total", "tax", "gst", "vat", "total", "amount due", "balance", "net total"]

    for it in items or []:
        name = str(it.get("item", "")).strip()
        if not name:
            continue
        low = name.lower()
        if any(b in low for b in bad_words):
            continue

        try:
            qty = int(float(it.get("quantity", 1) or 1))
        except:
            qty = 1

        try:
            price = float(it.get("price", 0) or 0)
        except:
            price = 0.0

        cleaned.append({"item": name, "quantity": qty, "price": price})
    return cleaned


# =========================
# VALIDATION
# =========================
def calc_items_sum(data: dict) -> float:
    total = 0.0
    for it in (data.get("items", []) or []):
        try:
            q = float(it.get("quantity", 1) or 1)
            p = float(it.get("price", 0) or 0)
            total += (q * p)
        except:
            continue
    return float(total)


def validate_invoice(data: dict, raw_text: str) -> dict:
    data["items"] = clean_items(data.get("items", []))

    try:
        total = float(data.get("total_amount", 0) or 0)
    except:
        total = 0.0
    if total <= 0:
        total = extract_total_from_text(raw_text)
        data["total_amount"] = total

    try:
        tax = float(data.get("tax_amount", 0) or 0)
    except:
        tax = 0.0
    if tax <= 0:
        tax = extract_tax_from_text(raw_text)
        data["tax_amount"] = tax

    items_sum = calc_items_sum(data)
    tol = max(2.0, total * 0.02)

    status = "VALID"
    if len(data.get("items", []) or []) == 0:
        status = "NOT_VALID"
    else:
        if abs(items_sum - total) <= tol:
            status = "VALID"
        elif abs((items_sum + tax) - total) <= tol:
            status = "VALID"
        else:
            status = "NOT_VALID"

    data["items_sum"] = items_sum
    data["validation_status"] = status
    data["currency"] = data.get("currency") or detect_currency(raw_text)
    return data


def show_validation_calculation(data: dict):
    total = float(data.get("total_amount", 0) or 0)
    tax = float(data.get("tax_amount", 0) or 0)
    items_sum = float(data.get("items_sum", 0) or 0)
    sum_plus_tax = items_sum + tax
    diff = abs(sum_plus_tax - total)

    st.markdown("### ✅ Validation Calculation")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Items Sum", f"{items_sum:.2f}")
    c2.metric("Tax", f"{tax:.2f}")
    c3.metric("Total (Bill)", f"{total:.2f}")
    c4.metric("Items + Tax", f"{sum_plus_tax:.2f}")
    c5.metric("Difference", f"{diff:.2f}")

    status = data.get("validation_status", "NOT_VALID")
    if status == "VALID":
        st.success("✅ VALID (Items match total)")
    else:
        st.warning("⚠ NOT VALID (Mismatch in totals)")


# =========================
# GEMINI HELPERS
# =========================
def clean_gemini_json(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1]
    return cleaned


def fallback_extract(raw_text: str) -> dict:
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    vendor = lines[0] if lines else "Unknown Vendor"
    invoice_id = f"INV-{int(time.time())}"
    total = extract_total_from_text(raw_text)
    tax = extract_tax_from_text(raw_text)
    currency = detect_currency(raw_text)

    return {
        "invoice_id": invoice_id,
        "vendor_name": vendor,
        "date": "",
        "items": [],
        "total_amount": total,
        "tax_amount": tax,
        "currency": currency
    }


def file_cache_key(uploaded_file, page_no: int = 1) -> str:
    """
    ✅ Unique cache key for each file + page (for PDF)
    """
    # For PDF we already read content while converting -> so we use name + page
    base = f"{uploaded_file.name}_{uploaded_file.size}_{page_no}"
    return hashlib.md5(base.encode()).hexdigest()


def gemini_extract_structured(raw_text: str, cache_key: str) -> dict:
    """
    ✅ Gemini runs ONCE per file (cached)
    ✅ NO infinite retries
    ✅ NO 60s sleep
    ✅ fallback if quota exceeded
    """

    if "gemini_cache" not in st.session_state:
        st.session_state["gemini_cache"] = {}

    if cache_key in st.session_state["gemini_cache"]:
        return st.session_state["gemini_cache"][cache_key]

    model = genai.GenerativeModel(MODEL_NAME)

    prompt = f"""
Return ONLY STRICT JSON. No explanation.

Schema:
{{
  "invoice_id": "string",
  "vendor_name": "string",
  "date": "YYYY-MM-DD",
  "currency": "INR/USD/EUR/GBP/etc",
  "tax_amount": 0,
  "items": [{{"item":"string","quantity":0,"price":0}}],
  "total_amount": 0
}}

Rules for items:
- items MUST contain only purchased products/services (line items).
- DO NOT include rows like Subtotal, Tax, Total, Amount Due, Balance.
- If quantity missing, use 1.

TEXT:
{raw_text}
"""

    try:
        response = model.generate_content(prompt)
        data = json.loads(clean_gemini_json(response.text))

        data.setdefault("invoice_id", f"INV-{int(time.time())}")
        data.setdefault("vendor_name", "")
        data.setdefault("date", "")
        data.setdefault("currency", detect_currency(raw_text))
        data.setdefault("tax_amount", 0)
        data.setdefault("items", [])
        data.setdefault("total_amount", 0)

        data["items"] = clean_items(data.get("items", []))

        # ✅ Save to cache
        st.session_state["gemini_cache"][cache_key] = data
        return data

    except ResourceExhausted:
        st.error("⚠ Gemini quota exceeded / rate limit hit.")
        data = fallback_extract(raw_text)
        st.session_state["gemini_cache"][cache_key] = data
        return data

    except NotFound:
        data = fallback_extract(raw_text)
        st.session_state["gemini_cache"][cache_key] = data
        return data

    except Exception:
        data = fallback_extract(raw_text)
        st.session_state["gemini_cache"][cache_key] = data
        return data


# =========================
# DB FUNCTIONS
# =========================
def save_to_db(data: dict, raw_text: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    invoice_id = str(data.get("invoice_id", "")).strip() or f"INV-{int(time.time())}"
    vendor_name = str(data.get("vendor_name", "")).strip()
    date = str(data.get("date", "")).strip()

    currency = str(data.get("currency", "")).strip() or detect_currency(raw_text)

    total_amount = float(data.get("total_amount", 0) or 0)
    tax_amount = float(data.get("tax_amount", 0) or 0)
    items_sum = float(data.get("items_sum", 0) or 0)

    items_list = [it.get("item", "") for it in (data.get("items", []) or [])]
    category = classify_category(vendor_name, items_list)

    cur.execute("""
        INSERT OR REPLACE INTO invoices(
            invoice_id, vendor_name, date, total_amount, raw_text, created_at, category,
            currency, tax_amount, items_sum, validation_status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        invoice_id, vendor_name, date, total_amount, raw_text, now, category,
        currency, tax_amount, items_sum, data.get("validation_status", "")
    ))

    cur.execute("DELETE FROM items WHERE invoice_id=?", (invoice_id,))
    for it in data.get("items", []) or []:
        item_name = str(it.get("item", "")).strip()
        if not item_name:
            continue
        qty = int(it.get("quantity", 1) or 1)
        price = float(it.get("price", 0) or 0)

        cur.execute("""
            INSERT INTO items(invoice_id, item_name, quantity, price)
            VALUES (?, ?, ?, ?)
        """, (invoice_id, item_name, qty, price))

    conn.commit()
    return invoice_id


def fetch_invoices_summary() -> pd.DataFrame:
    df = pd.read_sql(
        """
        SELECT invoice_id, vendor_name, date, total_amount, created_at, category
        FROM invoices ORDER BY created_at DESC
        """,
        conn
    )
    df["category"] = df["category"].fillna("Others").replace("", "Others")
    return df


def fetch_items(invoice_id: str) -> pd.DataFrame:
    return pd.read_sql(
        "SELECT item_name, quantity, price FROM items WHERE invoice_id=?",
        conn,
        params=(invoice_id,)
    )


def fetch_invoice_details(invoice_id: str) -> pd.DataFrame:
    row = cur.execute(
        """
        SELECT invoice_id, vendor_name, date, total_amount, created_at, category, currency, tax_amount
        FROM invoices WHERE invoice_id=?
        """,
        (invoice_id,)
    ).fetchone()

    if not row:
        return pd.DataFrame()

    return pd.DataFrame([{
        "invoice_id": row[0],
        "vendor_name": row[1],
        "date": row[2],
        "total_amount": row[3],
        "created_at": row[4],
        "category": row[5],
        "currency": row[6],
        "tax_amount": row[7],
    }])


def find_duplicate_invoice(raw_text: str):
    if not raw_text.strip():
        return None
    prefix = raw_text[:200]
    row = cur.execute(
        "SELECT invoice_id FROM invoices WHERE raw_text LIKE ? LIMIT 1",
        (prefix + "%",)
    ).fetchone()
    return row[0] if row else None


def delete_invoice(invoice_id: str):
    cur.execute("DELETE FROM items WHERE invoice_id=?", (invoice_id,))
    cur.execute("DELETE FROM invoices WHERE invoice_id=?", (invoice_id,))
    conn.commit()


def reset_database():
    cur.execute("DELETE FROM items")
    cur.execute("DELETE FROM invoices")
    conn.commit()


# =========================
# UI HELPERS
# =========================
def card(title: str, icon: str):
    st.markdown(
        f"""
        <div style="border:1px solid #e6e6e6;border-radius:14px;
        padding:14px;box-shadow:0 2px 10px rgba(0,0,0,0.04);
        background:white;margin-bottom:12px;">
            <h4 style="margin:0 0 10px 0;">{icon} {title}</h4>
        """,
        unsafe_allow_html=True
    )


def end_card():
    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# NAVIGATION
# =========================
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Upload & Extract", "Dashboard", "History"])

st.sidebar.divider()
st.sidebar.subheader("⚠ Database Controls")
reset_confirm = st.sidebar.checkbox("I understand this will delete ALL data")
if st.sidebar.button("🧹 Reset Database", use_container_width=True, disabled=not reset_confirm):
    reset_database()
    st.sidebar.success("✅ Database reset successful!")
    st.rerun()


# =========================
# PAGE 1: Upload & Extract
# =========================
if page == "Upload & Extract":
    st.markdown(
        """
        <div style="background:#1f77d0;padding:18px;border-radius:14px;color:white;">
            <h2 style="margin:0;">Receipt & Invoice Digitizer</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")

    col_upload, col_preprocess, col_output = st.columns([1.0, 1.2, 1.3])

    with col_upload:
        card("File Upload Interface", "📤")
        uploaded_file = st.file_uploader(
            "Upload Receipt / Invoice",
            type=["png", "jpg", "jpeg", "pdf"],
            label_visibility="collapsed"
        )
        st.caption("✅ Supports PNG / JPG / JPEG / PDF")
        end_card()

    if uploaded_file:
        if st.session_state.get("last_uploaded") != uploaded_file.name:
            st.session_state["last_uploaded"] = uploaded_file.name
            st.session_state.pop("gemini_cache", None)  # clear cache for new file
            st.session_state.pop("last_invoice_id", None)

        # ✅ PDF handling
        if uploaded_file.type == "application/pdf":
            pdf_images = pdf_to_images(uploaded_file)
            if not pdf_images:
                st.error("❌ Unable to extract pages from PDF.")
                st.stop()

            if len(pdf_images) == 1:
                page_no = 1
                st.info("📄 Single-page PDF detected. Using page 1 automatically.")
            else:
                page_no = st.slider("Select PDF Page", 1, len(pdf_images), 1)

            image = pdf_images[page_no - 1]
        else:
            page_no = 1
            image = Image.open(uploaded_file).convert("RGB")

        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        with col_preprocess:
            card("Image Preprocessing", "🧪")
            denoise_strength = st.slider("Denoise", 0, 30, 5)
            block_size = st.slider("Adaptive Block Size", 15, 61, 21, step=2)
            c_value = st.slider("C Value", 0, 25, 7)

            cA, cB = st.columns(2)
            with cA:
                st.image(image, caption="Original", use_container_width=True)

            processed = preprocess_image(img_cv, denoise_strength, block_size, c_value)
            with cB:
                st.image(processed, caption="Processed", use_container_width=True, clamp=True)

            end_card()

        with col_output:
            card("OCR + Extraction", "🧾")

            use_gemini = st.checkbox("Use Gemini Extraction", value=True)

            col_btn1, col_btn2 = st.columns(2)
            run = col_btn1.button("🚀 Extract & Save", use_container_width=True)
            retry = col_btn2.button("🔁 Retry Gemini", use_container_width=True)

            if retry:
                st.session_state.pop("gemini_cache", None)
                st.warning("✅ Gemini cache cleared. Try Extract again.")

            if run:
                with st.spinner("Running OCR + Extraction..."):
                    raw_text = ocr_extract_text(processed)

                    dup_id = find_duplicate_invoice(raw_text)
                    if dup_id:
                        invoice_id = dup_id
                        st.info(f"✅ Duplicate found: {invoice_id}")
                        st.session_state["last_invoice_id"] = invoice_id
                    else:
                        cache_key = file_cache_key(uploaded_file, page_no)

                        if use_gemini:
                            data = gemini_extract_structured(raw_text, cache_key=cache_key)
                        else:
                            data = fallback_extract(raw_text)

                        data = validate_invoice(data, raw_text)

                        show_validation_calculation(data)

                        if data.get("validation_status") == "VALID":
                            invoice_id = save_to_db(data, raw_text)
                            st.success(f"✅ Saved: {invoice_id}")
                            st.session_state["last_invoice_id"] = invoice_id
                        
                        else:
                            st.warning("⚠ Not valid. Not saved to database.")
                            st.stop()


                        if data.get("validation_status") == "VALID":
                            invoice_id = save_to_db(data, raw_text)
                            st.success(f"✅ Saved: {invoice_id}")
                            st.session_state["last_invoice_id"] = invoice_id

                            # ✅ clear invalid data if any
                            st.session_state.pop("temp_invalid_data", None)

                        else:
                            st.warning("⚠ Not valid. Showing extracted items, but NOT saving to database.")

                            # ✅ Store invalid extraction temporarily for display
                        st.session_state["temp_invalid_data"] = data
                        st.session_state["temp_invalid_raw_text"] = raw_text


            if "last_invoice_id" in st.session_state:
                invoice_id = st.session_state["last_invoice_id"]
                st.markdown("### 📌 Uploaded Invoice Details")

                details_df = fetch_invoice_details(invoice_id)
                st.dataframe(details_df, use_container_width=True, hide_index=True)

                st.markdown("### 🧾 Items List")
                items = fetch_items(invoice_id)

                if items.empty:
                    st.info("No items available.")
                else:
                    items["price"] = pd.to_numeric(items["price"], errors="coerce").fillna(0)
                    items["quantity"] = pd.to_numeric(items["quantity"], errors="coerce").fillna(1)
                    items["line_total"] = items["quantity"] * items["price"]

                    st.dataframe(
                        items.rename(columns={
                            "item_name": "Item",
                            "quantity": "Qty",
                            "price": "Price",
                            "line_total": "Total"
                        }),
                        use_container_width=True,
                        hide_index=True
                    )

            end_card()


# =========================
# PAGE 2: DASHBOARD
# =========================
elif page == "Dashboard":
    st.title("📊 Financial Overview")
    st.caption("Updated live from database (vault.db).")

    df = fetch_invoices_summary()
    if df.empty:
        st.warning("No invoices found yet.")
        st.stop()

    st.subheader("🔍 Filters")
    f1, f2, f3, f4 = st.columns(4)

    vendors = ["All"] + sorted([v for v in df["vendor_name"].fillna("").unique().tolist() if v.strip()])
    categories = ["All"] + CATEGORY_LIST

    vendor_filter = f1.selectbox("Vendor", vendors)
    category_filter = f2.selectbox("Category", categories)

    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    valid_dates = df["date_dt"].dropna()

    min_date = valid_dates.min().date() if len(valid_dates) else datetime.now().date()
    max_date = valid_dates.max().date() if len(valid_dates) else datetime.now().date()

    start_date = f3.date_input("From", value=min_date)
    end_date = f4.date_input("To", value=max_date)

    filtered = df.copy()

    if vendor_filter != "All":
        filtered = filtered[filtered["vendor_name"] == vendor_filter]

    if category_filter != "All":
        filtered = filtered[filtered["category"] == category_filter]

    filtered = filtered[
        (filtered["date_dt"].isna()) |
        ((filtered["date_dt"] >= pd.to_datetime(start_date)) & (filtered["date_dt"] <= pd.to_datetime(end_date)))
    ]

    total_spend = float(filtered["total_amount"].sum()) if not filtered.empty else 0.0
    avg_bill = float(filtered["total_amount"].mean()) if not filtered.empty else 0.0

    if filtered.empty:
        top_cat = "N/A"
    else:
        cat_sum = filtered.groupby("category")["total_amount"].sum().sort_values(ascending=False)
        top_cat = cat_sum.index[0] if len(cat_sum) > 0 else "N/A"

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Tracked Spending", f"₹{total_spend:.2f}")
    c2.metric("Average Bill Value", f"₹{avg_bill:.2f}")
    c3.metric("Top Category", top_cat)

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Spending by Category")
        if filtered.empty:
            st.info("No data found for selected filters.")
        else:
            cat_df = filtered.groupby("category")["total_amount"].sum().reset_index()
            fig = px.pie(cat_df, names="category", values="total_amount", hole=0.5)
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Recent Spending History")
        if filtered.empty:
            st.info("No data found for selected filters.")
        else:
            recent = filtered.sort_values("created_at").tail(15)
            fig2 = px.bar(recent, x="created_at", y="total_amount")
            st.plotly_chart(fig2, use_container_width=True)


# =========================
# PAGE 3: HISTORY
# =========================
elif page == "History":
    st.title("🗂 History")

    df = fetch_invoices_summary()
    if df.empty:
        st.warning("No invoices stored.")
    else:
        st.markdown("### 📌 Stored Summary (click invoice_id row)")
        event = st.dataframe(
            df[["invoice_id", "vendor_name", "date", "total_amount", "created_at", "category"]],
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row"
        )

        if event.selection.rows:
            idx = event.selection.rows[0]
            selected_id = df.iloc[idx]["invoice_id"]

            st.markdown(f"### 🧾 Items for: `{selected_id}`")
            items = fetch_items(selected_id)
            if items.empty:
                st.info("No items available.")
            else:
                items["price"] = pd.to_numeric(items["price"], errors="coerce").fillna(0)
                items["quantity"] = pd.to_numeric(items["quantity"], errors="coerce").fillna(1)
                items["line_total"] = items["quantity"] * items["price"]

                st.dataframe(
                    items.rename(columns={
                        "item_name": "Item",
                        "quantity": "Qty",
                        "price": "Price",
                        "line_total": "Total"
                    }),
                    use_container_width=True,
                    hide_index=True
                )

            st.divider()
            st.markdown("### 🗑 Delete Invoice")
            confirm = st.checkbox("Confirm delete", key="confirm_delete")
            if st.button("❌ Delete Selected Invoice", disabled=not confirm):
                delete_invoice(selected_id)
                st.success(f"✅ Deleted invoice {selected_id}")
                st.rerun()


