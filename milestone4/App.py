import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from PIL import Image, ImageOps
import pytesseract
from datetime import datetime, date # Import date
import json
from groq import Groq
import io
from pdf2image import convert_from_bytes
import re
from contextlib import contextmanager

# --- CONFIGURATION ---
DB_NAME = 'receipt_vault_v6.db'
GROQ_MODEL = "llama-3.3-70b-versatile"

# --- TEMPLATE DEFINITIONS (SIMULATED DATABASE) ---
VENDOR_TEMPLATES = {
    "coffee house": {
        "standardized_name": "Coffee House",
        "tax_rate": 0.0825,
        "aliases": ["coffee house inc", "the coffee house", "coffee house llc"]
    },
    "walmart": {
        "standardized_name": "Walmart",
        "tax_rate": None,
        "aliases": ["walmart supercenter", "wm supercenter"]
    },
    "coffee house inc.": {
         "standardized_name": "Coffee House",
         "tax_rate": 0.0825,
         "aliases": []
    }
}

# --- CONNECTION POOLING CONTEXT MANAGER ---
@contextmanager
def get_db_connection():
    """Context manager for database connections with proper error handling"""
    conn = sqlite3.connect(DB_NAME)
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

# --- DATABASE FUNCTIONS (OPTIMIZED) ---
def init_db():
    """Initialize database with optimized indexes"""
    with get_db_connection() as conn:
        c = conn.cursor()

        # Create tables
        c.execute('''CREATE TABLE IF NOT EXISTS receipts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        merchant TEXT,
                        date TEXT,
                        invoice_number TEXT,
                        subtotal REAL,
                        tax REAL,
                        total_amount REAL,
                        filename TEXT,
                        upload_timestamp TEXT,
                        category TEXT DEFAULT 'Uncategorized'
                    )''')

        try:
            c.execute("ALTER TABLE receipts ADD COLUMN category TEXT DEFAULT 'Uncategorized'")
        except sqlite3.OperationalError:
            pass

        c.execute('''CREATE TABLE IF NOT EXISTS line_items (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        receipt_id INTEGER,
                        name TEXT,
                        qty INTEGER,
                        price REAL,
                        FOREIGN KEY (receipt_id) REFERENCES receipts (id)
                    )''')

        c.execute('''CREATE TABLE IF NOT EXISTS monthly_budgets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        month_year TEXT UNIQUE,
                        budget_amount REAL
                    )''')

        # Create indexes for frequently queried columns
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_receipts_date ON receipts(date)",
            "CREATE INDEX IF NOT EXISTS idx_receipts_merchant ON receipts(merchant)",
            "CREATE INDEX IF NOT EXISTS idx_receipts_category ON receipts(category)",
            "CREATE INDEX IF NOT EXISTS idx_receipts_upload_month ON receipts(substr(upload_timestamp, 1, 7))",
            "CREATE INDEX IF NOT EXISTS idx_line_items_receipt_id ON line_items(receipt_id)"
        ]

        for idx in indexes:
            c.execute(idx)

def check_if_receipt_exists(merchant, date, total, invoice_num):
    """Optimized duplicate detection using SQL-level filtering"""
    with get_db_connection() as conn:
        c = conn.cursor()

        # Build SQL query with proper filtering
        query = """
            SELECT id, merchant, total_amount, invoice_number
            FROM receipts
            WHERE date = ?
            AND ABS(total_amount - ?) <= 0.05
        """
        params = [date, total]

        # Add invoice number check if available
        if invoice_num and invoice_num != "Unknown":
            query += " AND invoice_number = ?"
            params.append(invoice_num)

        c.execute(query, params)
        candidates = c.fetchall()

        # Check for matches
        for row in candidates:
            db_id, db_merch, db_total, db_inv = row

            # Invoice number exact match
            if invoice_num and invoice_num != "Unknown" and db_inv and db_inv != "Unknown":
                if invoice_num == db_inv:
                    return True, 1, db_id

            # Merchant name fuzzy match
            m1 = merchant.lower().strip()
            m2 = db_merch.lower().strip()
            merchant_match = (m1 in m2 or m2 in m1)

            if merchant_match:
                return True, 1, db_id

        return False, 0, None

def save_receipt_to_db(data, filename, line_items_data):
    """Optimized save with batch insert for line items"""
    with get_db_connection() as conn:
        c = conn.cursor()
        upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Insert receipt
        c.execute("""INSERT INTO receipts
                     (merchant, date, invoice_number, subtotal, tax, total_amount, filename, upload_timestamp, category)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                  (data['merchant'], data['date'], data['invoice_number'],
                   data['subtotal'], data['tax'], data['total'], filename, upload_time,
                   data.get('category', 'Uncategorized')))
        receipt_id = c.lastrowid

        # Batch insert line items using executemany
        if line_items_data:
            line_items_values = [
                (receipt_id, item['name'], item['qty'], item['price'])
                for item in line_items_data
            ]
            c.executemany(
                "INSERT INTO line_items (receipt_id, name, qty, price) VALUES (?, ?, ?, ?)",
                line_items_values
            )

        return receipt_id

@st.cache_data(ttl=300)  # 5 minute cache
def get_all_receipts():
    """Cached retrieval of all receipts"""
    with get_db_connection() as conn:
        try:
            df = pd.read_sql_query("SELECT * FROM receipts ORDER BY date DESC, id DESC", conn)
            if 'category' not in df.columns:
                df['category'] = 'Uncategorized'
        except:
            df = pd.DataFrame()
    return df

def get_receipt_by_id(receipt_id):
    """Get single receipt by ID"""
    with get_db_connection() as conn:
        try:
            df = pd.read_sql_query("SELECT * FROM receipts WHERE id = ?", conn, params=(receipt_id,))
            if not df.empty and 'category' not in df.columns:
                df['category'] = 'Uncategorized'
        except:
            df = pd.DataFrame()
    return df

@st.cache_data(ttl=60)  # 1 minute cache
def get_line_items(receipt_id):
    """Cached retrieval of line items for a receipt"""
    with get_db_connection() as conn:
        try:
            df = pd.read_sql_query(
                "SELECT name, qty, price FROM line_items WHERE receipt_id = ?",
                conn,
                params=(receipt_id,)
            )
        except:
            df = pd.DataFrame()
    return df

@st.cache_data(ttl=60)  # 1 minute cache
def get_all_line_items_global():
    """Cached retrieval of all line items with receipt details"""
    with get_db_connection() as conn:
        try:
            query = """
                SELECT li.name, li.qty, li.price, r.merchant, r.date, r.invoice_number, r.id as receipt_id
                FROM line_items li
                JOIN receipts r ON li.receipt_id = r.id
            """
            df = pd.read_sql_query(query, conn)
        except:
            df = pd.DataFrame()
    return df

def delete_receipt(receipt_id):
    """Delete receipt and clear cache"""
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute("DELETE FROM line_items WHERE receipt_id = ?", (receipt_id,))
        c.execute("DELETE FROM receipts WHERE id = ?", (receipt_id,))

    # Clear caches
    get_all_receipts.clear()
    get_line_items.clear()
    get_all_line_items_global.clear()
    get_analytics_summary.clear()
    get_time_series_data.clear()

def clear_database():
    """Clear all database tables and caches"""
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute("DELETE FROM line_items")
        c.execute("DELETE FROM receipts")
        c.execute("DELETE FROM monthly_budgets")

    # Clear all caches
    get_all_receipts.clear()
    get_line_items.clear()
    get_all_line_items_global.clear()
    get_analytics_summary.clear()
    get_time_series_data.clear()
    get_monthly_spending_summary.clear()
    get_day_of_week_spending.clear()
    get_top_items.clear()

def get_monthly_budget(month_year):
    """Get budget for a specific month"""
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT budget_amount FROM monthly_budgets WHERE month_year = ?", (month_year,))
        result = c.fetchone()
    return result[0] if result else 0.0

def set_monthly_budget(month_year, amount):
    """Set or update monthly budget"""
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute("""INSERT INTO monthly_budgets (month_year, budget_amount)
                     VALUES (?, ?)
                     ON CONFLICT(month_year)
                     DO UPDATE SET budget_amount = ?""", (month_year, amount, amount))

def get_filtered_receipts(keyword=None, date_filter=None, month_filter=None, receipt_id=None):
    """Optimized filtering using SQL WHERE clauses instead of pandas"""
    with get_db_connection() as conn:
        # Build dynamic SQL query
        base_query = """
            SELECT DISTINCT r.*
            FROM receipts r
            LEFT JOIN line_items li ON r.id = li.receipt_id
            WHERE 1=1
        """
        params = []

        # Add keyword search
        if keyword:
            base_query += """
                AND (LOWER(r.merchant) LIKE ?
                OR LOWER(r.invoice_number) LIKE ?
                OR LOWER(COALESCE(r.category, '')) LIKE ?
                OR LOWER(li.name) LIKE ?)
            """
            keyword_pattern = f"%{keyword.lower()}%"
            params.extend([keyword_pattern] * 4)

        # Add receipt ID filter
        if receipt_id and receipt_id > 0:
            base_query += " AND r.id = ?"
            params.append(receipt_id)

        # Add date filter
        if date_filter:
            base_query += " AND r.date = ?"
            params.append(date_filter)

        # Add month filter
        if month_filter and month_filter != "All Months":
            base_query += " AND substr(r.upload_timestamp, 1, 7) = ?"
            params.append(month_filter)

        base_query += " ORDER BY r.date DESC, r.id DESC"

        try:
            df = pd.read_sql_query(base_query, conn, params=params)
            if not df.empty and 'category' not in df.columns:
                df['category'] = 'Uncategorized'
        except:
            df = pd.DataFrame()

    return df

def search_receipts_by_keyword(keyword):
    """Search receipts using optimized SQL query"""
    return get_filtered_receipts(keyword=keyword)

@st.cache_data(ttl=60)  # 1 minute cache
def get_available_months():
    """Cached retrieval of available months"""
    with get_db_connection() as conn:
        try:
            query = """
                SELECT DISTINCT strftime('%Y-%m', upload_timestamp) as month_year
                FROM receipts
                WHERE upload_timestamp IS NOT NULL
                ORDER BY month_year DESC
            """
            df = pd.read_sql_query(query, conn)
            months = df['month_year'].tolist()
        except:
            months = []
    return months

# --- PRE-AGGREGATED ANALYTICS FUNCTIONS ---
@st.cache_data(ttl=300)  # 5 minute cache
def get_analytics_summary():
    """Pre-aggregated analytics summary for dashboard KPIs"""
    with get_db_connection() as conn:
        query = """
            SELECT
                COUNT(*) as total_receipts,
                SUM(total_amount) as total_spend,
                AVG(total_amount) as avg_ticket,
                SUM(tax) as total_tax,
                MAX(date) as latest_date
            FROM receipts
        """
        result = pd.read_sql_query(query, conn)
    return result.iloc[0] if not result.empty else None

@st.cache_data(ttl=300)  # 5 minute cache
def get_monthly_spending_summary():
    """Pre-aggregated monthly spending by category"""
    with get_db_connection() as conn:
        query = """
            SELECT
                strftime('%Y-%m', date) as month_year,
                category,
                SUM(total_amount) as total_spend,
                COUNT(*) as receipt_count
            FROM receipts
            WHERE date IS NOT NULL
            GROUP BY month_year, category
            ORDER BY month_year DESC, total_spend DESC
        """
        df = pd.read_sql_query(query, conn)
    return df

@st.cache_data(ttl=300)  # 5 minute cache
def get_time_series_data():
    """Pre-aggregated time series data for trend charts"""
    with get_db_connection() as conn:
        query = """
            SELECT
                strftime('%Y-%m', date) as month_year,
                SUM(total_amount) as monthly_total,
                COUNT(*) as receipt_count,
                AVG(total_amount) as avg_receipt
            FROM receipts
            WHERE date IS NOT NULL
            GROUP BY month_year
            ORDER BY month_year ASC
        """
        df = pd.read_sql_query(query, conn)
    return df

@st.cache_data(ttl=300)  # 5 minute cache
def get_day_of_week_spending():
    """Pre-aggregated spending by day of week"""
    with get_db_connection() as conn:
        query = """
            SELECT
                CASE CAST(strftime('%w', date) AS INTEGER)
                    WHEN 0 THEN 'Sunday'
                    WHEN 1 THEN 'Monday'
                    WHEN 2 THEN 'Tuesday'
                    WHEN 3 THEN 'Wednesday'
                    WHEN 4 THEN 'Thursday'
                    WHEN 5 THEN 'Friday'
                    WHEN 6 THEN 'Saturday'
                END as day_name,
                SUM(total_amount) as total_spend,
                COUNT(*) as receipt_count
            FROM receipts
            WHERE date IS NOT NULL
            GROUP BY strftime('%w', date)
        """
        df = pd.read_sql_query(query, conn)
    return df

@st.cache_data(ttl=300)  # 5 minute cache
def get_top_items(limit=10):
    """Pre-aggregated top purchased items"""
    with get_db_connection() as conn:
        query = """
            SELECT
                name as item_name,
                COUNT(*) as purchase_count,
                SUM(qty * price) as total_value
            FROM line_items
            GROUP BY name
            ORDER BY purchase_count DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(limit,))
    return df

@st.cache_data(ttl=300)  # 5 minute cache
def get_merchant_breakdown():
    """Pre-aggregated merchant spending breakdown"""
    with get_db_connection() as conn:
        query = """
            SELECT
                merchant,
                SUM(total_amount) as total_spend,
                COUNT(*) as receipt_count,
                AVG(total_amount) as avg_receipt
            FROM receipts
            GROUP BY merchant
            ORDER BY total_spend DESC
        """
        df = pd.read_sql_query(query, conn)
    return df

# MILESTONE 4: ADVANCED SEARCH & FILTER FUNCTIONS
def advanced_search_receipts(filters):
    """
    MILESTONE 4: Advanced search with multiple filter criteria
    Supports: keyword, date range, amount range, category, merchant
    Uses indexed columns for optimal performance
    """
    with get_db_connection() as conn:

        # Build dynamic query based on filters
        base_query = "SELECT DISTINCT r.* FROM receipts r"
        conditions = []
        params = []

        # Keyword search (searches across multiple fields)
        if filters.get('keyword'):
            keyword_pattern = f"%{filters['keyword'].lower()}%"
            base_query += " LEFT JOIN line_items li ON r.id = li.receipt_id"
            conditions.append("""(LOWER(r.merchant) LIKE ?
                                 OR LOWER(r.invoice_number) LIKE ?
                                 OR LOWER(COALESCE(r.category, '')) LIKE ?
                                 OR LOWER(li.name) LIKE ?) """)
            params.extend([keyword_pattern] * 4)

        # Date range filter
        if filters.get('date_from'):
            conditions.append("r.date >= ?")
            params.append(filters['date_from'])

        if filters.get('date_to'):
            conditions.append("r.date <= ?")
            params.append(filters['date_to'])

        # Amount range filter
        if filters.get('amount_min') is not None:
            conditions.append("r.total_amount >= ?")
            params.append(filters['amount_min'])

        if filters.get('amount_max') is not None:
            conditions.append("r.total_amount <= ?")
            params.append(filters['amount_max'])

        # Category filter
        if filters.get('category') and filters['category'] != 'All':
            conditions.append("r.category = ?")
            params.append(filters['category'])

        # Merchant filter
        if filters.get('merchant') and filters['merchant'] != 'All':
            conditions.append("r.merchant = ?")
            params.append(filters['merchant'])

        # Combine conditions
        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)

        base_query += " ORDER BY r.date DESC, r.id DESC"

        # Execute query
        try:
            df = pd.read_sql_query(base_query, conn, params=params)
            if not df.empty and 'category' not in df.columns:
                df['category'] = 'Uncategorized'
        except Exception as e:
            st.error(f"Search error: {e}")
            df = pd.DataFrame()

    return df

@st.cache_data(ttl=60)
def get_unique_categories():
    """Get list of unique categories for filter dropdown"""
    with get_db_connection() as conn:
        try:
            query = "SELECT DISTINCT category FROM receipts WHERE category IS NOT NULL ORDER BY category"
            df = pd.read_sql_query(query, conn)
            categories = df['category'].tolist()
        except Exception as e:
            st.error(f"Error fetching categories: {e}")
            categories = []
    return ['All'] + categories

@st.cache_data(ttl=60)
def get_unique_merchants():
    """Get list of unique merchants for filter dropdown"""
    with get_db_connection() as conn:
        try:
            query = "SELECT DISTINCT merchant FROM receipts WHERE merchant IS NOT NULL ORDER BY merchant"
            df = pd.read_sql_query(query, conn)
            merchants = df['merchant'].tolist()
        except Exception as e:
            st.error(f"Error fetching merchants: {e}")
            merchants = []
    return ['All'] + merchants

# --- PROCESSING FUNCTIONS ---
def preprocess_image(image):
    return ImageOps.grayscale(image)

def extract_text(image):
    return pytesseract.image_to_string(image)

def convert_pdf_to_images(pdf_file):
    return convert_from_bytes(pdf_file.getvalue(), poppler_path='/usr/bin')

def parse_with_groq(raw_text, api_key):
    client = Groq(api_key=api_key)
    prompt = f"""
    Extract structured data from this receipt text.
    Return ONLY a JSON object with these keys:
    'merchant', 'date' (YYYY-MM-DD), 'invoice_number', 'subtotal', 'tax', 'total',
    'category' (e.g., Groceries, Dining, Shopping, Transportation, Utilities, Healthcare, Entertainment, Other),
    and 'line_items' (a list of objects with 'name', 'qty', 'price').

    Do NOT infer tax if it is not explicitly stated. Set tax to 0 if not found.
    If 'subtotal' is missing but 'total' and 'tax' exist, calculate it.

    Text:
    {raw_text}
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL,
            response_format={"type": "json_object"}
        )
        return json.loads(chat_completion.choices[0].message.content)
    except Exception as e:
        st.error(f"Groq Parsing Error: {e}")
        return None

# --- TEMPLATE PARSING LOGIC ---
def calculate_parsing_score(data):
    """Simulates an accuracy score based on the completeness of critical fields"""
    score = 0
    if data.get('merchant') and data['merchant'] != 'Unknown': score += 25
    if data.get('date'): score += 25
    if data.get('total') and data['total'] > 0: score += 25
    if data.get('tax') and data['tax'] > 0: score += 15
    if data.get('invoice_number') and data['invoice_number'] != 'Unknown': score += 10
    return min(max(score, 0), 100)

def find_matching_template(merchant_name):
    """Finds a template based on merchant name or known aliases"""
    clean_name = merchant_name.lower().strip()

    if clean_name in VENDOR_TEMPLATES:
        return VENDOR_TEMPLATES[clean_name]

    for vendor_key, template_data in VENDOR_TEMPLATES.items():
        if clean_name == vendor_key or clean_name in template_data['aliases']:
            return template_data

    for vendor_key, template_data in VENDOR_TEMPLATES.items():
        if vendor_key in clean_name:
             return template_data

    return None

def apply_template_parsing(standard_data):
    """Takes standard AI-parsed data and refines it using vendor templates"""
    refined_data = standard_data.copy()
    merchant = refined_data.get('merchant', 'Unknown')

    template = find_matching_template(merchant)
    template_applied = False

    if template:
        template_applied = True
        refined_data['merchant'] = template['standardized_name']

        total = refined_data.get('total', 0)
        tax = refined_data.get('tax', 0)
        known_rate = template.get('tax_rate')

        if total > 0 and (tax == 0 or tax is None) and known_rate is not None:
            subtotal = total / (1 + known_rate)
            calculated_tax = total - subtotal

            refined_data['tax'] = round(calculated_tax, 2)
            refined_data['subtotal'] = round(subtotal, 2)
            refined_data['tax_rate_applied'] = known_rate

    return refined_data, template_applied

def get_ai_budget_suggestions(total_spend, budget, category_breakdown, api_key):
    client = Groq(api_key=api_key)
    categories_text = "\n".join([f"- {cat}: ${amt:.2f}" for cat, amt in category_breakdown.items()])
    prompt = f"""
    You are a financial advisor. A user has exceeded their monthly budget.

    Monthly Budget: ${budget:.2f}
    Actual Spending: ${total_spend:.2f}
    Overspend: ${total_spend - budget:.2f}

    Category Breakdown:
    {categories_text}

    Provide 3-5 specific, actionable recommendations to reduce spending and stay within budget.
    Be practical and considerate. Format as a numbered list.
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Unable to generate suggestions: {e}"

# --- VALIDATION LOGIC ---
def validate_receipt(data, is_dup_bool, dup_id=None):
    results = {}

    sub = float(data.get('subtotal', 0))
    tax = float(data.get('tax', 0))
    total = float(data.get('total', 0))

    calculated_total = sub + tax

    if abs(calculated_total - total) <= 0.03:
        status_msg = f"Valid: {sub:.2f} + {tax:.2f} = {total:.2f}"
        results['sum_check'] = (True, status_msg, f"{sub:.2f}+{tax:.2f}={total:.2f}")
    else:
        diff = calculated_total - total
        status_msg = f"Invalid: {sub:.2f} + {tax:.2f} != {total:.2f} (Diff: {diff:.2f})"
        results['sum_check'] = (False, status_msg, f"Exp: {calculated_total:.2f}")

    if not is_dup_bool:
        results['dup'] = (True, f"No duplicate found")
    else:
        results['dup'] = (False, f"Duplicate of Vault ID: {dup_id}")

    if sub > 0:
        rate = (tax / sub) * 100
        if 0 <= rate <= 30:
            results['tax_rate'] = (True, f"Tax Rate: {rate:.1f}% (Normal)")
        else:
            results['tax_rate'] = (False, f"Suspicious Tax Rate: {rate:.1f}%")
    else:
         results['tax_rate'] = (True, "N/A (Subtotal is 0)")

    missing = []
    if data['merchant'] == "Unknown": missing.append("Merchant")
    if not data['date']: missing.append("Date")
    if total == 0.0: missing.append("Total")

    if not missing:
        results['fields'] = (True, "All required fields present")
    else:
        results['fields'] = (False, f"Missing: {', '.join(missing)}")

    return results

# --- CUSTOM CSS FOR UI ---
def local_css():
    st.markdown("""
    <style>
    .parsing-header {
        background-color: #0E68CE;
        color: white;
        padding: 10px;
        border-radius: 5px 5px 0 0;
        font-weight: bold;
        display: flex;
        align-items: center;
    }
    .parsing-header svg { margin-right: 8px; }
    .comparison-container {
        border: 1px solid #ddd;
        border-top: none;
        padding: 15px;
        border-radius: 0 0 5px 5px;
        background-color: #f9f9f9;
    }
    .parsing-box {
        background-color: white;
        border: 1px solid #eee;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .parsing-title {
        font-weight: bold;
        font-size: 1.1em;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .accuracy-badge {
        background-color: #e3f2fd;
        color: #0d47a1;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: bold;
    }
    .field-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
        font-size: 0.95em;
    }
    .field-label { color: #666; }
    .field-value { font-weight: 500; color: #333; }
    .highlight-blue { color: #0E68CE; font-weight: bold; }
    .not-detected { color: #999; font-style: italic; }
    .feature-badges { margin-top: 15px; display: flex; gap: 10px; }
    .feature-badge {
        background-color: #e3f2fd;
        color: #0E68CE;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.85em;
        font-weight: 600;
        display: flex;
        align-items: center;
    }
    .feature-badge svg { margin-right: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- MAIN APP ---
def main():
    st.set_page_config(page_title="Receipt & Invoice Digitizer", layout="wide", page_icon="🧾")
    init_db()
    local_css()

    # Session State Initialization
    if 'current_receipt' not in st.session_state: st.session_state['current_receipt'] = None
    if 'current_line_items' not in st.session_state: st.session_state['current_line_items'] = []
    if 'validation_status' not in st.session_state: st.session_state['validation_status'] = None
    if 'is_key_valid' not in st.session_state: st.session_state['is_key_valid'] = False
    if 'pending_duplicate_save' not in st.session_state: st.session_state['pending_duplicate_save'] = False
    if 'duplicate_conflict_id' not in st.session_state: st.session_state['duplicate_conflict_id'] = None
    if 'last_uploaded_filename' not in st.session_state: st.session_state['last_uploaded_filename'] = ""
    if 'parsing_comparison' not in st.session_state: st.session_state['parsing_comparison'] = None

    with st.sidebar:
        st.header("🔑 API Configuration")
        user_groq_key = st.text_input("Enter Groq API Key", type="password", help="Get your key at console.groq.com")
        if user_groq_key:
            try:
                client = Groq(api_key=user_groq_key)
                client.models.list()
                st.session_state['is_key_valid'] = True
                st.success("✅ Valid API Key", icon="🔐")
            except Exception as e:
                st.session_state['is_key_valid'] = False
                st.error("❌ Invalid API Key", icon="⛔")
        else:
            st.session_state['is_key_valid'] = False

        st.divider()
        st.header("⚙️ Settings")

        # Clear Cache Button
        if st.button("🔄 Clear Cache", help="Clear all cached data to refresh analytics"):
            get_all_receipts.clear()
            get_line_items.clear()
            get_all_line_items_global.clear()
            get_analytics_summary.clear()
            get_time_series_data.clear()
            get_monthly_spending_summary.clear()
            get_day_of_week_spending.clear()
            get_top_items.clear()
            get_merchant_breakdown.clear()
            get_available_months.clear()
            st.toast("Cache cleared!", icon="🔄")
            st.rerun()

        if st.button("🗑️ Clear Database"):
            clear_database()
            st.toast("Database cleared!", icon="🗑️")
            st.rerun()

    st.title("🧾 Receipts & Invoice Digitizer")

    tab_vault, tab_validation, tab_history, tab_analytics = st.tabs(["📤 Upload & Process", "✅ Extraction & Validation", "📜 Bill History", "📊 Analytics"])

    # === TAB 1: UPLOAD & PROCESS ===
    with tab_vault:
        st.markdown("### 1. Document Ingestion")
        uploaded_file = st.file_uploader("Upload Receipt", type=["png", "jpg", "jpeg", "pdf"])

        if uploaded_file:
            st.session_state['last_uploaded_filename'] = uploaded_file.name
            if uploaded_file.type == "application/pdf":
                images = convert_pdf_to_images(uploaded_file)
                if images:
                    image = images[0]
                    st.info("PDF uploaded. Processing page.")
                else:
                    st.error("Could not convert PDF to image.")
                    return
            else:
                image = Image.open(uploaded_file)

            cleaned_image = preprocess_image(image)
            st.subheader("Image Processing")
            col_img1, col_img2 = st.columns(2)
            with col_img1:
                st.image(image, caption="Original Document", use_container_width=True)
            with col_img2:
                st.image(cleaned_image, caption="Cleaned (Grayscale) for OCR", use_container_width=True)

            st.divider()

            if st.button("🚀 Extract & Process", type="primary", use_container_width=True):
                if not st.session_state['is_key_valid']:
                    st.error("Please enter a VALID Groq API Key in the sidebar first.")
                else:
                    with st.spinner("Running OCR, Standard AI Parsing, and Template Matching..."):
                        raw_text = extract_text(cleaned_image)

                        structured_data_standard = parse_with_groq(raw_text, user_groq_key)

                        if structured_data_standard:
                            receipt_data_standard = {
                                "merchant": structured_data_standard.get('merchant', 'Unknown'),
                                "date": structured_data_standard.get('date'),
                                "invoice_number": structured_data_standard.get('invoice_number', 'Unknown'),
                                "subtotal": float(structured_data_standard.get('subtotal') or 0),
                                "tax": float(structured_data_standard.get('tax') or 0),
                                "total": float(structured_data_standard.get('total') or 0),
                                "category": structured_data_standard.get('category', 'Uncategorized')
                            }
                            line_items = structured_data_standard.get('line_items', [])

                            standard_score = calculate_parsing_score(receipt_data_standard)
                            receipt_data_template, template_applied = apply_template_parsing(receipt_data_standard)
                            template_score = calculate_parsing_score(receipt_data_template)

                            st.session_state['parsing_comparison'] = {
                                'standard': receipt_data_standard,
                                'standard_score': standard_score,
                                'template': receipt_data_template,
                                'template_score': template_score,
                                'template_applied': template_applied
                            }

                            final_receipt_data = receipt_data_template
                            if not final_receipt_data['date']:
                                 final_receipt_data['date'] = datetime.now().strftime("%Y-%m-%d")

                            st.session_state['current_receipt'] = final_receipt_data
                            st.session_state['current_line_items'] = line_items

                            is_dup, _, conflict_id = check_if_receipt_exists(
                                final_receipt_data['merchant'], final_receipt_data['date'],
                                final_receipt_data['total'], final_receipt_data['invoice_number']
                            )

                            val_results = validate_receipt(final_receipt_data, is_dup, conflict_id)
                            st.session_state['validation_status'] = val_results

                            if is_dup:
                                st.session_state['pending_duplicate_save'] = True
                                st.session_state['duplicate_conflict_id'] = conflict_id
                                st.warning(f"⚠️ Duplicate Detected! Matches Vault ID: {conflict_id}")
                            else:
                                st.session_state['pending_duplicate_save'] = False
                                st.session_state['duplicate_conflict_id'] = None
                                new_id = save_receipt_to_db(final_receipt_data, uploaded_file.name, line_items)

                                # Clear relevant caches
                                get_all_receipts.clear()
                                get_analytics_summary.clear()
                                get_time_series_data.clear()
                                get_merchant_breakdown.clear()

                                st.success(f"Processing Complete! Added to Vault with ID: {new_id}")
                                st.toast("Receipt processed successfully!", icon="✅")

                        else:
                            st.error("AI could not parse the receipt.")

            if st.session_state.get('pending_duplicate_save'):
                conflict_id = st.session_state.get('duplicate_conflict_id')
                st.error(f"This receipt duplicates Vault ID: {conflict_id} (Same Date, Vendor & Amount).")
                col_force, col_view = st.columns(2)
                with col_force:
                    if st.button("⚠️ Ignore & Force Save"):
                        r_data = st.session_state['current_receipt']
                        l_items = st.session_state['current_line_items']
                        f_name = st.session_state['last_uploaded_filename']
                        save_receipt_to_db(r_data, f_name, l_items)

                        # Clear relevant caches
                        get_all_receipts.clear()
                        get_analytics_summary.clear()

                        st.session_state['pending_duplicate_save'] = False
                        st.success("Forced save successful!")
                        st.rerun()
                with col_view:
                    st.info(f"Go to 'Bill History' and search ID {conflict_id} to compare.")

    # === TAB 2: EXTRACTION & VALIDATION ===
    with tab_validation:
        st.markdown("""
        <div class="parsing-header">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 14.66V20a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h5.34"></path><polygon points="18 2 22 6 12 16 8 16 8 12 18 2"></polygon></svg>
            Template-Based Parsing
        </div>
        """, unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="comparison-container">', unsafe_allow_html=True)

            comparison_data = st.session_state.get('parsing_comparison')

            if comparison_data:
                std = comparison_data['standard']
                tmpl = comparison_data['template']
                std_score = comparison_data['standard_score']
                tmpl_score = comparison_data['template_score']
                score_diff = tmpl_score - std_score

                std_tax_display = f"${std.get('tax', 0):.2f}"
                if std.get('tax') == 0 or std.get('tax') is None:
                    std_tax_display = '<span class="not-detected">Not detected</span>'

                tmpl_tax_val = tmpl.get('tax', 0)
                tmpl_tax_display = f"${tmpl_tax_val:.2f}"
                if tmpl.get('tax_rate_applied'):
                     rate_pct = tmpl['tax_rate_applied'] * 100
                     tmpl_tax_display = f'<span class="highlight-blue">${tmpl_tax_val:.2f} ({rate_pct:.2f}%)</span>'
                elif tmpl_tax_val == 0:
                     tmpl_tax_display = '<span class="not-detected">Not detected</span>'

                c1, c2 = st.columns(2)

                with c1:
                    st.markdown(f"""
                    <div class="parsing-box">
                        <div class="parsing-title">
                            Standard Parsing
                            <span class="accuracy-badge">{std_score}% Accuracy</span>
                        </div>
                        <div class="field-row"><span class="field-label">Date:</span> <span class="field-value">{std.get('date', 'N/A')}</span></div>
                        <div class="field-row"><span class="field-label">Vendor:</span> <span class="field-value">{std.get('merchant', 'Unknown')}</span></div>
                        <div class="field-row"><span class="field-label">Total:</span> <span class="field-value">${std.get('total', 0):.2f}</span></div>
                        <div class="field-row"><span class="field-label">Tax:</span> <span class="field-value">{std_tax_display}</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                with c2:
                     st.markdown(f"""
                    <div class="parsing-box">
                        <div class="parsing-title">
                            Template Parsing
                            <span class="accuracy-badge">{tmpl_score}% Accuracy</span>
                        </div>
                        <div class="field-row"><span class="field-label">Date:</span> <span class="field-value">{tmpl.get('date', 'N/A')}</span></div>
                        <div class="field-row"><span class="field-label">Vendor:</span> <span class="field-value" style="color: #0E68CE;">{tmpl.get('merchant', 'Unknown')}</span></div>
                        <div class="field-row"><span class="field-label">Total:</span> <span class="field-value">${tmpl.get('total', 0):.2f}</span></div>
                        <div class="field-row"><span class="field-label">Tax:</span> <span class="field-value">{tmpl_tax_display}</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                accuracy_badge = ""
                if score_diff > 0:
                    accuracy_badge = f"""
                    <div class="feature-badge">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
                        +{score_diff}% Accuracy
                    </div>
                    """

            else:
                st.info("Upload and process a document to see the template parsing comparison.")

            st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        st.markdown("### 🔍 Validation Details & Database Status")
        if st.session_state['current_receipt']:
            data = st.session_state['current_receipt']
            items = st.session_state['current_line_items']
            val = st.session_state['validation_status']

            c_validate, c_db = st.columns(2)
            with c_validate:
                st.info("🔹 Parsing & Logic Validation")
                if val:
                    res_sum = val['sum_check']
                    st.write(f"**Sum Check**: {'✅' if res_sum[0] else '❌'} {res_sum[1]}")

                    res_dup = val['dup']
                    st.write(f"**Duplicate Detected**: {'❌' if not res_dup[0] else '✅'} {res_dup[1]}")

                    res_tax = val['tax_rate']
                    st.write(f"**Tax Logic**: {'✅' if res_tax[0] else '⚠️'} {res_tax[1]}")

                    res_fields = val['fields']
                    st.write(f"**Required Fields**: {'✅' if res_fields[0] else '⚠️'} {res_fields[1]}")

            with c_db:
                st.info("🔹 Vault Status (Recent Entries)")
                df_all = get_all_receipts()
                if not df_all.empty:
                    st.dataframe(df_all[['id', 'merchant', 'date', 'total_amount']].head(5), hide_index=True, use_container_width=True)
                else:
                    st.write("Vault is empty.")
        else:
             st.warning("Please upload a document first to view validation details.")

    # === TAB 3: ENHANCED BILL HISTORY ===
    with tab_history:
        st.header("📜 Comprehensive Invoice Management & Analysis")

        df_all_invoices = get_all_receipts()

        if df_all_invoices.empty:
            st.warning("⚠️ No invoices available in the vault yet. Upload some receipts to get started!")
        else:
            st.info(f"📊 Total Invoices in Vault: **{len(df_all_invoices)}**")

            with st.expander("📂 Export Data (CSV / Excel)", expanded=False):
                st.write("Download your vault data for accounting or external analysis.")
                df_items_export = get_all_line_items_global()
                col_exp1, col_exp2 = st.columns(2)

                with col_exp1:
                    st.subheader("📑 Receipts Summary")
                    st.caption("One row per receipt (Totals, Dates, Merchants).")
                    csv_receipts = df_all_invoices.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_receipts,
                        file_name=f"receipts_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        key='csv_rec'
                    )
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        df_all_invoices.to_excel(writer, sheet_name='Receipts', index=False)
                    buffer.seek(0)
                    st.download_button(
                        label="📥 Download Excel",
                        data=buffer.getvalue(),
                        file_name=f"receipts_summary_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.ms-excel",
                        key='xls_rec'
                    )

                with col_exp2:
                    st.subheader("🛒 Itemized Details")
                    st.caption("One row per line item (Product Name, Qty, Price + Receipt Info).")
                    if not df_items_export.empty:
                        csv_items = df_items_export.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_items,
                            file_name=f"line_items_detailed_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            key='csv_item'
                        )
                        buffer_items = io.BytesIO()
                        with pd.ExcelWriter(buffer_items, engine='xlsxwriter') as writer:
                            df_items_export.to_excel(writer, sheet_name='Line Items', index=False)
                        buffer_items.seek(0)
                        st.download_button(
                            label="📥 Download Excel",
                            data=buffer_items.getvalue(),
                            file_name=f"line_items_detailed_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.ms-excel",
                            key='xls_item'
                        )
                    else:
                        st.info("No line items available to export.")

            st.divider()

            # ================================================================
            # MILESTONE 4: ADVANCED SEARCH & FILTER SECTION
            # ================================================================
            # Initialize session state for filter application control
            if 'filters_applied' not in st.session_state:
                st.session_state['filters_applied'] = False
            if 'current_filters' not in st.session_state:
                st.session_state['current_filters'] = {}

            with st.expander("🔍 Advanced Search & Filters", expanded=True):
                st.markdown("### Multi-Criteria Search")

                col_s1, col_s2 = st.columns(2)

                with col_s1:
                    # Keyword search
                    search_keyword = st.text_input(
                        "🔎 Keyword Search",
                        placeholder="Search vendor, invoice, category, or item name",
                        key="adv_keyword",
                        value=st.session_state.current_filters.get('keyword', '')
                    )

                    # Date range
                    st.markdown("**📅 Date Range**")
                    col_d1, col_d2 = st.columns(2)
                    date_from_val = None
                    if 'date_from' in st.session_state.current_filters:
                        try:
                            date_from_val = date.fromisoformat(st.session_state.current_filters['date_from'])
                        except ValueError:
                            date_from_val = None
                    with col_d1:
                        date_from = st.date_input("From", value=date_from_val, key="date_from")

                    date_to_val = None
                    if 'date_to' in st.session_state.current_filters:
                        try:
                            date_to_val = date.fromisoformat(st.session_state.current_filters['date_to'])
                        except ValueError:
                            date_to_val = None
                    with col_d2:
                        date_to = st.date_input("To", value=date_to_val, key="date_to")

                with col_s2:
                    # Category filter
                    categories = get_unique_categories()
                    current_category_idx = 0
                    if st.session_state.current_filters.get('category'):
                        try:
                            current_category_idx = categories.index(st.session_state.current_filters['category'])
                        except ValueError:
                            current_category_idx = 0 # Category not found, default to 'All'
                    selected_category = st.selectbox("📁 Category", categories, key="filter_category", index=current_category_idx)

                    # Merchant filter
                    merchants = get_unique_merchants()
                    current_merchant_idx = 0
                    if st.session_state.current_filters.get('merchant'):
                        try:
                            current_merchant_idx = merchants.index(st.session_state.current_filters['merchant'])
                        except ValueError:
                            current_merchant_idx = 0 # Merchant not found, default to 'All'
                    selected_merchant = st.selectbox("🏪 Merchant", merchants, key="filter_merchant", index=current_merchant_idx)

                # Amount range
                st.markdown("**💰 Amount Range**")
                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    amount_min = st.number_input("Min Amount ($", min_value=0.0, value=st.session_state.current_filters.get('amount_min', 0.0), step=10.0, key="amount_min")
                with col_a2:
                    amount_max = st.number_input("Max Amount ($", min_value=0.0, value=st.session_state.current_filters.get('amount_max', 10000.0), step=10.0, key="amount_max")

                # Apply filters button
                col_apply, col_reset = st.columns([1, 1])
                with col_apply:
                    if st.button("🔍 Apply Filters", type="primary", use_container_width=True):
                        st.session_state['filters_applied'] = True
                        # Collect current filter values into session state for persistence and application
                        temp_filters = {}
                        if search_keyword: temp_filters['keyword'] = search_keyword
                        if date_from: temp_filters['date_from'] = date_from.strftime("%Y-%m-%d")
                        if date_to: temp_filters['date_to'] = date_to.strftime("%Y-%m-%d")
                        if amount_min > 0: temp_filters['amount_min'] = amount_min
                        if amount_max < 10000: temp_filters['amount_max'] = amount_max
                        if selected_category != 'All': temp_filters['category'] = selected_category
                        if selected_merchant != 'All': temp_filters['merchant'] = selected_merchant
                        st.session_state['current_filters'] = temp_filters
                        st.rerun() # Rerun to apply filters based on updated session state

                with col_reset:
                    if st.button("🔄 Reset", use_container_width=True):
                        st.session_state['filters_applied'] = False
                        st.session_state['current_filters'] = {} # Clear stored filters
                        st.rerun() # Rerun to reset widgets to default and show all invoices


            # Get filtered results based on filters_applied state
            if st.session_state['filters_applied'] and st.session_state['current_filters']:
                df_filtered = advanced_search_receipts(st.session_state['current_filters'])
                st.success(f"✅ Found {len(df_filtered)} matching receipts")
            else:
                # If no filters applied or reset, show all invoices
                df_filtered = df_all_invoices

            st.divider()

            # Month filter for budget management
            available_months = get_available_months()
            if available_months:
                available_months.insert(0, "All Months")
            else:
                available_months = ["All Months"]
            selected_month = st.selectbox("📅 Filter by Month", available_months, key="month_filter")

            # Apply month filter to the filtered results
            if selected_month != "All Months":
                month_str = selected_month
                df_filtered = df_filtered[df_filtered['upload_timestamp'].str.startswith(month_str)]

            st.divider()


            if selected_month != "All Months":
                st.subheader(f"💰 Budget Management - {selected_month}")
                monthly_total = df_filtered['total_amount'].sum()
                monthly_budget = get_monthly_budget(selected_month)
                col_budget1, col_budget2, col_budget3 = st.columns(3)
                with col_budget1:
                    st.metric("Total Spending", f"${monthly_total:.2f}")
                with col_budget2:
                    st.metric("Monthly Budget", f"${monthly_budget:.2f}" if monthly_budget > 0 else "Not Set")
                with col_budget3:
                    if monthly_budget > 0:
                        budget_usage = (monthly_total / monthly_budget) * 100
                        st.metric("Budget Usage", f"{budget_usage:.1f}%")
                if monthly_budget > 0:
                    st.progress(min(monthly_total / monthly_budget, 1.0))
                with st.expander("⚙️ Set/Update Monthly Budget"):
                    new_budget = st.number_input(
                        f"Budget for {selected_month}",
                        min_value=0.0,
                        value=float(monthly_budget),
                        step=100.0,
                        key="budget_input"
                    )
                    if st.button("💾 Save Budget"):
                        set_monthly_budget(selected_month, new_budget)
                        st.success(f"Budget for {selected_month} set to ${new_budget:.2f}")
                        st.rerun()
                st.divider()

                st.subheader("📊 Category-Wise Spending Analysis")
                if 'category' in df_filtered.columns and not df_filtered.empty:
                    category_spending = df_filtered.groupby('category')['total_amount'].sum().reset_index()
                    category_spending.columns = ['Category', 'Total Spend']
                    category_spending = category_spending.sort_values('Total Spend', ascending=False)
                    st.dataframe(category_spending, use_container_width=True, hide_index=True)

                    if monthly_budget > 0:
                        if monthly_total <= monthly_budget:
                            st.success(f"✅ You're within budget! Remaining: ${monthly_budget - monthly_total:.2f}")
                        else:
                            overspend = monthly_total - monthly_budget
                            st.warning(f"⚠️ Budget exceeded by ${overspend:.2f}")
                            if st.session_state['is_key_valid']:
                                with st.expander("🤖 AI Budget Optimization Suggestions", expanded=True):
                                    if st.button("Generate AI Recommendations"):
                                        with st.spinner("Analyzing your spending patterns..."):
                                            category_dict = category_spending.set_index('Category')['Total Spend'].to_dict()
                                            suggestions = get_ai_budget_suggestions(
                                                monthly_total,
                                                monthly_budget,
                                                category_dict,
                                                user_groq_key
                                            )
                                            st.markdown(suggestions)
                            else:
                                st.info("💡 Enter a valid Groq API key in the sidebar to get AI-powered budget suggestions.")
                st.divider()

            st.subheader("📋 Invoice List")
            if not df_filtered.empty:
                required_cols = ['id', 'merchant', 'date', 'total_amount', 'category', 'invoice_number']
                for col in required_cols:
                    if col not in df_filtered.columns:
                        if col == 'category':
                            df_filtered[col] = 'Uncategorized'
                        else:
                            df_filtered[col] = ''
                display_df = df_filtered[required_cols].copy()
                display_df.columns = ['ID', 'Vendor', 'Date', 'Amount', 'Category', 'Invoice #']

                selected_id = None
                st.markdown("""
                <style>
                .inv-header {font-weight:600; color:#333}
                .inv-cell {color:#444}
                </style>
                """, unsafe_allow_html=True)
                cols_w = [0.6, 0.8, 3.5, 1.2, 1.2, 1.5]
                header_cols = st.columns(cols_w)
                header_cols[0].markdown("**Select**")
                header_cols[1].markdown("**ID**")
                header_cols[2].markdown("**Vendor**")
                header_cols[3].markdown("**Date**")
                header_cols[4].markdown("**Amount**")
                header_cols[5].markdown("**Invoice #**")

                id_list = display_df['ID'].tolist()
                for _, row in display_df.iterrows():
                    rcols = st.columns(cols_w)
                    rid = int(row['ID'])
                    key = f"invoice_chk_{rid}"
                    if key not in st.session_state:
                        st.session_state[key] = False
                    checked = rcols[0].checkbox("", value=st.session_state.get(key, False), key=key)
                    if checked:
                        for other in id_list:
                            ok = f"invoice_chk_{int(other)}"
                            if ok != key and st.session_state.get(ok, False):
                                st.session_state[ok] = False
                        selected_id = rid
                    rcols[1].markdown(f"**{rid}**")
                    rcols[2].markdown(f"{row['Vendor']}")
                    rcols[3].markdown(f"{row['Date']}")
                    rcols[4].markdown(f"{row['Amount']}")
                    rcols[5].markdown(f"{row['Invoice #']}" if pd.notna(row['Invoice #']) else "")

                st.markdown("---")
                if selected_id:
                    st.divider()
                    st.subheader(f"📄 Invoice Details - ID: {selected_id}")
                    selected_receipt = get_receipt_by_id(selected_id)
                    if not selected_receipt.empty:
                        receipt_row = selected_receipt.iloc[0]
                        col_detail1, col_detail2, col_detail3, col_detail4 = st.columns(4)
                        col_detail1.metric("🏪 Vendor", receipt_row['merchant'])
                        col_detail2.metric("💵 Total Amount", f"${receipt_row['total_amount']:.2f}")
                        category_value = receipt_row.get('category', 'Uncategorized')
                        if pd.isna(category_value):
                            category_value = 'Uncategorized'
                        col_detail3.metric("📁 Category", category_value)
                        col_detail4.metric("📅 Date", receipt_row['date'])
                        st.markdown(f"**Invoice Number:** {receipt_row['invoice_number']}")
                        st.markdown(f"**Uploaded:** {receipt_row['upload_timestamp']}")
                        st.divider()

                        st.subheader("🛒 Itemized Breakdown")
                        line_items_df = get_line_items(selected_id)
                        if not line_items_df.empty:
                            line_items_df['qty'] = pd.to_numeric(line_items_df['qty'], errors='coerce').fillna(0).astype(int)
                            line_items_df['price'] = pd.to_numeric(line_items_df['price'], errors='coerce').fillna(0.0)
                            line_items_df['Total'] = line_items_df['qty'] * line_items_df['price']
                            line_items_df.columns = ['Item Name', 'Quantity', 'Price', 'Total']
                            st.dataframe(line_items_df, use_container_width=True, hide_index=True)
                            st.markdown("#### 📤 Export This Invoice")
                            csv_single = line_items_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Items as CSV",
                                data=csv_single,
                                file_name=f"invoice_{selected_id}_items.csv",
                                mime="text/csv",
                                key=f"csv_single_{selected_id}"
                            )
                        else:
                            st.info("No detailed line items found for this invoice.")
                        st.divider()

                        st.markdown("### 🗑️ Delete Invoice")
                        st.warning("⚠️ This action cannot be undone. All related line items will also be deleted.")
                        delete_confirm = st.checkbox(f"I confirm I want to delete Invoice ID: {selected_id}", key=f"del_confirm_{selected_id}")
                        if delete_confirm:
                            if st.button(f"🗑️ Permanently Delete Invoice {selected_id}", type="primary"):
                                delete_receipt(selected_id)
                                st.success(f"✅ Invoice {selected_id} has been permanently deleted.")
                                st.rerun()
            else:
                st.info("No invoices match the current filters.")

    # === TAB 4: ANALYTICS (OPTIMIZED) ===
    with tab_analytics:
        st.header("📊 Spending Analytics")

        # Use pre-aggregated summary
        summary = get_analytics_summary()

        if summary is not None and summary['total_receipts'] > 0:
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)

            total_spend = summary['total_spend']
            avg_ticket = summary['avg_ticket']
            total_tax = summary['total_tax']
            latest_date = summary['latest_date']

            try:
                recent_date = datetime.strptime(latest_date, '%Y-%m-%d').strftime('%b %d, %Y')
            except:
                recent_date = latest_date

            kpi1.metric("Total Spend", f"${total_spend:,.2f}")
            kpi2.metric("Avg Receipt", f"${avg_ticket:,.2f}")
            kpi3.metric("Total Tax Paid", f"${total_tax:,.2f}")
            kpi4.metric("Last Purchase", recent_date)

            st.divider()

            st.subheader("🏢 Merchant Breakdown")
            col_a, col_b = st.columns(2)

            # Use pre-aggregated merchant data
            merchant_df = get_merchant_breakdown()

            with col_a:
                if not merchant_df.empty:
                    fig_bar = px.bar(merchant_df, x='merchant', y='total_spend', color='merchant',
                                 title="Total Spend per Vendor",
                                 text_auto='.2s')
                    st.plotly_chart(fig_bar, use_container_width=True)
                    st.markdown("**Insight**: This chart shows how much money has been spent at each vendor. Identifying the top vendors can help in negotiating better deals or optimizing spending habits if certain merchants consume a significant portion of the budget.")

            with col_b:
                if not merchant_df.empty:
                    fig_pie = px.pie(merchant_df, values='total_spend', names='merchant',
                                 title="Share of Wallet (Spend %)",
                                 hole=0.4)
                    st.plotly_chart(fig_pie, use_container_width=True)
                    st.markdown("**Insight**: This pie chart illustrates the proportional contribution of each merchant to the total spending. A large slice indicates a dominant vendor, which might be an area to explore for cost-saving opportunities or loyalty program benefits.")

            st.subheader("📅 Time Trends")

            # Use pre-aggregated time series data
            time_series_df = get_time_series_data()

            if not time_series_df.empty:
                col_c, col_d = st.columns(2)

                with col_c:
                    fig_line = px.line(time_series_df, x='month_year', y='monthly_total', markers=True,
                                   title="Monthly Spending Trend",
                                   labels={'month_year': 'Month', 'monthly_total': 'Amount ($)'})
                    st.plotly_chart(fig_line, use_container_width=True)
                    st.markdown("**Insight**: This line graph visualizes spending patterns over months. Upward trends could indicate increasing expenses, while downward trends might suggest cost-saving measures or seasonal variations in spending.")

                with col_d:
                    time_series_df['cumulative'] = time_series_df['monthly_total'].cumsum()
                    fig_area = px.area(time_series_df, x='month_year', y='cumulative',
                                   title="Cumulative Spending Over Time",
                                   labels={'month_year': 'Month', 'cumulative': 'Running Total ($)'})
                    st.plotly_chart(fig_area, use_container_width=True)
                    st.markdown("**Insight**: This area chart displays the total accumulated spending over time. A steep curve indicates rapid spending, whereas a flatter curve suggests more controlled expenditure. This helps in understanding overall financial progression.")

            st.subheader("🧠 Spending Behavior")
            col_e, col_f = st.columns(2)

            with col_e:
                # Use pre-aggregated day of week data
                day_df = get_day_of_week_spending()
                if not day_df.empty:
                    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    day_df['day_name'] = pd.Categorical(day_df['day_name'], categories=days_order, ordered=True)
                    day_df = day_df.sort_values('day_name')

                    fig_day = px.bar(day_df, x='day_name', y='total_spend',
                                 title="Spending by Day of Week",
                                 color='total_spend', color_continuous_scale='Viridis')
                    st.plotly_chart(fig_day, use_container_width=True)
                    st.markdown("**Insight**: This bar chart reveals spending habits by day of the week. Peaks on certain days might correspond to routine activities like weekend shopping or regular weekly expenses, offering insights into lifestyle spending.")

            with col_f:
                # For histogram, we still need individual receipt data
                df = get_all_receipts()
                if not df.empty:
                    fig_hist = px.histogram(df, x='total_amount', nbins=20,
                                        title="Distribution of Receipt Amounts",
                                        labels={'total_amount': 'Receipt Value ($)'})
                    st.plotly_chart(fig_hist, use_container_width=True)
                    st.markdown("**Insight**: This histogram shows the frequency distribution of receipt amounts. It helps identify common spending thresholds; for instance, many small transactions versus a few large purchases, which can inform budgeting strategies.")

            st.subheader("🛒 Item Analysis")

            # Use pre-aggregated top items
            top_items_df = get_top_items(10)
            if not top_items_df.empty:
                fig_items = px.bar(top_items_df, x='purchase_count', y='item_name', orientation='h',
                               title="Top 10 Most Frequent Items",
                               color='purchase_count')
                fig_items.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_items, use_container_width=True)
                st.markdown("**Insight**: This chart highlights the top 10 most frequently purchased items. This can reveal recurring needs or popular products, useful for inventory management if applicable, or for understanding regular consumption patterns.")

        else:
            st.info("No data in vault yet. Upload some receipts to see analytics!")

if __name__ == "__main__":
    main()
