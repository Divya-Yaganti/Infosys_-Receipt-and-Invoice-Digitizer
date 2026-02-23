
import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from PIL import Image, ImageOps
import pytesseract
from datetime import datetime
import json
from groq import Groq
import io
from pdf2image import convert_from_bytes # Import pdf2image

# --- CONFIGURATION ---
DB_NAME = 'receipt_vault_v6.db'
GROQ_MODEL = "llama-3.3-70b-versatile"

# --- DATABASE FUNCTIONS ---
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
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

    # Add category column if it doesn't exist (for backward compatibility)
    try:
        c.execute("ALTER TABLE receipts ADD COLUMN category TEXT DEFAULT 'Uncategorized'")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists

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
    conn.commit()
    conn.close()

def check_if_receipt_exists(merchant, date, total, invoice_num):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id, merchant, total_amount, invoice_number FROM receipts WHERE date = ?", (date,))
    candidates = c.fetchall()
    conn.close()

    for row in candidates:
        db_id, db_merch, db_total, db_inv = row
        if invoice_num and invoice_num != "Unknown" and db_inv and db_inv != "Unknown":
            if invoice_num == db_inv:
                return True, 1, db_id

        price_match = abs(db_total - total) <= 0.05
        m1 = merchant.lower().strip()
        m2 = db_merch.lower().strip()
        merchant_match = (m1 in m2 or m2 in m1)

        if price_match and merchant_match:
            return True, 1, db_id

    return False, 0, None

def save_receipt_to_db(data, filename, line_items_data):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("""INSERT INTO receipts
                 (merchant, date, invoice_number, subtotal, tax, total_amount, filename, upload_timestamp, category)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
              (data['merchant'], data['date'], data['invoice_number'],
               data['subtotal'], data['tax'], data['total'], filename, upload_time,
               data.get('category', 'Uncategorized')))
    receipt_id = c.lastrowid
    for item in line_items_data:
        c.execute("INSERT INTO line_items (receipt_id, name, qty, price) VALUES (?, ?, ?, ?)",
                  (receipt_id, item['name'], item['qty'], item['price']))
    conn.commit()
    conn.close()
    return receipt_id

def get_all_receipts():
    conn = sqlite3.connect(DB_NAME)
    try:
        df = pd.read_sql_query("SELECT * FROM receipts ORDER BY date DESC, id DESC", conn)
        # Ensure category column exists in dataframe
        if 'category' not in df.columns:
            df['category'] = 'Uncategorized'
    except:
        df = pd.DataFrame()
    conn.close()
    return df

def get_receipt_by_id(receipt_id):
    conn = sqlite3.connect(DB_NAME)
    try:
        df = pd.read_sql_query("SELECT * FROM receipts WHERE id = ?", conn, params=(receipt_id,))
        # Ensure category column exists in dataframe
        if not df.empty and 'category' not in df.columns:
            df['category'] = 'Uncategorized'
    except:
        df = pd.DataFrame()
    conn.close()
    return df

def get_line_items(receipt_id):
    conn = sqlite3.connect(DB_NAME)
    try:
        df = pd.read_sql_query("SELECT name, qty, price FROM line_items WHERE receipt_id = ?", conn, params=(receipt_id,))
    except:
        df = pd.DataFrame()
    conn.close()
    return df

def get_all_line_items_global():
    conn = sqlite3.connect(DB_NAME)
    try:
        query = """
            SELECT li.name, li.qty, li.price, r.merchant, r.date, r.invoice_number, r.id as receipt_id
            FROM line_items li
            JOIN receipts r ON li.receipt_id = r.id
        """
        df = pd.read_sql_query(query, conn)
    except:
        df = pd.DataFrame()
    conn.close()
    return df

def delete_receipt(receipt_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM line_items WHERE receipt_id = ?", (receipt_id,))
    c.execute("DELETE FROM receipts WHERE id = ?", (receipt_id,))
    conn.commit()
    conn.close()

def clear_database():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM line_items")
    c.execute("DELETE FROM receipts")
    c.execute("DELETE FROM monthly_budgets")
    conn.commit()
    conn.close()

def get_monthly_budget(month_year):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT budget_amount FROM monthly_budgets WHERE month_year = ?", (month_year,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else 0.0

def set_monthly_budget(month_year, amount):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""INSERT INTO monthly_budgets (month_year, budget_amount)
                 VALUES (?, ?)
                 ON CONFLICT(month_year)
                 DO UPDATE SET budget_amount = ?""", (month_year, amount, amount))
    conn.commit()
    conn.close()

def search_receipts_by_keyword(keyword):
    conn = sqlite3.connect(DB_NAME)
    query = """
        SELECT DISTINCT r.*
        FROM receipts r
        LEFT JOIN line_items li ON r.id = li.receipt_id
        WHERE LOWER(r.merchant) LIKE ?
           OR LOWER(r.invoice_number) LIKE ?
           OR LOWER(COALESCE(r.category, '')) LIKE ?
           OR LOWER(li.name) LIKE ?
        ORDER BY r.date DESC, r.id DESC
    """
    keyword_pattern = f"%{keyword.lower()}%"
    df = pd.read_sql_query(query, conn, params=(keyword_pattern, keyword_pattern, keyword_pattern, keyword_pattern))
    # Ensure category column exists
    if not df.empty and 'category' not in df.columns:
        df['category'] = 'Uncategorized'
    conn.close()
    return df

def get_available_months():
    conn = sqlite3.connect(DB_NAME)
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
    conn.close()
    return months

# --- PROCESSING FUNCTIONS ---
def preprocess_image(image):
    return ImageOps.grayscale(image)

def extract_text(image):
    return pytesseract.image_to_string(image)

def convert_pdf_to_images(pdf_file):
    # Convert PDF to a list of PIL images
    return convert_from_bytes(pdf_file.getvalue(), poppler_path='/usr/bin') # Specify poppler path for colab

def parse_with_groq(raw_text, api_key):
    client = Groq(api_key=api_key)
    prompt = f"""
    Extract structured data from this receipt text.
    Return ONLY a JSON object with these keys:
    'merchant', 'date' (YYYY-MM-DD), 'invoice_number', 'subtotal', 'tax', 'total',
    'category' (e.g., Groceries, Dining, Shopping, Transportation, Utilities, Healthcare, Entertainment, Other),
    and 'line_items' (a list of objects with 'name', 'qty', 'price').

    If 'subtotal' is missing but 'total' and 'tax' exist, calculate it.
    If 'tax' is missing, try to infer it or set to 0.
    Try to categorize based on merchant name and items.

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

# --- MAIN APP ---
def main():
    st.set_page_config(page_title="Receipt & Invoice Digitizer", layout="wide", page_icon="🧾")
    init_db()

    if 'current_receipt' not in st.session_state: st.session_state['current_receipt'] = None
    if 'current_line_items' not in st.session_state: st.session_state['current_line_items'] = []
    if 'validation_status' not in st.session_state: st.session_state['validation_status'] = None
    if 'is_key_valid' not in st.session_state: st.session_state['is_key_valid'] = False
    if 'pending_duplicate_save' not in st.session_state: st.session_state['pending_duplicate_save'] = False
    if 'duplicate_conflict_id' not in st.session_state: st.session_state['duplicate_conflict_id'] = None
    if 'last_uploaded_filename' not in st.session_state: st.session_state['last_uploaded_filename'] = ""
    if 'view_receipt_id' not in st.session_state: st.session_state['view_receipt_id'] = None
    if 'delete_confirmation' not in st.session_state: st.session_state['delete_confirmation'] = False

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
        if st.button("Clear Database"):
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
            
            # Handle PDF files
            if uploaded_file.type == "application/pdf":
                images = convert_pdf_to_images(uploaded_file)
                if images:
                    image = images[0]
                    st.info("PDF uploaded. Processing page.")
                else:
                    st.error("Could not convert PDF to image.")
                    return # Stop processing if PDF conversion fails
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
                    with st.spinner("Running OCR & Groq AI Analysis..."):
                        raw_text = extract_text(cleaned_image)
                        structured_data = parse_with_groq(raw_text, user_groq_key)

                        if structured_data:
                            receipt_data = {
                                "merchant": structured_data.get('merchant', 'Unknown'),
                                "date": structured_data.get('date', datetime.now().strftime("%Y-%m-%d")),
                                "invoice_number": structured_data.get('invoice_number', 'Unknown'),
                                "subtotal": float(structured_data.get('subtotal') or 0),
                                "tax": float(structured_data.get('tax') or 0),
                                "total": float(structured_data.get('total') or 0),
                                "category": structured_data.get('category', 'Uncategorized')
                            }
                            line_items = structured_data.get('line_items', [])

                            st.session_state['current_receipt'] = receipt_data
                            st.session_state['current_line_items'] = line_items

                            is_dup, _, conflict_id = check_if_receipt_exists(
                                receipt_data['merchant'], receipt_data['date'],
                                receipt_data['total'], receipt_data['invoice_number']
                            )

                            val_results = validate_receipt(receipt_data, is_dup, conflict_id)
                            st.session_state['validation_status'] = val_results

                            if is_dup:
                                st.session_state['pending_duplicate_save'] = True
                                st.session_state['duplicate_conflict_id'] = conflict_id
                                st.warning(f"⚠️ Duplicate Detected! Matches Vault ID: {conflict_id}")
                            else:
                                st.session_state['pending_duplicate_save'] = False
                                st.session_state['duplicate_conflict_id'] = None
                                new_id = save_receipt_to_db(receipt_data, uploaded_file.name, line_items)
                                st.success(f"Processing Complete! Added to Vault with ID: {new_id}")

                            st.markdown("#### Quick Validation Check")
                            v1, v2, v3 = st.columns(3)

                            sum_ok = val_results['sum_check'][0]
                            sum_help = val_results['sum_check'][2]
                            v1.metric("Sum Logic", "Pass" if sum_ok else "Fail", help=sum_help, delta_color="normal")

                            if val_results['dup'][0]:
                                v2.metric("Duplicate", "None")
                            else:
                                v2.metric("Duplicate", f"ID {conflict_id}")

                            v3.metric("Tax Rate", "OK" if val_results['tax_rate'][0] else "Suspicious")
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
                        st.session_state['pending_duplicate_save'] = False
                        st.success("Forced save successful!")
                        st.rerun()
                with col_view:
                    st.info(f"Go to 'Bill History' and search ID {conflict_id} to compare.")

    # === TAB 2: DETAILED VALIDATION ===
    with tab_validation:
        st.markdown("## Field Extraction & Validation Details")

        if st.session_state['current_receipt']:
            data = st.session_state['current_receipt']
            items = st.session_state['current_line_items']
            val = st.session_state['validation_status']

            c_extract, c_validate, c_db = st.columns(3)
            with c_extract:
                st.info("🔹 Field Extraction")
                with st.container(border=True):
                    st.text_input("Vendor", value=data.get('merchant', ''), disabled=True)
                    st.text_input("Date", value=data.get('date', ''), disabled=True)
                    st.text_input("Invoice #", value=data.get('invoice_number', ''), disabled=True)
                    st.text_input("Category", value=data.get('category', 'Uncategorized'), disabled=True)
                    st.markdown("---")
                    c1, c2 = st.columns(2)
                    c1.text_input("Subtotal", value=f"{data.get('subtotal', 0):.2f}", disabled=True)
                    c2.text_input("Tax", value=f"{data.get('tax', 0):.2f}", disabled=True)
                    st.text_input("Total", value=f"{data.get('total', 0):.2f}", disabled=True)
                    if items:
                        st.dataframe(pd.DataFrame(items), hide_index=True, height=150)
            with c_validate:
                st.info("🔹 Validation Logic")
                if val:
                    res_sum = val['sum_check']
                    if res_sum[0]:
                        st.success(f"**Sum Check**: ✅ {res_sum[1]}")
                    else:
                        st.error(f"**Sum Check**: ❌ {res_sum[1]}")

                    res_dup = val['dup']
                    st.write(f"**Duplicate**: {'✅' if res_dup[0] else '❌'} {res_dup[1]}")

                    res_tax = val['tax_rate']
                    st.write(f"**Tax Rate**: {'✅' if res_tax[0] else '❌'} {res_tax[1]}")
            with c_db:
                st.info("🔹 Vault Status")
                df_all = get_all_receipts()
                if not df_all.empty:
                    st.metric("Total Vault Entries", len(df_all))
                    st.dataframe(df_all[['id', 'merchant', 'total_amount']].head(10), hide_index=True)
        else:
            st.warning("Please upload a document first.")

    # === TAB 3: ENHANCED BILL HISTORY ===
    with tab_history:
        st.header("📜 Comprehensive Invoice Management & Analysis")

        # 1️ Fetch all invoices
        #========================
        df_all_invoices = get_all_receipts()

        if df_all_invoices.empty:
            st.warning("⚠️ No invoices available in the vault yet. Upload some receipts to get started!")
        else:
            st.info(f"📊 Total Invoices in Vault: **{len(df_all_invoices)}**")

            # === 2️ DATA EXPORT SECTION ===
            #===============================
            with st.expander("📂 Export Data (CSV / Excel)", expanded=False):
                st.write("Download your vault data for accounting or external analysis.")

                df_items_export = get_all_line_items_global()

                col_exp1, col_exp2 = st.columns(2)

                # CSV/Excel for Receipts Summary
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

                # CSV/Excel for Line Items
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

            # === 3️ GLOBAL SEARCH ===
            #========================
            st.subheader("🔍 Search & Filter")
            col_search1, col_search2 = st.columns([2, 1])

            with col_search1:
                search_keyword = st.text_input(
                    "Search by Vendor, Invoice ID, Category, or Item Name",
                    placeholder="e.g., Walmart, INV-123, Groceries",
                    key="global_search"
                )

            with col_search2:
                search_by_id = st.number_input("Or Search by Receipt ID", min_value=0, step=1, value=0, key="id_search")

            # === 4️ DATE FILTER ===
            #=========================
            col_date1, col_date2 = st.columns([1, 2])
            with col_date1:
                enable_date_filter = st.checkbox("Filter by Date", value=False)
            with col_date2:
                if enable_date_filter:
                    filter_date = st.date_input("Select Date", value=datetime.now())
                else:
                    filter_date = None

            # === 5️ MONTH-WISE FILTER ===
            #=============================
            available_months = get_available_months()
            if available_months:
                available_months.insert(0, "All Months")
            else:
                available_months = ["All Months"]

            selected_month = st.selectbox("📅 Filter by Month", available_months, key="month_filter")

            st.divider()

            # Apply Filters
            df_filtered = df_all_invoices.copy()

            # Ensure category column exists
            if 'category' not in df_filtered.columns:
                df_filtered['category'] = 'Uncategorized'

            # Search filter
            if search_keyword:
                df_filtered = search_receipts_by_keyword(search_keyword)
                # Re-ensure category column exists after search
                if 'category' not in df_filtered.columns:
                    df_filtered['category'] = 'Uncategorized'

            # ID filter
            if search_by_id > 0:
                df_filtered = df_filtered[df_filtered['id'] == search_by_id]

            # Date filter
            if enable_date_filter and filter_date:
                filter_date_str = filter_date.strftime("%Y-%m-%d")
                df_filtered = df_filtered[df_filtered['date'] == filter_date_str]

            # Month filter (only apply if not "All Months")
            if selected_month and selected_month != "All Months":
                df_filtered['month_year'] = pd.to_datetime(df_filtered['upload_timestamp']).dt.strftime('%Y-%m')
                df_filtered = df_filtered[df_filtered['month_year'] == selected_month]
                # Clean up temporary column
                df_filtered = df_filtered.drop(columns=['month_year'])

            # === 6️ MONTHLY BUDGET MANAGEMENT ===
            #=====================================
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

                # Budget Progress Bar
                if monthly_budget > 0:
                    st.progress(min(monthly_total / monthly_budget, 1.0))

                # Set/Update Budget
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

                # === 7️ CATEGORY-WISE SPENDING ===
                #=================================
                st.subheader("📊 Category-Wise Spending Analysis")

                if 'category' in df_filtered.columns and not df_filtered.empty:
                    category_spending = df_filtered.groupby('category')['total_amount'].sum().reset_index()
                    category_spending.columns = ['Category', 'Total Spend']
                    category_spending = category_spending.sort_values('Total Spend', ascending=False)

                    st.dataframe(category_spending, use_container_width=True, hide_index=True)

                    # === 8️ BUDGET STATUS ===
                    #========================
                    if monthly_budget > 0:
                        if monthly_total <= monthly_budget:
                            st.success(f"✅ You're within budget! Remaining: ${monthly_budget - monthly_total:.2f}")
                        else:
                            overspend = monthly_total - monthly_budget
                            st.warning(f"⚠️ Budget exceeded by ${overspend:.2f}")

                            # === 9 AI SPENDING SUGGESTIONS ===
                            #=================================
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

            # === 10 INTERACTIVE INVOICE TABLE ===
            #===================================
            st.subheader("📋 Invoice List")

            if not df_filtered.empty:
                # Ensure all required columns exist
                required_cols = ['id', 'merchant', 'date', 'total_amount', 'category', 'invoice_number']
                for col in required_cols:
                    if col not in df_filtered.columns:
                        if col == 'category':
                            df_filtered[col] = 'Uncategorized'
                        else:
                            df_filtered[col] = ''

                display_df = df_filtered[required_cols].copy()
                display_df.columns = ['ID', 'Vendor', 'Date', 'Amount', 'Category', 'Invoice #']

                # Render invoice list with a per-row checkbox for selection (single-select enforced)
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
                    # Ensure key exists in session_state
                    if key not in st.session_state:
                        st.session_state[key] = False

                    checked = rcols[0].checkbox("", value=st.session_state.get(key, False), key=key)
                    if checked:
                        # enforce single-selection: clear other checkboxes
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

                    # Get receipt details
                    selected_receipt = get_receipt_by_id(selected_id)

                    if not selected_receipt.empty:
                        receipt_row = selected_receipt.iloc[0]

                        # Display metrics
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

                        # === SHOW LINE ITEMS ===
                        st.subheader("🛒 Itemized Breakdown")
                        line_items_df = get_line_items(selected_id)

                        if not line_items_df.empty:
                            # Safe conversion of numeric columns
                            line_items_df['qty'] = pd.to_numeric(line_items_df['qty'], errors='coerce').fillna(0).astype(int)
                            line_items_df['price'] = pd.to_numeric(line_items_df['price'], errors='coerce').fillna(0.0)
                            line_items_df['Total'] = line_items_df['qty'] * line_items_df['price']

                            line_items_df.columns = ['Item Name', 'Quantity', 'Price', 'Total']

                            st.dataframe(line_items_df, use_container_width=True, hide_index=True)

                            # Export option
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

                        # === DELETE OPTION ===
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

    # === TAB 4: ANALYTICS ===
    with tab_analytics:
        st.header("📊 Spending Analytics")
        df = get_all_receipts()

        if not df.empty:
            df['total_amount'] = pd.to_numeric(df['total_amount'], errors='coerce').fillna(0)
            df['tax'] = pd.to_numeric(df['tax'], errors='coerce').fillna(0)
            df['date_obj'] = pd.to_datetime(df['date'], errors='coerce')
            df_clean = df.dropna(subset=['date_obj']).copy().sort_values('date_obj')

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            total_spend = df['total_amount'].sum()
            avg_ticket = df['total_amount'].mean()
            total_tax = df['tax'].sum()
            recent_date = df_clean['date_obj'].max().strftime('%b %d, %Y') if not df_clean.empty else "N/A"

            kpi1.metric("Total Spend", f"${total_spend:,.2f}")
            kpi2.metric("Avg Receipt", f"${avg_ticket:,.2f}")
            kpi3.metric("Total Tax Paid", f"${total_tax:,.2f}")
            kpi4.metric("Last Purchase", recent_date)

            st.divider()

            st.subheader("🏢 Merchant Breakdown")
            col_a, col_b = st.columns(2)

            with col_a:
                fig_bar = px.bar(df, x='merchant', y='total_amount', color='merchant',
                             title="Total Spend per Vendor",
                             text_auto='.2s')
                st.plotly_chart(fig_bar, use_container_width=True)
                st.markdown("**Insight**: This chart shows how much money has been spent at each vendor. Identifying the top vendors can help in negotiating better deals or optimizing spending habits if certain merchants consume a significant portion of the budget.")

            with col_b:
                fig_pie = px.pie(df, values='total_amount', names='merchant',
                             title="Share of Wallet (Spend %)",
                             hole=0.4)
                st.plotly_chart(fig_pie, use_container_width=True)
                st.markdown("**Insight**: This pie chart illustrates the proportional contribution of each merchant to the total spending. A large slice indicates a dominant vendor, which might be an area to explore for cost-saving opportunities or loyalty program benefits.")

            st.subheader("📅 Time Trends")
            if not df_clean.empty:
                col_c, col_d = st.columns(2)

                with col_c:
                    df_clean['month_year'] = df_clean['date_obj'].dt.strftime('%Y-%m')
                    monthly_spend = df_clean.groupby('month_year')['total_amount'].sum().reset_index()
                    fig_line = px.line(monthly_spend, x='month_year', y='total_amount', markers=True,
                                   title="Monthly Spending Trend",
                                   labels={'month_year': 'Month', 'total_amount': 'Amount ($)'})
                    st.plotly_chart(fig_line, use_container_width=True)
                    st.markdown("**Insight**: This line graph visualizes spending patterns over months. Upward trends could indicate increasing expenses, while downward trends might suggest cost-saving measures or seasonal variations in spending.")

                with col_d:
                    df_clean['cumulative'] = df_clean['total_amount'].cumsum()
                    fig_area = px.area(df_clean, x='date_obj', y='cumulative',
                                   title="Cumulative Spending Over Time",
                                   labels={'date_obj': 'Date', 'cumulative': 'Running Total ($)'})
                    st.plotly_chart(fig_area, use_container_width=True)
                    st.markdown("**Insight**: This area chart displays the total accumulated spending over time. A steep curve indicates rapid spending, whereas a flatter curve suggests more controlled expenditure. This helps in understanding overall financial progression.")

            st.subheader("🧠 Spending Behavior")
            col_e, col_f = st.columns(2)

            with col_e:
                if not df_clean.empty:
                    df_clean['day_name'] = df_clean['date_obj'].dt.day_name()
                    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    df_clean['day_name'] = pd.Categorical(df_clean['day_name'], categories=days_order, ordered=True)
                    day_counts = df_clean.groupby('day_name')['total_amount'].sum().reset_index()

                    fig_day = px.bar(day_counts, x='day_name', y='total_amount',
                                 title="Spending by Day of Week",
                                 color='total_amount', color_continuous_scale='Viridis')
                    st.plotly_chart(fig_day, use_container_width=True)
                    st.markdown("**Insight**: This bar chart reveals spending habits by day of the week. Peaks on certain days might correspond to routine activities like weekend shopping or regular weekly expenses, offering insights into lifestyle spending.")

            with col_f:
                fig_hist = px.histogram(df, x='total_amount', nbins=20,
                                    title="Distribution of Receipt Amounts",
                                    labels={'total_amount': 'Receipt Value ($)'})
                st.plotly_chart(fig_hist, use_container_width=True)
                st.markdown("**Insight**: This histogram shows the frequency distribution of receipt amounts. It helps identify common spending thresholds; for instance, many small transactions versus a few large purchases, which can inform budgeting strategies.")

            st.subheader("🛒 Item Analysis")
            df_items = get_all_line_items_global()
            if not df_items.empty:
                top_items = df_items['name'].value_counts().head(10).reset_index()
                top_items.columns = ['Item Name', 'Count']

                fig_items = px.bar(top_items, x='Count', y='Item Name', orientation='h',
                               title="Top 10 Most Frequent Items",
                               color='Count')
                fig_items.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_items, use_container_width=True)
                st.markdown("**Insight**: This chart highlights the top 10 most frequently purchased items. This can reveal recurring needs or popular products, useful for inventory management if applicable, or for understanding regular consumption patterns.")

        else:
            st.info("No data in vault yet. Upload some receipts to see analytics!")

if __name__ == "__main__":
    main()
