import os
import shutil
import pandas as pd
import fitz  # PyMuPDF
import pytesseract
import sqlite3
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
from fastapi import FastAPI
from collections import deque
import openai
from fastapi.responses import JSONResponse

# Set OpenAI API Key (use env variable for security)
openai.api_key = "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"

app = FastAPI()

# Global variables to store DataFrame, table name, and query history
df = None
table_name = None
query_history = deque(maxlen=5)  # Store last 5 interactions

def save_file_for_processing():
    file_path = input("Enter the path of the file: ").strip()
    if not os.path.isfile(file_path):
        print("Error: The specified file does not exist.")
        return None, None
    
    save_dir = "processed_files"
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.basename(file_path)
    destination_path = os.path.join(save_dir, file_name)
    shutil.copy(file_path, destination_path)
    print(f"File saved for processing at: {destination_path}")
    return destination_path, os.path.splitext(file_name)[0]

def process_file():
    global df, table_name
    file_path, table_name = save_file_for_processing()
    if not file_path:
        return "No file to process."
    ext = os.path.splitext(file_path)[1]
    try:
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif ext in [".csv", ".xlsx"]:
            df = pd.read_csv(file_path, encoding="ISO-8859-1", on_bad_lines="skip") if ext == ".csv" else pd.read_excel(file_path)
            return "File processed successfully."
        elif ext == ".pdf":
            doc = fitz.open(file_path)
            text = " ".join([page.get_text() for page in doc])
            return text
        elif ext in [".jpg", ".png"]:
            text = pytesseract.image_to_string(Image.open(file_path))
            return text
        else:
            return "Unsupported file format"
    except UnicodeDecodeError:
        return "Unicode Error: Unable to read file due to encoding issues."

def query_llm(prompt):
    """
    Uses GPT to generate an SQL query, executes it on the loaded DataFrame,
    and returns the results without displaying the SQL query itself.
    """
    global df, table_name
    if df is None or table_name is None:
        return "No data loaded. Please process a file first."

    # Create an in-memory SQLite database and load the DataFrame
    conn = sqlite3.connect(":memory:")
    df.to_sql(table_name, conn, index=False, if_exists="replace")  # Use dynamic table name

    # Get column names to provide context to GPT
    column_names = ", ".join(df.columns)

    # Get query history as context
    context = "\n".join(query_history)

    # Request only SQL from GPT
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": f"You are an expert SQL assistant. The table is named '{table_name}' and has columns: {column_names}. Only return a valid SQL query. Do NOT explain anything. The output should start with SELECT."},
            {"role": "user", "content": f"Previous Queries: {context}\nGenerate an SQL query for this data. Question: {prompt}"}
        ]
    )

    sql_query = response["choices"][0]["message"]["content"].strip()

    # Ensure GPT returns a valid SQL query
    if not sql_query.lower().startswith("select"):
        return f"Error: GPT generated an invalid SQL query: {sql_query}"

    try:
        result_df = pd.read_sql_query(sql_query, conn)
        query_result = result_df.to_dict(orient="records")
        query_history.append(f"User: {prompt}\nAI: {query_result}")  # Store the query and response
        return query_result
    except Exception as e:
        return f"Error executing SQL: {str(e)}"

@app.post("/query/")
async def query_api(query: str):
    """ FastAPI route to handle user queries. """
    response = query_llm(query)
    
    # Ensure response is always a JSON object
    if isinstance(response, list) or isinstance(response, dict):
        return JSONResponse(content={"response": response})
    else:
        return JSONResponse(content={"error": response})  

def select_best_columns():
    """ Ensures X-axis is categorical and Y-axis is numerical. """
    global df
    if df is None or df.empty:
        return None, None

    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # **Step 1: Pick a categorical column first (5-30 unique values)**
    x_col = None
    if categorical_cols:
        suitable_cat_cols = [col for col in categorical_cols if 5 <= df[col].nunique() <= 30]
        if suitable_cat_cols:
            # Pick the one with the most even distribution (not dominated by a single value)
            x_col = min(suitable_cat_cols, key=lambda col: df[col].value_counts().max())

    # **Step 2: If no categorical column, check for date-related columns**
    if x_col is None:
        for col in df.columns:
            if "date" in col.lower() or "year" in col.lower() or pd.api.types.is_datetime64_any_dtype(df[col]):
                x_col = col
                break

    # **Step 3: If no categorical/date column, use a numerical column with logical groupings**
    if x_col is None and numerical_cols:
        grouped_numerical_cols = [col for col in numerical_cols if df[col].nunique() <= 50]  # Avoid continuous values
        if grouped_numerical_cols:
            x_col = min(grouped_numerical_cols, key=lambda col: df[col].nunique())

    # **Y-axis: Choose the numerical column with the highest variance**
    y_col = max(numerical_cols, key=lambda col: df[col].var(), default=None)

    if x_col and y_col and x_col != y_col:
        return x_col, y_col
    return None, None

def select_best_columns():
    """ Ensures X-axis is categorical and Y-axis is the most relevant numerical column. """
    global df
    if df is None or df.empty:
        return None, None

    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # **Step 1: Prioritize a categorical column for X-axis**
    x_col = None
    if categorical_cols:
        suitable_cat_cols = [col for col in categorical_cols if 5 <= df[col].nunique() <= 30]
        if suitable_cat_cols:
            x_col = min(suitable_cat_cols, key=lambda col: df[col].value_counts().max())

    # **Step 2: If no categorical column, check for date-related columns**
    if x_col is None:
        for col in df.columns:
            if "date" in col.lower() or "year" in col.lower() or pd.api.types.is_datetime64_any_dtype(df[col]):
                x_col = col
                break

    # **Step 3: If no categorical/date column, use a grouped numerical column**
    if x_col is None and numerical_cols:
        grouped_numerical_cols = [col for col in numerical_cols if df[col].nunique() <= 50]
        if grouped_numerical_cols:
            x_col = min(grouped_numerical_cols, key=lambda col: df[col].nunique())

    # **Step 4: Choose the most meaningful numerical column for Y-axis**
    y_col = None
    if numerical_cols:
        # Prioritize sum/count-based metrics (e.g., "total", "count", "sales", "revenue")
        priority_keywords = ["total","release_year", "count", "sum", "sales", "revenue", "profit"]
        priority_cols = [col for col in numerical_cols if any(keyword in col.lower() for keyword in priority_keywords)]
        
        if priority_cols:
            y_col = priority_cols[0]  # Pick the first relevant column
        else:
            # If no priority column, use the column with the highest variance
            y_col = max(numerical_cols, key=lambda col: df[col].var(), default=None)

    if x_col and y_col and x_col != y_col:
        return x_col, y_col
    return None, None

def generate_bar_chart():
    """ Generates a bar chart with auto-selected X and Y axes. """
    global df
    x_col, y_col = select_best_columns()
    
    if not x_col or not y_col:
        print("Error: Could not find appropriate columns for visualization.")
        return None

    plt.figure(figsize=(12, 6))
    plt.bar(df[x_col].astype(str), df[y_col], color='skyblue')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'{y_col} vs {x_col}')
    
    plt.xticks(rotation=45)  # Rotate labels for readability if categorical
    plt.show()  # Display graph
    
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    return base64.b64encode(img_io.read()).decode('utf-8')

@app.post("/visualize/")
async def visualize():
    global df
    if df is None:
        return {"error": "No data available. Please process a file first."}
    
    img_str = generate_bar_chart()
    if img_str:
        return {"image": img_str}
    else:
        return {"error": "Could not generate visualization."}
    
if __name__ == "__main__":
    result = process_file()
    print("Processed File Content:")
    print(result)
    while True:
        user_query = input("Enter your query (or type 'exit' to quit): ").strip()
        if user_query.lower() == "exit":
            print("Exiting program.")
            break
        response = query_llm(user_query)
        print("Query Result:")
        print(response)
        
        print("Generating visualization automatically...")
        generate_bar_chart()
        print("Graph displayed successfully.")
